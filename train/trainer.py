#!/usr/bin/env python3
"""Marvel SLM - training loop with checkpoint, resume, and live sampling.

Built so a multi-day run is survivable and observable:

  * checkpoints every `--ckpt-every` steps, resume is automatic
  * validation loss and a *generated sample* printed periodically, so it is
    obvious within the first hour whether the model is learning rather than
    after four days
  * Ctrl-C checkpoints before exiting

Stage 1 (general English), then stage 2 (Marvel) from stage 1's weights:

  py train/trainer.py --stage 1
  py train/trainer.py --stage 2 --init-from checkpoints/stage1/latest.pt

Smoke test first -- 10 minutes, proves the pipeline before any long run:

  py train/trainer.py --stage 2 --smoke
"""
from __future__ import annotations
import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
PACKED = BASE_DIR / "data" / "packed"
CKPT_DIR = BASE_DIR / "checkpoints"
TOKENIZER = BASE_DIR / "tokenizer" / "marvel_bpe_50257.model"

def _load_gpt():
    """Reuse the model definition the benchmark already validated."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "benchmark", Path(__file__).resolve().parent / "benchmark.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.GPT, m.CONFIGS, m.BLOCK


# ------------------------------------------------------------------ data

class DataLoader:
    """Random contiguous windows from a flat uint16 token file."""

    def __init__(self, path, block_size: int, batch_size: int, device: str):
        self.path = Path(path)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.data) < block_size + 1:
            raise ValueError(
                f"{self.path} has {len(self.data)} tokens, "
                f"need at least {block_size + 1}")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def sample_starts(hi: int, n: int) -> np.ndarray:
        """Random window starts in [0, hi).

        dtype=int64 is required, not cosmetic: numpy's randint defaults to
        int32 on Windows, and the general tier has 2.94e9 tokens, so the
        default overflows with "high is out of bounds for int32". The Marvel
        tier (5.5e7) fits, which is why a smoke test on it passes and stage 1
        does not.
        """
        return np.random.randint(0, hi, size=n, dtype=np.int64)

    def get_batch(self):
        hi = len(self.data) - self.block_size - 1
        idx = self.sample_starts(hi, self.batch_size)
        x = np.stack([self.data[i:i + self.block_size] for i in idx])
        y = np.stack([self.data[i + 1:i + 1 + self.block_size] for i in idx])
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        if self.device.startswith("cuda"):
            return x.pin_memory().to(self.device, non_blocking=True), \
                   y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


class MaskedDataLoader(DataLoader):
    """Stage 3: windows plus a loss mask, so only the assistant turn is scored.

    Training on the whole sequence would teach the model to generate its own
    `Context:` blocks, which is the hallucination retrieval exists to prevent.
    """

    def __init__(self, path, mask_path, block_size, batch_size, device):
        super().__init__(path, block_size, batch_size, device)
        self.mask = np.memmap(Path(mask_path), dtype=np.uint8, mode="r")
        if len(self.mask) != len(self.data):
            raise ValueError(
                f"mask/token length mismatch: {len(self.mask)} vs {len(self.data)}")

    def get_batch(self):
        hi = len(self.data) - self.block_size - 1
        idx = self.sample_starts(hi, self.batch_size)
        x = np.stack([self.data[i:i + self.block_size] for i in idx])
        y = np.stack([self.data[i + 1:i + 1 + self.block_size] for i in idx])
        # the mask belongs to the TARGETS, so it shifts with y - otherwise the
        # loss is scored one position off and the whole run is subtly wrong
        m = np.stack([self.mask[i + 1:i + 1 + self.block_size] for i in idx])
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        m = torch.from_numpy(m.astype(np.float32))
        if self.device.startswith("cuda"):
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True),
                    m.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device), m.to(self.device)


class MixedDataLoader(DataLoader):
    """Windows drawn from two corpora, to stop a fine-tune erasing a skill.

    Stage 2 on pure Marvel moved general-English perplexity from 24 to 80 in
    a single epoch: nothing in the loss mentioned English, so nothing held it
    in place. Mixing a fraction of the stage-1 corpus back in ("replay")
    keeps that term alive at the cost of some Marvel perplexity - an easy
    trade here, because Marvel FACTS come from retrieval, not the weights.

    Each sequence is drawn wholly from one corpus. Mixing sources inside a
    window would pair Marvel inputs with English targets, which trains fine
    and teaches nonsense.
    """

    def __init__(self, path, mix_path, block_size, batch_size, device,
                 mix_prob: float = 0.3, rng=None):
        super().__init__(path, block_size, batch_size, device)
        if not 0.0 <= mix_prob <= 1.0:
            raise ValueError(f"mix_prob must be in [0, 1], got {mix_prob}")
        self.mix_prob = mix_prob
        self.mix_path = Path(mix_path)
        self.mix = np.memmap(self.mix_path, dtype=np.uint16, mode="r")
        if len(self.mix) < block_size + 1:
            raise ValueError(
                f"{self.mix_path} has {len(self.mix)} tokens, "
                f"need at least {block_size + 1}")
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_batch(self):
        take_mix = self.rng.random(self.batch_size) < self.mix_prob
        xs, ys = [], []
        for use_mix in take_mix:
            src = self.mix if use_mix else self.data
            hi = len(src) - self.block_size - 1
            # int64: general_train.bin is 2.94e9 tokens; see sample_starts
            i = int(self.rng.integers(0, hi, dtype=np.int64))
            xs.append(src[i:i + self.block_size])
            ys.append(src[i + 1:i + 1 + self.block_size])
        x = torch.from_numpy(np.stack(xs).astype(np.int64))
        y = torch.from_numpy(np.stack(ys).astype(np.int64))
        if self.device.startswith("cuda"):
            return x.pin_memory().to(self.device, non_blocking=True),                    y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


def replay_prob_for(scored_share: float, target_sft_share: float = 0.7) -> float:
    """Sequence probability for the replay corpus, from SCORED-token share.

    Stage 3 masks loss to the assistant turn, so only ~13% of an SFT window
    contributes gradient, while a replay window of ordinary text contributes
    100%. Mixing 30% replay by SEQUENCE therefore puts ~76% of the *scored*
    tokens in the replay corpus - general pretraining with a side of SFT,
    which is the opposite of what stage 3 is for.

    Solving p*s / (p*s + (1-p)) = t for the SFT probability p gives
    p = t / (s*(1-t) + t); the replay probability is 1 - p. At s=0.132 and
    t=0.7 that is 0.054, not 0.30.
    """
    if not 0.0 < scored_share <= 1.0:
        raise ValueError(f"scored_share must be in (0, 1], got {scored_share}")
    if not 0.0 <= target_sft_share < 1.0:
        raise ValueError(f"target_sft_share must be in [0, 1), got {target_sft_share}")
    t = target_sft_share
    p_sft = t / (scored_share * (1.0 - t) + t)
    return 1.0 - p_sft


class MaskedMixedDataLoader(MaskedDataLoader):
    """Stage 3 with replay: masked SFT windows plus general-English windows.

    The SFT windows keep their mask, so only the assistant turn is scored.
    Replay windows are ordinary text with nothing to exclude, so they are
    scored in full.

    As in MixedDataLoader, each sequence is drawn wholly from one corpus:
    splicing SFT inputs onto English targets trains fine and teaches nonsense.
    """

    def __init__(self, path, mask_path, mix_path, block_size, batch_size,
                 device, mix_prob: float = 0.054, rng=None):
        super().__init__(path, mask_path, block_size, batch_size, device)
        if not 0.0 <= mix_prob <= 1.0:
            raise ValueError(f"mix_prob must be in [0, 1], got {mix_prob}")
        self.mix_prob = mix_prob
        self.mix_path = Path(mix_path)
        self.mix = np.memmap(self.mix_path, dtype=np.uint16, mode="r")
        if len(self.mix) < block_size + 1:
            raise ValueError(
                f"{self.mix_path} has {len(self.mix)} tokens, "
                f"need at least {block_size + 1}")
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_batch(self):
        take_mix = self.rng.random(self.batch_size) < self.mix_prob
        xs, ys, ms = [], [], []
        for use_mix in take_mix:
            src = self.mix if use_mix else self.data
            hi = len(src) - self.block_size - 1
            # int64: general_train.bin is 2.94e9 tokens; see sample_starts
            i = int(self.rng.integers(0, hi, dtype=np.int64))
            xs.append(src[i:i + self.block_size])
            ys.append(src[i + 1:i + 1 + self.block_size])
            if use_mix:
                ms.append(np.ones(self.block_size, dtype=np.uint8))
            else:
                # the mask belongs to the TARGETS, so it shifts with y
                ms.append(self.mask[i + 1:i + 1 + self.block_size])
        x = torch.from_numpy(np.stack(xs).astype(np.int64))
        y = torch.from_numpy(np.stack(ys).astype(np.int64))
        m = torch.from_numpy(np.stack(ms).astype(np.float32))
        if self.device.startswith("cuda"):
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True),
                    m.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device), m.to(self.device)



def masked_loss(logits, targets, mask):
    """Cross-entropy averaged over unmasked positions only."""
    v = logits.size(-1)
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, v).float(), targets.reshape(-1), reduction="none")
    m = mask.reshape(-1).to(per_token.dtype)
    denom = m.sum()
    if float(denom) == 0.0:            # a window of pure prompt contributes nothing
        return (per_token * m).sum() * 0.0
    return (per_token * m).sum() / denom


# -------------------------------------------------------------- schedule

def lr_at(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup, then cosine decay to `min_lr`, then flat."""
    if step < warmup:
        return lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


# ------------------------------------------------------------ checkpoints

def save_checkpoint(path, model, opt, step: int, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "config": config}, tmp)
    tmp.replace(path)                    # atomic: never a half-written checkpoint


class BestTracker:
    """Tracks the lowest validation loss seen.

    Only `latest.pt` is kept otherwise, so a run that starts overfitting near
    the end overwrites its own best weights. Stage 2 does 4 epochs over 55M
    tokens with a 250M model, where that is a real possibility.
    """

    def __init__(self):
        self.best = float("inf")

    def improved(self, value: float) -> bool:
        if value < self.best:
            self.best = value
            return True
        return False


def load_checkpoint(path, model, opt, device: str):
    path = Path(path)
    if not path.exists():
        return 0, {}
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if opt is not None and ck.get("optimizer"):
        opt.load_state_dict(ck["optimizer"])
    return int(ck.get("step", 0)), ck.get("config", {})


# --------------------------------------------------------------- sampling

@torch.no_grad()
def sample(model, sp, device, prompt: str, max_new: int = 90, temp: float = 0.8):
    model.eval()
    ids = sp.encode(prompt) or [sp.piece_to_id("<|endoftext|>")]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits = model.logits(x[:, -1024:])
        probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
    model.train()
    return sp.decode(x[0].tolist())


@torch.no_grad()
def estimate_loss(model, loader, iters, device, amp, masked=False):
    model.eval()
    total = 0.0
    for _ in range(iters):
        batch = loader.get_batch()
        with torch.autocast(device_type=device.split(":")[0], dtype=amp):
            if masked:
                x, y, m = batch
                total += float(masked_loss(model.logits(x), y, m))
            else:
                x, y = batch
                total += float(model(x, y))
    model.train()
    return total / iters


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--params", type=int, default=250)
    ap.add_argument("--device", default=None)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = derive from tokens")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--smoke", action="store_true", help="short run to prove the pipeline")
    ap.add_argument("--run-name", default=None,
                    help="checkpoint subdirectory; defaults to stage<N>. "
                         "Use a fresh name to avoid resuming another run.")
    ap.add_argument("--mix-bin", default=None,
                    help="replay corpus mixed into training, e.g. "
                         "data/packed/general_train.bin")
    ap.add_argument("--mix-prob", type=float, default=0.3,
                    help="fraction of sequences drawn from --mix-bin")
    ap.add_argument("--watch-bin", default=None,
                    help="score this corpus at every eval as an early warning. "
                         "A narrow fine-tune erases what it does not train on, and "
                         "the run's own val set cannot see that happening.")
    args = ap.parse_args()

    GPT, CONFIGS, BLOCK = _load_gpt()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        amp = torch.float16

    tier = {1: "general", 2: "marvel", 3: "sft"}[args.stage]
    train_bin = PACKED / f"{tier}_train.bin"
    val_bin = PACKED / f"{tier}_val.bin"
    if not train_bin.exists():
        raise SystemExit(f"missing {train_bin} â€” run: py train/pack.py --{tier}")

    if args.smoke:
        args.max_steps, args.eval_every, args.ckpt_every = 200, 50, 100
        args.warmup = 20

    run_dir = CKPT_DIR / (args.run_name or f"stage{args.stage}")
    latest = run_dir / "latest.pt"

    if args.stage == 3 and args.mix_bin:
        # Stage 3 masks loss to the assistant turn, so an SFT window
        # contributes ~13% of its tokens while a replay window contributes
        # 100%. mix_prob is a SEQUENCE probability: use replay_prob_for() to
        # pick one from the scored-token split you actually want.
        train_dl = MaskedMixedDataLoader(
            train_bin, PACKED / "sft_train_mask.bin", args.mix_bin,
            BLOCK, args.micro_batch, device, args.mix_prob)
        val_dl = MaskedDataLoader(val_bin, PACKED / "sft_val_mask.bin",
                                  BLOCK, args.micro_batch, device)
        print(f"replay: {args.mix_prob:.1%} of sequences from {args.mix_bin}")
    elif args.stage == 3:
        # stage 3 scores only the assistant turn; see MaskedDataLoader
        train_dl = MaskedDataLoader(train_bin, PACKED / "sft_train_mask.bin",
                                    BLOCK, args.micro_batch, device)
        val_dl = MaskedDataLoader(val_bin, PACKED / "sft_val_mask.bin",
                                  BLOCK, args.micro_batch, device)
    elif args.mix_bin:
        # replay: keep general English in the loss so stage 2 stops erasing it
        train_dl = MixedDataLoader(train_bin, args.mix_bin, BLOCK,
                                   args.micro_batch, device, args.mix_prob)
        # validation stays PURE target-domain, so the number remains
        # comparable with every run that came before it
        val_dl = DataLoader(val_bin, BLOCK, args.micro_batch, device)
        print(f"replay: {args.mix_prob:.0%} of sequences from {args.mix_bin}")
    else:
        train_dl = DataLoader(train_bin, BLOCK, args.micro_batch, device)
        val_dl = DataLoader(val_bin, BLOCK, args.micro_batch, device)


    # Stage 2 drove general-English perplexity from 24 to 162 while its own
    # Marvel val loss improved throughout: a run's validation set is drawn
    # from its training distribution and structurally cannot see the damage.
    watch_dl = DataLoader(args.watch_bin, BLOCK, args.micro_batch,
                          device) if args.watch_bin else None
    if watch_dl:
        print(f"watching {args.watch_bin} for forgetting")

    n_layer, n_head, n_embd = CONFIGS[args.params]
    model = GPT(n_layer, n_head, n_embd).to(device)
    model.use_checkpoint = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=(device == "cuda"))

    step = 0
    if latest.exists():
        step, _ = load_checkpoint(latest, model, opt, device)
        print(f"resumed from {latest} at step {step}")
    elif args.init_from:
        load_checkpoint(args.init_from, model, None, device)
        print(f"initialised weights from {args.init_from} (fresh optimizer)")

    tokens_per_step = args.micro_batch * args.grad_accum * BLOCK
    if not args.max_steps:
        args.max_steps = len(train_dl) // tokens_per_step
    min_lr = args.lr / 10

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))

    print(f"stage {args.stage} ({tier}) | {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"train {len(train_dl):,} tokens | val {len(val_dl):,} tokens")
    print(f"{tokens_per_step:,} tokens/step | {args.max_steps:,} steps "
          f"| {args.max_steps*tokens_per_step/1e9:.2f}B tokens total")
    print(f"checkpoints -> {run_dir}\n", flush=True)

    stopping = {"flag": False}

    def _stop(sig, frame):
        stopping["flag"] = True
        print("\nstop requested â€” checkpointing before exit ...", flush=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    best = BestTracker()
    t0 = time.time()
    # steps completed before this process started; throughput and ETA must be
    # measured against work done in THIS session, or every resume reports a
    # wildly inflated rate (all steps divided by seconds since relaunch).
    start_step = step
    while step < args.max_steps and not stopping["flag"]:
        lr = lr_at(step, args.warmup, args.max_steps, args.lr, min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        # accumulate the true mean loss; each micro-batch contributes 1/accum
        running_loss = 0.0
        for micro in range(args.grad_accum):
            batch = train_dl.get_batch()
            with torch.autocast(device_type=device.split(":")[0], dtype=amp):
                if args.stage == 3:
                    x, y, m = batch
                    loss = masked_loss(model.logits(x), y, m) / args.grad_accum
                else:
                    x, y = batch
                    loss = model(x, y) / args.grad_accum
            running_loss += float(loss.detach())
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 10 == 0:
            el = time.time() - t0
            tps = (step - start_step) * tokens_per_step / max(el, 1e-9)
            print(f"step {step:>7,}/{args.max_steps:,}  loss {running_loss:.3f}  "
                  f"lr {lr:.2e}  {tps:,.0f} tok/s  "
                  f"eta {(args.max_steps-step)*tokens_per_step/tps/3600:.1f}h", flush=True)

        if step % args.eval_every == 0:
            vl = estimate_loss(model, val_dl, 20, device, amp, args.stage == 3)
            print(f"  >> val loss {vl:.3f}  (perplexity {math.exp(min(vl,20)):.1f})", flush=True)
            if watch_dl is not None:
                # full loss, not masked: this corpus has no prompt to exclude
                wl = estimate_loss(model, watch_dl, 10, device, amp, False)
                print(f"  >> watch loss {wl:.3f}  (perplexity {math.exp(min(wl,20)):.1f})", flush=True)
            if best.improved(vl):
                save_checkpoint(run_dir / "best.pt", model, opt, step,
                                {"params": args.params, "stage": args.stage,
                                 "val_loss": vl})
                print(f"  >> new best ({vl:.3f}) -> best.pt", flush=True)
            txt = sample(model, sp, device, "Spider-Man" if args.stage == 2 else "The")
            print(f"  >> sample: {txt[:300]!r}\n", flush=True)

        if step % args.ckpt_every == 0:
            save_checkpoint(latest, model, opt, step,
                            {"params": args.params, "stage": args.stage})
            print(f"  >> checkpoint at step {step}", flush=True)

    save_checkpoint(latest, model, opt, step, {"params": args.params, "stage": args.stage})
    print(f"\nstopped at step {step}. checkpoint: {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())





