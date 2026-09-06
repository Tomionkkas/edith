#!/usr/bin/env python3
"""Marvel SLM - training throughput benchmark.

Measures real tokens/sec for a ~250M GPT on whatever device is present, then
extrapolates to a full pretraining run. Runs on CUDA, Apple MPS, or CPU with
the same code, so numbers from different machines are directly comparable.

  py train/benchmark.py                      # auto device, ~250M, finds max batch
  py train/benchmark.py --params 125         # 125M variant
  py train/benchmark.py --micro-batch 4      # skip the batch-size search
  py train/benchmark.py --no-checkpoint      # disable gradient checkpointing

On the Mac: python3 train/benchmark.py --device mps
"""
from __future__ import annotations
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# (n_layer, n_head, n_embd) -> roughly this many million params with vocab 50257
CONFIGS = {
    125: (12, 12, 768),
    250: (16, 16, 1024),
    350: (24, 16, 1024),
}
VOCAB = 50257
BLOCK = 1024


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.n_head, self.n_embd = n_head, n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cache=None):
        """`cache` holds this layer's past keys and values, for generation.

        Without it every generated token re-attends over the whole prompt from
        scratch: a 300-token prompt producing 60 tokens costs ~20,000
        token-forwards instead of ~360. Training passes cache=None and the
        maths is unchanged.
        """
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        past = 0
        if cache is not None:
            if cache.get("k") is not None:
                past = cache["k"].shape[2]
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
            cache["k"], cache["v"] = k, v

        # is_causal builds its mask from position indices, so it is only
        # correct when queries and keys start at the same place. Decoding one
        # token against N cached keys needs no mask at all: every cached key
        # already precedes the new query.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(past == 0 and T > 1))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln_1(x), cache)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, n_layer, n_head, n_embd):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, n_embd)
        self.wpe = nn.Embedding(BLOCK, n_embd)
        self.blocks = nn.ModuleList([Block(n_head, n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, VOCAB, bias=False)
        self.lm_head.weight = self.wte.weight        # weight tying
        self.use_checkpoint = False

        # Initialisation is NOT optional here. PyTorch defaults give nn.Embedding
        # a std of 1.0, and because lm_head is tied to wte those become the output
        # projection: logits explode and loss starts around 91 instead of
        # ln(50257) = 10.8. Residual projections are additionally scaled by
        # 1/sqrt(2*n_layer) so the residual stream does not grow with depth.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def new_cache(self):
        """One empty key/value slot per layer, for generation."""
        return [{} for _ in self.blocks]

    def logits(self, idx, cache=None):
        B, T = idx.shape
        # With a cache, `idx` is only the NEW tokens, so positions continue
        # from what the cache already holds. Restarting at 0 would give every
        # generated token the position embedding of the first token.
        past = 0
        if cache is not None and cache[0].get("k") is not None:
            past = cache[0]["k"].shape[2]
        if past + T > BLOCK:
            raise ValueError(
                f"sequence {past + T} exceeds block size {BLOCK}; "
                "shorten the prompt or max_new")
        pos = torch.arange(past, past + T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x, None if cache is None else cache[i])
        return self.lm_head(self.ln_f(x))

    def forward(self, idx, targets):
        logits = self.logits(idx)
        return F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def try_step(model, opt, device, micro_batch, amp_dtype, budget=None):
    """One fwd+bwd+step. False if it OOMs *or* spills past `budget` bytes.

    Windows' NVIDIA driver silently falls back to system RAM over PCIe instead
    of raising OOM, so a batch can appear to "fit" while running an order of
    magnitude slower. Peak allocation is checked explicitly against real VRAM.
    """
    try:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, VOCAB, (micro_batch, BLOCK), device=device)
        y = torch.randint(0, VOCAB, (micro_batch, BLOCK), device=device)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype):
            loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if budget and device.startswith("cuda"):
            if torch.cuda.max_memory_allocated() > budget:
                torch.cuda.empty_cache()
                return False
        return True
    except (torch.cuda.OutOfMemoryError if torch.cuda.is_available() else RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        opt.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=int, default=250, choices=sorted(CONFIGS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--micro-batch", type=int, default=0, help="0 = search for max")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--budget-tokens", type=float, default=3.63e9)
    args = ap.parse_args()

    device = pick_device(args.device)
    n_layer, n_head, n_embd = CONFIGS[args.params]
    amp_dtype = torch.bfloat16
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16

    print(f"device      : {device}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"gpu         : {p.name}  {p.total_memory/1e9:.1f} GB")
    print(f"config      : {n_layer}L / {n_head}H / {n_embd}d, block {BLOCK}, vocab {VOCAB}")
    print(f"amp dtype   : {amp_dtype}")

    model = GPT(n_layer, n_head, n_embd).to(device)
    model.use_checkpoint = not args.no_checkpoint
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters  : {n_params/1e6:.1f}M")
    print(f"grad ckpt   : {model.use_checkpoint}")
    budget = None
    if device == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        budget = int(free_b * 0.92)
        print(f"vram free   : {free_b/1e9:.2f} GB of {total_b/1e9:.2f} GB "
              f"(budget {budget/1e9:.2f} GB)")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=(device == "cuda"))

    # find the largest micro-batch that fits
    if args.micro_batch:
        micro = args.micro_batch
        if not try_step(model, opt, device, micro, amp_dtype, budget):
            print(f"micro-batch {micro} does not fit")
            return 1
    else:
        micro = 0
        for candidate in (1, 2, 4, 8, 12, 16, 24, 32):
            if try_step(model, opt, device, candidate, amp_dtype, budget):
                micro = candidate
                print(f"  micro-batch {candidate:>3}: fits")
            else:
                print(f"  micro-batch {candidate:>3}: too big (OOM or spilled to system RAM)")
                break
        if not micro:
            print("nothing fits - try --params 125")
            return 1
    print(f"micro-batch : {micro}  ({micro*BLOCK:,} tokens/step)")

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    for _ in range(args.warmup):
        try_step(model, opt, device, micro, amp_dtype, budget)
    sync()

    t0 = time.time()
    for _ in range(args.steps):
        try_step(model, opt, device, micro, amp_dtype, budget)
    sync()
    dt = time.time() - t0

    tok_per_step = micro * BLOCK
    tps = args.steps * tok_per_step / dt
    print(f"\nstep time   : {dt/args.steps*1000:.0f} ms")
    print(f"throughput  : {tps:,.0f} tokens/sec")
    if device == "cuda":
        print(f"peak VRAM   : {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    # 6*N*D is the standard training-FLOPs approximation
    flops = 6 * n_params * tps
    print(f"achieved    : {flops/1e12:.1f} TFLOP/s")

    print(f"\n--- extrapolation ---")
    for label, toks in (("stage 1 (general)", args.budget_tokens),
                        ("stage 2 (marvel x4 epochs)", 272e6),
                        ("chinchilla 20:1", 20 * n_params)):
        hrs = toks / tps / 3600
        print(f"  {label:<28} {toks/1e9:>6.2f}B tokens -> "
              f"{hrs:>6.1f} h  ({hrs/24:.1f} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

