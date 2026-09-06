"""Talk to a trained checkpoint.

Stage 2 is a continued-pretrain model, not an instruction-tuned one: it
CONTINUES text, it does not answer questions. Test it with a prefix
("Spider-Man is a"), not a question ("who is Spider-Man?"). Question
answering arrives with stage 3.

    py train/sample.py --ckpt checkpoints/stage2/stage2_1epoch_weights.pt
    py train/sample.py --ckpt <path> --prompt "Wolverine is" --n 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / "tokenizer" / "marvel_bpe_50257.model"
EOT = "<|endoftext|>"


def _load_gpt():
    """Same model definition the trainer used; see trainer._load_gpt."""
    spec = importlib.util.spec_from_file_location(
        "benchmark", Path(__file__).resolve().parent / "benchmark.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.GPT, m.CONFIGS, m.BLOCK


# ---------------------------------------------------------------- filters

def filter_logits(logits, top_k: int = 0, top_p: float = 1.0):
    """Mask out tokens we refuse to sample, per row.

    top_k keeps the k highest logits. top_p (nucleus) keeps the smallest set
    of tokens whose cumulative probability first reaches p — the top token is
    always kept, even when it alone already exceeds p, so a row can never be
    filtered down to nothing.
    """
    logits = logits.clone()

    if top_k and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        floor = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < floor, float("-inf"))

    if top_p is not None and top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        drop = cumulative > top_p
        # Shift right: the token that first crosses the threshold is the one
        # that reaches it, so it stays. Without this the nucleus is one token
        # short and a p of 0.9 on a 0.95-probability token empties the row.
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))

    return logits


def apply_repetition_penalty(logits, tokens, penalty: float = 1.0):
    """Discourage tokens already generated (the CTRL convention).

    Negative logits are multiplied rather than divided: dividing -2 by 2
    gives -1, which *raises* the token's odds instead of lowering them.
    """
    if penalty == 1.0 or not len(tokens):
        return logits
    logits = logits.clone()
    idx = torch.tensor(sorted(set(int(t) for t in tokens)),
                       device=logits.device, dtype=torch.long)
    seen = logits[..., idx]
    logits[..., idx] = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits


# ------------------------------------------------------ stop conditions

TERMINATORS = ".?!"


def looping(tokens, window: int = 8, repeats: int = 3) -> bool:
    """True when the tail has started cycling.

    Measured before choosing the threshold: across 74 answers from this
    checkpoint - including greedy sampling and with the repetition penalty
    switched off - the worst legitimate repeat was a five-word phrase
    appearing three times in 300 tokens, and no eight-token sequence ever
    recurred at all. Two occurrences would therefore be a bad threshold:
    "she is an Omega-level mutant" genuinely appears in both the species
    sentence and the powers sentence of the same answer.
    """
    if len(tokens) < window * repeats:
        return False
    tail = tokens[-window:]
    first, seen = tail[0], 0
    for i in range(len(tokens) - window + 1):
        if tokens[i] != first:            # cheap reject before slicing
            continue
        if tokens[i:i + window] == tail:
            seen += 1
            if seen >= repeats:
                return True
    return False


def clip_to_sentence(text: str) -> str:
    """Drop the fragment left behind when generation stopped mid-sentence.

    Stopping at the token cap ended answers on "...matter in ways that". Only
    applied when enough of the answer survives: one full stop near the start
    is not a reason to throw the rest away, and an answer with no sentence
    end at all reads better whole than not at all.
    """
    stripped = text.rstrip()
    if not stripped:
        return text
    last = stripped[-1]
    if last in TERMINATORS or (last in "\"'" and len(stripped) > 1
                               and stripped[-2] in TERMINATORS):
        return text
    cut = max(stripped.rfind(c) for c in TERMINATORS)
    if cut + 1 < len(stripped) / 3:
        return text
    return stripped[:cut + 1]


# --------------------------------------------------------------- sampling

@torch.no_grad()
def generate(model, sp, device, prompt: str, block: int, max_new: int = 120,
             temp: float = 0.8, top_k: int = 50, top_p: float = 0.95,
             repetition_penalty: float = 1.1, stream=None,
             use_cache: bool = True) -> str:
    """Continue `prompt`.

    With the KV cache the prompt is encoded once and each new token attends
    over stored keys. Without it every token re-ran a full forward pass over
    the whole sequence: a 300-token prompt producing 60 tokens cost ~20,000
    token-forwards instead of ~360.
    """
    model.eval()
    ids = sp.encode(prompt)
    if not ids:
        ids = [sp.piece_to_id(EOT)]
    eot = sp.piece_to_id(EOT)

    # The cache cannot exceed the position table, so the prompt yields
    # whatever room the generation needs. Trimming from the LEFT keeps the
    # question and the "Assistant:" tag, which sit at the end.
    room = block - max_new - 1
    if room > 0 and len(ids) > room:
        ids = ids[-room:]

    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated: list[int] = []
    shown = 0

    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())         else torch.float16 if device == "cuda" else torch.float32
    autocast = dict(device_type=device.split(":")[0], dtype=amp,
                    enabled=(device == "cuda"))

    cache = model.new_cache() if use_cache else None
    with torch.autocast(**autocast):
        logits = model.logits(x if cache is not None else x[:, -block:], cache)

    finished = False
    for _ in range(max_new):
        step = logits[:, -1, :].float()
        step = apply_repetition_penalty(step, ids + generated, repetition_penalty)
        step = filter_logits(step / max(temp, 1e-6), top_k, top_p)
        nxt = int(torch.multinomial(torch.softmax(step, dim=-1), 1))
        if nxt == eot:
            finished = True             # the model chose to stop
            break
        generated.append(nxt)
        if looping(generated):
            break
        nxt_t = torch.tensor([[nxt]], device=device)
        x = torch.cat([x, nxt_t], dim=1)

        if stream is not None:
            # decode the whole tail each time: sentencepiece pieces are not
            # independently decodable, so per-token decode mangles spacing
            text = sp.decode(generated)
            stream.write(text[shown:])
            stream.flush()
            shown = len(text)

        with torch.autocast(**autocast):
            # with a cache only the new token is fed; without one the whole
            # sequence is re-run, which is the slow path this replaced
            logits = model.logits(nxt_t if cache is not None else x[:, -block:],
                                  cache)

    text = sp.decode(generated)
    # Only when the model did NOT choose to stop: an answer it ended itself is
    # complete by definition. A streamed answer has already shown the fragment
    # on the way past - the clip is for the returned string.
    return text if finished else clip_to_sentence(text)



def _read_checkpoint(ckpt_path):
    """The state dict, its config and its step, from either published format.

    `.safetensors` is what we publish: torch.load unpickles, so a downloaded
    `.pt` would execute whatever the uploader put in it. The tied head is
    dropped on save (release/to_safetensors.py) and re-tied here.
    """
    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        cfg_path = ckpt_path.parent / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{ckpt_path.name} needs config.json beside it, and there is "
                f"none at {cfg_path}. A wrong architecture loads silently and "
                f"generates noise, so this is an error, not a default.")
        sd = load_file(str(ckpt_path))
        sd["lm_head.weight"] = sd["wte.weight"]      # undo the save-time drop
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return sd, cfg, int(cfg.get("step", 0))

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ck["model"], ck.get("config", {}), int(ck.get("step", 0))


def load_model(ckpt_path, device: str, params: int = 250):
    GPT, CONFIGS, BLOCK = _load_gpt()
    sd, cfg, step = _read_checkpoint(ckpt_path)
    params = int(cfg.get("params", params))
    model = GPT(*CONFIGS[params]).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, BLOCK, step, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--prompt", default=None, help="omit for an interactive loop")
    ap.add_argument("--n", type=int, default=1, help="samples per prompt")
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, block, step, cfg = load_model(Path(args.ckpt), device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{Path(args.ckpt).name}: step {step:,}, {n_params/1e6:.0f}M params, "
          f"stage {cfg.get('stage', '?')}, on {device}\n", flush=True)

    def run(prompt: str):
        for i in range(args.n):
            if args.n > 1:
                print(f"--- sample {i + 1} ---")
            print(prompt, end="", flush=True)
            generate(model, sp, device, prompt, block, args.max_new, args.temp,
                     args.top_k, args.top_p, args.repetition_penalty,
                     stream=sys.stdout)
            print("\n")

    if args.prompt:
        run(args.prompt)
        return 0

    print("Continuation model: give it a PREFIX, not a question.")
    print("Try:  Spider-Man is    |    Wolverine's real name is")
    print("Ctrl-C to quit.\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt:
            run(prompt)


if __name__ == "__main__":
    raise SystemExit(main())
