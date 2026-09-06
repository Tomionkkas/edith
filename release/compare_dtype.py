#!/usr/bin/env python3
"""fp16 or fp32 for the published weights — decided by measurement.

    py release/to_safetensors.py checkpoints/stage3_edith.pt out/fp32
    py release/to_safetensors.py checkpoints/stage3_edith.pt out/fp16 --fp16
    py release/compare_dtype.py out/fp32/model.safetensors out/fp16/model.safetensors

CPU only, on purpose: generate() turns on bf16 autocast for CUDA, which would
blur the very difference this measures. Greedy, on purpose: sampling would
make two runs of one checkpoint disagree.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Narrative questions with no field to compose from - that is what makes
# them exercise the model rather than the renderer. The actual guarantee is
# architectural, not a property of the wording: this script calls
# engine.build_prompt() directly and never engine.try_facts(), and
# build_prompt() has no path into record-composed text, so nothing here
# could report a false "identical" by answering from a field instead of
# the weights regardless of which questions were listed.
PROMPTS = [
    "what happened in civil war",
    "why did tony stark build the iron man armour",
    "what is the infinity gauntlet",
    "how did peter parker get his powers",
    "what caused the mutant registration act",
    "what is the phoenix force",
    "why do the x-men and the brotherhood fight",
    "what happened to gwen stacy",
    "what is the sokovia accords",
    "how did the fantastic four get their powers",
    "what is weapon x",
    "why did wanda create house of m",
    "what is the negative zone",
    "how did bruce banner become the hulk",
    "what happened during secret wars",
    "what is the symbiote",
    "why did thanos collect the infinity stones",
    "what is asgard",
    "how did daredevil lose his sight",
    "what is the sanctum sanctorum",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fp32", help="path to the fp32 model.safetensors")
    ap.add_argument("fp16", help="path to the fp16 model.safetensors")
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    search = _load("search", "retrieve/search.py")
    disambiguate = _load("disambiguate", "retrieve/disambiguate.py")
    facts = _load("facts", "infer/facts.py")
    resolve = _load("resolve", "retrieve/resolve.py")
    sft = _load("sft_data", "train/sft_data.py")
    sample = _load("sample", "train/sample.py")
    engine = _load("engine", "infer/engine.py")

    sp = spm.SentencePieceProcessor(
        model_file=str(ROOT / "tokenizer" / "marvel_bpe_50257.model"))
    index = search.Index.load()

    # Every prompt first, while the index is the only large thing in memory.
    prompts = [engine.build_prompt(q, index, sft, search, disambiguate,
                                   facts, resolve, k=3) for q in PROMPTS]

    def answers(weights):
        """One model at a time. Both models resident at once needs ~2 GB of
        weights on top of the index, and this machine does not reliably have
        it - the session tests already die of WinError 1455 under less. Greedy
        decoding is deterministic and stateless, so running the two models in
        sequence gives exactly the pairs an interleaved loop would."""
        model, block, _, _ = sample.load_model(Path(weights), "cpu")
        kw = dict(max_new=args.max_new, temp=1e-6, top_k=1, top_p=1.0,
                  repetition_penalty=1.0)
        out = [sample.generate(model, sp, "cpu", p, block, **kw).strip()
               for p in prompts]
        del model
        gc.collect()
        return out

    xs = answers(args.fp32)
    ys = answers(args.fp16)

    same = 0
    for q, x, y in zip(PROMPTS, xs, ys):
        ok = x == y
        same += ok
        print("=" * 70)
        print(f"Q: {q}\n{'IDENTICAL' if ok else 'DIVERGED'}")
        # Printed for every prompt, not just DIVERGED ones: a fully-identical
        # run used to produce zero answer text, so "both models wrote the
        # same substantive prose" and "both models silently produced empty
        # strings" printed the same thing. The char count makes a degenerate
        # empty run visible at a glance without reading every line.
        print(f"  fp32 ({len(x)} chars): {x}")
        if not ok:
            print(f"  fp16 ({len(y)} chars): {y}")

    print("=" * 70)
    print(f"{same}/{len(PROMPTS)} identical")
    print("Ship fp16 if every divergence is cosmetic; otherwise ship fp32 "
          "and record the prompt that decided it.")

    # Identical-but-empty is not a passing comparison, it is a comparison
    # that never happened - both sides could be failing to generate at all
    # and every prompt would still show "IDENTICAL". Catch that here rather
    # than let a broken run masquerade as ship-fp16 evidence.
    if all(not x for x in xs) or all(not y for y in ys):
        print("!" * 70)
        print("EVERY fp32 answer was empty, or every fp16 answer was empty: "
              "this run proved nothing about fp16 vs fp32, it only proved "
              "generation is broken. Fix that before trusting any verdict "
              "above.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
