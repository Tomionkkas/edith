#!/usr/bin/env python3
"""Train the BPE tokenizer for marvel-slm and measure what it costs.

One tokenizer serves both training stages, so it trains on a 50/50 mix of
Marvel text and general English. Vocabulary size is a real trade-off at this
scale -- the embedding is `vocab x d_model` and is weight-tied, so at 125M
(d=768) a 50k vocab is ~31% of the whole parameter budget -- so this trains
each candidate and reports compression rather than guessing.

  py train/train_tokenizer.py                 # build sample + train 32k and 50k
  py train/train_tokenizer.py --vocab 32000   # just one
  py train/train_tokenizer.py --eval-only     # re-measure existing models
"""
from __future__ import annotations
import argparse
import importlib.util
import random
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CURATED = BASE_DIR / "curated"
OUT_DIR = BASE_DIR / "tokenizer"
SAMPLE = OUT_DIR / "sample.txt"

_spec = importlib.util.spec_from_file_location(
    "tokenizer_corpus", Path(__file__).resolve().parent / "tokenizer_corpus.py")
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

MARVEL_BUDGET = 60_000_000       # chars
GENERAL_BUDGET = 60_000_000
SEED = 1337

# held-out probes, never in the tokenizer sample
MARVEL_PROBE = (
    "Doctor Strange (Earth-199999)\nAlso known as: Doctor Strange\n"
    "First appearance: Doctor Strange (film)\nReality: Earth-199999\n"
    "Peter Parker was bitten by a radioactive spider in Amazing Fantasy Vol 1 15, "
    "created by Stan Lee and Steve Ditko. The Kree-Skrull War and the Infinity "
    "Gauntlet reshaped Earth-616, while Earth-1610 followed the Ultimate universe."
)
GENERAL_PROBE = (
    "The Industrial Revolution began in Britain in the late eighteenth century. "
    "It's often described as the moment when economic growth became self-sustaining, "
    "though historians disagree about why it happened there first rather than in "
    "France or the Netherlands."
)
ENTITY_PROBES = ["Spider-Man", "Earth-616", "Earth-199999", "S.H.I.E.L.D.",
                 "Wolverine", "Galactus", "Wakanda", "Mjolnir"]


def build_sample() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    files = [f for f in sorted(CURATED.glob("*.txt")) if f.stat().st_size > 1000]
    total_bytes = sum(f.stat().st_size for f in files)
    marvel = []
    for f in files:
        share = int(MARVEL_BUDGET * f.stat().st_size / total_bytes)
        if share < 1000:
            continue
        # keep_prob spreads the sample across the file instead of taking a prefix
        keep = min(1.0, share / max(f.stat().st_size, 1) * 3)
        recs, got = tc.take_chars(tc.marvel_records(f), share, rng, keep_prob=keep)
        marvel += recs
        print(f"  marvel  {f.name:<24} {got/1e6:>6.1f} MB  ({len(recs):,} records)")

    print("  reading FineWeb-Edu ...", flush=True)
    general, gtotal = tc.general_docs(GENERAL_BUDGET, rng)
    print(f"  general {'fineweb-edu':<24} {gtotal/1e6:>6.1f} MB  ({len(general):,} docs)")

    merged = tc.interleave(marvel, general, rng)
    n = tc.write_sample(merged, SAMPLE)
    print(f"\nsample: {SAMPLE} — {SAMPLE.stat().st_size/1e6:.1f} MB, {n:,} lines")
    return SAMPLE


# Strings BPE cannot merge on its own, forced into the vocabulary.
#
# BPE learns merges from adjacency frequency, so a period between every letter
# stops them forming: `S.H.I.E.L.D.` costs 12 tokens despite 3,771 occurrences.
# The structural field labels are here for a different reason -- they are the
# schema of every curated record, so making them atomic should help the model
# learn the format, not just save tokens.
#
# Chosen by measured savings (frequency x tokens-saved) over 45 MB of curated
# text; the top 60 cut Marvel token count by 2.05% for 0.12% of the vocabulary.
# Regenerate the ranking with the analysis in CHANGELOG if the corpus changes.
#
# Written with ordinary spaces here for readability. sentencepiece represents a
# space as U+2581 internally, so a multi-word symbol containing a literal space
# lands in the vocabulary and then never matches anything -- the substitution
# happens where these are passed to the trainer.
ENTITY_SYMBOLS = [
    "S.H.I.E.L.D.", "A.I.M.", "M.O.D.O.K.", "U.S.",
    "Spider-Man", "Spider-Woman", "Spider-Girl", "Spider-Verse", "Ant-Man",
    "She-Hulk", "Giant-Man", "Sub-Mariner", "Two-Gun", "Mar-Vell", "Ka-Zar",
    "Shang-Chi", "All-New",
    "Earth-616", "Earth-199999", "Earth-1610",
    "First appearance", "Created by", "Base of operations", "Full name",
    "Identity status", "Also known as", "Marital status", "Affiliation",
    "Other aliases", "Place of birth", "Unusual features", "Relatives",
    "Weaknesses", "Synopsis", "Trivia",
]

# Shared by the real training and by test_tokenizer_roundtrip.py, so the tests
# pin the settings actually used.
#
# The three whitespace flags are NOT optional. sentencepiece defaults normalise
# a language-model corpus into mush: newlines become spaces and repeated spaces
# collapse, which would flatten every curated record into run-on prose and make
# the line-structured format unlearnable.
TRAINER_KWARGS = dict(
    model_type="bpe",
    character_coverage=0.9998,
    byte_fallback=True,                  # no UNK: any byte is representable
    normalization_rule_name="identity",  # keep newlines
    remove_extra_whitespaces=False,      # keep runs of spaces
    add_dummy_prefix=False,              # do not invent a leading space
    user_defined_symbols=["\n"] + [s.replace(" ", "▁") for s in ENTITY_SYMBOLS],
    input_sentence_size=4_000_000,
    shuffle_input_sentence=True,
    pad_id=-1, unk_id=0, bos_id=-1, eos_id=1,
    unk_piece="<unk>", eos_piece="<|endoftext|>",
    num_threads=8,
)


def train(vocab: int) -> Path:
    import sentencepiece as spm
    prefix = OUT_DIR / f"marvel_bpe_{vocab}"
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=str(SAMPLE),
        model_prefix=str(prefix),
        vocab_size=vocab,
        **TRAINER_KWARGS,
    )
    print(f"  trained {vocab} vocab in {(time.time()-t0)/60:.1f} min -> {prefix}.model")
    return Path(f"{prefix}.model")


def evaluate(models: list):
    import sentencepiece as spm
    import tiktoken
    gpt2 = tiktoken.get_encoding("gpt2")

    print(f"\n{'tokenizer':<20}{'vocab':>8}{'marvel':>10}{'general':>10}"
          f"{'chars/tok':>12}{'embed@768':>12}")
    print("-" * 74)

    rows = [("gpt2 (baseline)", gpt2.n_vocab,
             len(gpt2.encode(MARVEL_PROBE)), len(gpt2.encode(GENERAL_PROBE)))]
    sps = {}
    for m in models:
        sp = spm.SentencePieceProcessor(model_file=str(m))
        sps[m.stem] = sp
        rows.append((m.stem, sp.get_piece_size(),
                     len(sp.encode(MARVEL_PROBE)), len(sp.encode(GENERAL_PROBE))))

    chars = len(MARVEL_PROBE) + len(GENERAL_PROBE)
    for name, v, mt, gt in rows:
        print(f"{name:<20}{v:>8,}{mt:>10}{gt:>10}"
              f"{chars/(mt+gt):>12.2f}{v*768/1e6:>10.1f}M")

    print(f"\n{'entity':<18}" + "".join(f"{n.replace('marvel_bpe_',''):>12}"
                                        for n in ["gpt2"] + list(sps)))
    print("-" * (18 + 12 * (1 + len(sps))))
    for e in ENTITY_PROBES:
        cells = [f"{len(gpt2.encode(e)):>12}"]
        cells += [f"{len(sp.encode(e)):>12}" for sp in sps.values()]
        print(f"{e:<18}" + "".join(cells))

    # round-trip check on real text
    print()
    for name, sp in sps.items():
        ok = sp.decode(sp.encode(MARVEL_PROBE)) == MARVEL_PROBE
        print(f"  {name}: round-trip {'OK' if ok else 'MISMATCH'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, action="append",
                    help="repeatable; default trains 32000 and 50257")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    vocabs = args.vocab or [32000, 50257]

    if args.eval_only:
        models = [OUT_DIR / f"marvel_bpe_{v}.model" for v in vocabs]
        evaluate([m for m in models if m.exists()])
        return 0

    if not SAMPLE.exists():
        print("building tokenizer sample ...")
        build_sample()
    else:
        print(f"reusing sample: {SAMPLE} ({SAMPLE.stat().st_size/1e6:.1f} MB)")

    models = [train(v) for v in vocabs]
    evaluate(models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
