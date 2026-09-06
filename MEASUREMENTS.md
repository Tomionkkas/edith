# Measurements

Every number below was re-run on 2026-09-06, on this machine, on the
checkpoints and index that ship. Where a figure is a training-time record
that cannot be re-run today (the replay ratio probe), it is labelled as
such, dated, and left unchanged from `ROADMAP.md`.

**A note on today's conditions.** The GPU was in use for something else
tonight, so every command below ran on CPU
(`CUDA_VISIBLE_DEVICES=-1`, which is the only value that actually hides the
GPU on this machine — an empty string does not). This is slower but should
not change what a number *means*; where a CPU re-run disagrees with a
previously recorded GPU figure by more than noise, that is called out rather
than silently reconciled.

## 1. How to reproduce every number here

```bash
py train/eval_ppl.py --ckpt checkpoints/stage3_edith.pt --bin data/packed/general_val.bin --windows 200
py train/eval_ppl.py --ckpt checkpoints/stage3_edith.pt --bin data/packed/marvel_val.bin --windows 200
py train/eval_ppl.py --ckpt checkpoints/stage2_edith.pt --bin data/packed/general_val.bin --windows 200
py train/eval_ppl.py --ckpt checkpoints/stage2_edith.pt --bin data/packed/marvel_val.bin --windows 200
py infer/evaluate.py --n 300
py infer/run_cases.py
py -m pytest -q --ignore=infer/test_terminal_session.py
py -m pytest -q infer/test_terminal_session.py
```

(The suite is two commands, not one — a single combined run exhausts the
Windows paging file on this machine; see §8.)

On a GPU these run in minutes; on CPU tonight they were noticeably slower
across the board (not independently timed with a stopwatch, so no specific
duration is claimed here beyond "plan for longer than the brief's GPU-sized
estimates"), except the two `pytest` invocations, which reported their own
wall-clock and finished in under a minute combined — see §8. Check the
numbers yourself before trusting them.

## 2. The model

250M parameters: 16 layers, 16 heads, `d_model=1024`, tied embeddings,
block size 1,024, vocabulary 50,257 (`marvel_bpe_50257`, a SentencePiece BPE
trained on the corpus). Trained in two stages — general-English pretrain,
then Marvel continued-pretrain — followed by instruction-tuning (SFT).
Retrained from scratch on 2026-09-04 on a rented RTX 4090, roughly 20 hours
end to end, about $8 at $0.32/hr (`docs/HARDWARE.md`).

## 3. Perplexity, both validation sets, both stages

Fixed 200-window evaluation (`train/eval_ppl.py`), so runs are directly
comparable to each other and to prior recordings.

| checkpoint | step | general_val | marvel_val |
|---|---:|---:|---:|
| stage 2 (`stage2_edith.pt`) | 6,764 | 32.72 | 12.24 |
| stage 3 (`stage3_edith.pt`) | 2,500 | 38.22 | 18.59 |

Measured twice, on CPU and on a local RTX 4060, because a first CPU-only run
raised a question about device sensitivity. The two agree to within 0.01 on
all four figures, so the table above is simply what this script prints. GPU
raw output:

```
stage3_edith.pt on general_val.bin: loss 3.6434  perplexity 38.22  (200 fixed windows, step 2,500)
stage3_edith.pt on marvel_val.bin:  loss 2.9225  perplexity 18.59  (200 fixed windows, step 2,500)
stage2_edith.pt on general_val.bin: loss 3.4880  perplexity 32.72  (200 fixed windows, step 6,764)
stage2_edith.pt on marvel_val.bin:  loss 2.5047  perplexity 12.24  (200 fixed windows, step 6,764)
```

CPU raw output, for comparison:

```
stage2_edith.pt on general_val.bin: loss 3.4877  perplexity 32.71  (200 fixed windows, step 6,764)
stage2_edith.pt on marvel_val.bin:  loss 2.5044  perplexity 12.24  (200 fixed windows, step 6,764)
stage3_edith.pt on general_val.bin: loss 3.6431  perplexity 38.21  (200 fixed windows, step 2,500)
stage3_edith.pt on marvel_val.bin:  loss 2.9223  perplexity 18.58  (200 fixed windows, step 2,500)
```

**What moved from the previously recorded figures, and why the first
explanation for it was wrong.** `CLAUDE.md` records this same pair of
checkpoints, from the same 2026-09-04 retrain, at stage 3 general 38.22 /
Marvel 17.48 and stage 2 general 32.72 / Marvel 11.59. The checkpoint files
are unchanged since that retrain. General-val matches to within 0.01 on both
stages; Marvel-val is higher by about 1.1 (stage 3) and 0.65 (stage 2).

The first run of these numbers was CPU-only, and the explanation offered was
that `train/eval_ppl.py` enables autocast only on CUDA, so the recorded
figures came from GPU autocast and the new ones from CPU fp32. **That
explanation was wrong, and re-running on a GPU disproved it.** Same script,
same checkpoints, same fixed windows, on CUDA:

| checkpoint | general | Marvel |
|---|---:|---:|
| stage 3 | 38.22 | 18.59 |
| stage 2 | 32.72 | 12.24 |

CPU and GPU agree to within 0.01 on all four. The device was never the
cause. The 17.48 and 11.59 figures are simply stale — recorded under
conditions this repository no longer records, and not reproducible by the
command above on the shipped weights. **18.59 / 12.24 is what
`train/eval_ppl.py` prints, on either device, on the weights that ship.**

The reason this correction is kept rather than quietly edited: the earlier
explanation was reasonable, specific, and testable, and it was still wrong.
A number published with a plausible story attached is not the same as a
number that has been checked.

`ROADMAP.md`'s "Where we are" table shows a third, older pair (stage 2
33.28 / 12.46) — those are explicitly the pre-retrain reference numbers
`CLAUDE.md` says the 2026-09-04 retrain beat, left stale in that table. Not
a candidate for "the real number"; noted only so nobody reconciles against it.

## 4. The replay decision

*Recorded during training, 2026-09-04. Not re-run tonight — it needed a
training loop, not an eval script, and the decision it produced is already
baked into which checkpoints exist.*

The stage-2 Marvel/English mix ratio was chosen by a 4-arm probe, 800 steps
per arm, identical methodology:

| Marvel/English | general ppl | Marvel ppl | English gained per Marvel point lost |
|---|---:|---:|---:|
| 100/0 | 61.55 | 14.91 | — |
| **70/30** | **31.02** | **16.01** | **27.8** |
| 50/50 | 29.03 | 17.16 | 1.7 |
| 30/70 | 27.75 | 19.29 | 0.6 |

The first 30% of replay recovers about half the English loss for roughly one
point of Marvel perplexity. Past that point the trade gets steadily worse:
50/50 buys 1.7 points of general ppl per Marvel point, 30/70 buys 0.6. That
elbow at 30% is why stage 2 replays at 70/30, not further.

## 5. Retrieval

`py infer/evaluate.py --n 300`, tonight:

```
300 records sampled from 202,171

probe         exact record   owns the name
headline            99.0%          100.0%
real name           83.8%          100.0%
profile             99.0%          100.0%
field               99.0%           99.0%
ALL                 97.5%           99.7%   (999 probes)

flagships      40/40      100.0%   (ambiguous names, judgement calls)

25 failures; first 12 [...]
```

**Round-trip: 97.5% exact record over 999 probes.** **Flagships: 40/40
(100%)** — the hand-adjudicated ambiguous-name set (Spider-Man, Venom,
Captain America, and 37 others) all resolve to the intended character.
Both are close to the brief's expectation (97.3% / 40/40); the ALL figure
moved up 0.2 points, within normal sample-to-sample noise for a random
300-record draw.

**The corpus is bigger than `ROADMAP.md` says.** The population sampled
from tonight is 202,171 records — the corpus was re-curated and the index
rebuilt earlier tonight (`retrieve/index.pkl` now 522 MB, `retrieve/names.pkl`
201,352 names of which 4,578 notable). `ROADMAP.md`'s "Where we are" table
still says 201,815; that number is stale as of tonight's rebuild, by 356
records.

**Two index rewrites, from `ROADMAP.md`'s decisions log** (recorded during
Phase 4, 2026-09-01/02; not re-run tonight, since they measure code paths
rather than model or corpus quality):

| what | before | after | what changed |
|---|---:|---:|---|
| BM25 `search()` (fallback path) | 353 ms | 3.4 ms | Three quarters of the old cost was re-tokenising 187,584 headlines per call; those are now precomputed at build time, postings are flat int32 arrays instead of a dict per term, and scoring is vectorised. |
| `resolve()` | 11.2 ms | 0.11 ms | Rewritten to resolve by shared token index rather than scanning all names and building a `set()` per name; the union of per-token name lists is provably the complete candidate set, verified against the exhaustive scan with 0/905 disagreements. |
| index load | 1.5 s | 0.33 s | A side effect of the flat-array format used by the `search()` rewrite above. |

## 6. Answers — `run_cases.py`

`py infer/run_cases.py`, tonight: **23/27**, matching the previously
recorded score exactly. It ran on CPU, not timed with a stopwatch, but
noticeably slower to the eye than the brief's ~10-minute estimate (which
reads as GPU-sized, given every other estimate in the brief undershoots
tonight's CPU-only conditions the same way).

The harness drives `infer/terminal.py --trace` as one subprocess per case —
the real product, not `engine.plan()` in isolation — and keys each case on
the `Page:` line the trace reports (a record's identity) rather than on a
headline string, because headline strings collide (five different
records are all headlined "Doctor Doom").

Failures, by number and question, from tonight's run:

| # | question | reason |
|---|---|---|
| 6 | `who is cable` | missing `'new mutants'` from the answer |
| 9 | `who is power man` | offered a picker of 10 records; an answer was expected |
| 22 | `who is spider-man 2099` | offered a picker of 5 records; an answer was expected |
| 24 | `who is spider-man earth-1610` | refused instead of offering a choice |

All four resolve to the *correct* underlying record (the harness confirms
the `Page:` line independently of the failure reason) — every failure here
is about how the answer was delivered (a picker instead of an answer, a
refusal instead of a picker, a missing detail), not about which character
was found.

## 7. fp16 vs fp32

From Task 3 (`.superpowers/sdd/2026-09-05-packaging/task-3-report.md`),
comparing `checkpoints/model.safetensors` (fp16, shipped) against an fp32
conversion of the same checkpoint, greedy decoding (`temp=1e-6, top_k=1,
top_p=1.0, repetition_penalty=1.0`) on 20 real questions:

| | bytes | MB |
|---|---:|---:|
| fp32 | 1,015,636,256 | 1015.6 |
| fp16 (shipped) | 507,824,368 | 507.8 |

**20/20 answers byte-identical.** No divergence to quote — every prompt hit
the `IDENTICAL` branch, so there is no fp32-vs-fp16 text pair to show. Two
of the twenty answers are quoted here anyway because they resurface in this
project's "what it gets wrong" writeups, and identical output means fp16
inherits them exactly rather than introducing them:

> `what happened in civil war` → *"Civil War was created by Jonathan Hickman
> and Esad Ribić."*

> `how did bruce banner become the hulk` → *"Bruce Banner became the Hulk
> after his transformation into the Hulk. He is from Earth-TRN1589."*

**Verdict: fp16 ships**, at half the size (508 MB vs 1016 MB), with a
measured, not assumed, guarantee that it answers identically to fp32 on
every question tested.

## 8. The test suite

Two invocations, run separately because one combined run exhausts the
Windows paging file on this machine:

```bash
py -m pytest -q --ignore=infer/test_terminal_session.py
# 1101 passed, 1 warning in 26.30s

py -m pytest -q infer/test_terminal_session.py
# 8 passed in 54.31s
```

**1109 passed, 0 failed, tonight.** The second run's warning is a
`pytest-asyncio` deprecation notice, not a test failure. This is 3 more
passing tests than the 1098 (+ 8) recorded earlier in this packaging phase —
consistent with the test files this and the preceding packaging tasks
added (`test_install.py`, expansions to `release/test_export_repo.py`); no
test was removed or skipped to reach this count.
