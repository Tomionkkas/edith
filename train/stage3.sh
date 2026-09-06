#!/bin/bash
# Stage 3: instruction tuning, with replay.
#
# 2,500 steps. An 800-step run scored well (general 37.86, Marvel 15.69) but
# could not GROUND: asked who created Wolverine it answered Jeff Parker and
# Gabriel Hardman while Len Wein sat in its context. 800 steps is only ~3.5M
# scored tokens, a fifth of one epoch. The SFT val loss plateaus at 0.35
# regardless, because it is dominated by easy template tokens while the hard
# part - copying the right fact out of the context - keeps improving. Reading
# that plateau as "learned" was wrong.
#
# Replay rises to 11.9% (a 50/50 scored split, up from 60/40) to pay for the
# extra steps: the 2,963-step run reached general 52.84 at only 70/30.
#
# --mix-prob 0.082 is NOT a typo for 0.30. Stage 3 masks loss to the assistant
# turn, so an SFT window contributes ~13.5% of its tokens while a replay
# window of ordinary text contributes 100%. 0.082 yields a 60/40 split of the
# tokens actually SCORED - see replay_prob_for() in trainer.py. Slightly more
# replay than the first run's 70/30, because the SFT signal saturates long
# before the schedule ends and English needs the protection more.
#
# --lr 1e-4, not the 6e-4 stages 1 and 2 used: this changes the objective, and
# the pretraining rate overwrites stage 2 faster than it teaches the format.
#
# --watch-bin scores general English at EVERY eval. A run's own validation set
# comes from its training distribution and cannot see forgetting: stage 2's
# Marvel val loss improved the whole way while general perplexity went 24->162.
#
# No stop logic here: train/idle_watchdog.sh owns shutdown.

set -u
cd /root/marvel-slm
PY=/venv/main/bin/python

log() { echo "[$(date -u '+%H:%M:%S')] $*"; }

log "stage 3: 2,500 steps, replay 11.9% of sequences (50/50 by scored token)"
$PY train/trainer.py --stage 3 --run-name stage3 \
    --mix-bin data/packed/general_slice.bin --mix-prob 0.119 \
    --watch-bin data/packed/general_val.bin \
    --init-from checkpoints/stage2/edith_stage2_replay.pt \
    --lr 1e-4 --max-steps 2500 --warmup 100 --eval-every 250 --ckpt-every 500 \
    > stage3.log 2>&1

log "finished: $(grep 'stopped at step' stage3.log | tail -1)"
log "best saves: $(grep -c 'new best' stage3.log)"

log "stripping to weights-only for a 1 GB download"
$PY train/strip_ckpt.py checkpoints/stage3/best.pt checkpoints/stage3/latest.pt

# No "if exists" guard: a missing val set must be loud. Skipping silently is
# how the first run finished reporting no forgetting numbers at all.
log "scoring both validation sets"
for c in checkpoints/stage3/best.pt checkpoints/stage3/latest.pt; do
  for b in general_val marvel_val; do
    $PY train/eval_ppl.py --ckpt "$c" --bin "data/packed/${b}.bin" --windows 200 2>&1 | tail -1
  done
done

log "reference - stage 2 was general 33.28 / marvel 12.46"
log "reference - stage 3 at 2,963 steps was general 52.84 / marvel 25.03"
log "done"
