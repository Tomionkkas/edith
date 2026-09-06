#!/bin/bash
# Stage 2 with REPLAY: 70% Marvel, 30% general English.
#
# Why: pure Marvel wrecked general English. Perplexity on general_val went
# 24.2 (stage 1) -> 79.5 (1 epoch) -> 161.6 (4 epochs), while Marvel
# perplexity barely moved after the first epoch (13.62 -> 13.07). Nothing in
# the loss mentioned English, so nothing held it in place. Mixing the stage-1
# corpus back in keeps that term alive.
#
# Same 6,764 steps, same LR schedule, same init as the pure-Marvel run, so
# the data mix is the ONLY variable that changed and the comparison is clean.
#
# Writes to checkpoints/stage2_mix/, so the pure-Marvel results stay intact
# and the trainer cannot auto-resume from them.

set -u
cd /root/marvel-slm
PY=/venv/main/bin/python
export PATH=/opt/instance-tools/bin:$PATH

log() { echo "[$(date -u '+%H:%M:%S')] $*"; }

# No stop logic here. train/idle_watchdog.sh owns shutdown: it stops the box
# after 20 consecutive idle-GPU minutes, which covers a crash, a kill, or a
# dropped session, AND leaves a window to download the model before shutdown.
# A trap on this script would have stopped the box the instant training ended.
# Two scripts have already fallen through to a stop line at the wrong moment;
# one GPU-watching mechanism is easier to reason about than a trap per run.

log "stage 2 replay: 70/30 marvel/general, 6,764 steps"

$PY train/trainer.py --stage 2 --run-name stage2_mix \
    --mix-bin data/packed/general_train.bin --mix-prob 0.3 \
    --init-from checkpoints/stage1/stage1_final_backup.pt \
    --max-steps 6764 --warmup 200 --eval-every 200 --ckpt-every 400 \
    > stage2_mix.log 2>&1

log "finished: $(grep 'stopped at step' stage2_mix.log | tail -1)"
log "best saves: $(grep -c 'new best' stage2_mix.log)"
log "best val:   $(grep 'new best' stage2_mix.log | tail -1)"

log "stripping to weights-only for a 1 GB download"
$PY train/strip_ckpt.py checkpoints/stage2_mix/best.pt checkpoints/stage2_mix/latest.pt

# The whole point of the run: does replay hold general English up? Scored
# here so the numbers are in the log even though the box stops afterwards.
log "scoring both validation sets"
for c in checkpoints/stage2_mix/best.pt checkpoints/stage2_mix/latest.pt; do
  for b in general_val marvel_val; do
    $PY train/eval_ppl.py --ckpt "$c" --bin "data/packed/$b.bin" --windows 200 2>&1 | tail -1
  done
done

log "reference - pure Marvel was general 161.58 / marvel 13.07 at 4 epochs"
ls -la checkpoints/stage2_mix/ | awk '{print "  " $9 "  " $5}'
log "done"
