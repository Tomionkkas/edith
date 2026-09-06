#!/bin/bash
# Measure the Marvel/English trade-off curve before committing 75 minutes.
#
# We only have the two ENDPOINTS: pure Marvel (general ppl 161.6, marvel
# 13.07 at 4 epochs) and stage 1 (general 24.2, marvel 100.7). Every ratio in
# between is guesswork, so this measures four of them instead of arguing.
#
# The 0% arm is the control: it makes the comparison like-for-like at 800
# steps, since the 161.6 figure came from a 6,764-step run.
#
# Each arm runs a COMPLETE cosine schedule so the models are comparable.
# 800 steps is ~0.5 Marvel epochs: this gives reliable ORDERING and rough
# shape, not the numbers a full run would reach.
#
# No stop logic here on purpose - train/idle_watchdog.sh handles that, so it
# cannot shut the box down between arms or during the follow-up run.

set -u
cd /root/marvel-slm
PY=/venv/main/bin/python
log() { echo "[$(date -u '+%H:%M:%S')] $*"; }

STEPS=800

log "probing 4 ratios at ${STEPS} steps each (~9 min per arm)"
for pair in "0.0 0" "0.3 30" "0.5 50" "0.7 70"; do
  set -- $pair
  P=$1; PCT=$2
  NAME="probe_e${PCT}"
  log "===== $((100 - PCT))/${PCT} marvel/english (mix_prob=${P}) ====="

  rm -rf "checkpoints/${NAME}"
  $PY train/trainer.py --stage 2 --run-name "${NAME}" \
      --mix-bin data/packed/general_train.bin --mix-prob "${P}" \
      --init-from checkpoints/stage1/stage1_final_backup.pt \
      --max-steps ${STEPS} --warmup 50 --eval-every 400 --ckpt-every 100000 \
      > "${NAME}.log" 2>&1

  log "  $(grep 'stopped at step' ${NAME}.log | tail -1)"
  # latest.pt = end of a completed schedule, the only fair comparison point
  for b in general_val marvel_val; do
    $PY train/eval_ppl.py --ckpt "checkpoints/${NAME}/latest.pt" \
        --bin "data/packed/${b}.bin" --windows 200 2>&1 | tail -1
  done
  rm -rf "checkpoints/${NAME}"          # 6 GB per arm; reclaim before the next
  df -h /root | tail -1
done

log "===== reference points ====="
log "  stage 1 (all english):        general  24.20   marvel 100.65"
log "  4 epochs pure marvel:         general 161.58   marvel  13.07"
log "probe complete"
