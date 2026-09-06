#!/bin/bash
# Stop the instance after N consecutive idle minutes.
#
# A safety net that does not depend on any run script finishing cleanly: if
# training dies, if a chain script is killed, or if the operator's session
# drops, the box still shuts itself down instead of billing all night.
#
# "Idle" is NOT just an idle GPU. An earlier version watched only
# utilisation and stopped the box mid-workflow while a 385 MB upload was in
# flight and the dataset was being rebuilt locally: real work, no GPU load.
# Anything below counts as busy, and the timeout is generous, because the
# cost of a false stop (a lost box, and a GPU another renter can take) is far
# higher than the cost of a few idle minutes at $0.40/hr.

set -u
export PATH=/opt/instance-tools/bin:$PATH
IDLE_MIN=${1:-45}
WORKDIR=/root/marvel-slm
BUSY_PROCS='trainer.py|eval_ppl|strip_ckpt|sft_data|pack_sft|slice_bin|scp|rsync|sftp-server'

log() { echo "[$(date -u '+%H:%M:%S')] $*"; }

busy_reason() {
  local util
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${util:-0}" -ge 5 ] && { echo "gpu ${util}%"; return 0; }
  pgrep -f "$BUSY_PROCS" > /dev/null 2>&1 && { echo "a job is running"; return 0; }
  # An incoming file transfer holds an established connection on port 22.
  if command -v ss > /dev/null 2>&1 && \
     ss -tn state established 2>/dev/null | grep -q ':22 '; then
    echo "an ssh/scp session is open"; return 0
  fi
  # Touch $WORKDIR/.keep_alive to hold the box during anything else.
  if [ -f "$WORKDIR/.keep_alive" ] && \
     [ "$(( $(date +%s) - $(stat -c %Y "$WORKDIR/.keep_alive") ))" -lt 3600 ]; then
    echo ".keep_alive is fresh"; return 0
  fi
  return 1
}

log "watchdog armed: stop after ${IDLE_MIN} consecutive idle minutes"
log "busy = GPU >=5%, a running job, an open ssh/scp session, or a fresh .keep_alive"
idle=0
while true; do
  if reason=$(busy_reason); then
    [ "$idle" -gt 0 ] && log "busy again (${reason}), idle counter reset from ${idle}"
    idle=0
  else
    idle=$((idle + 1))
  fi
  if [ "$idle" -ge "$IDLE_MIN" ]; then
    log "idle ${IDLE_MIN} min -> stopping instance $CONTAINER_ID"
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" 2>&1 | tail -2
    exit 0
  fi
  sleep 60
done
