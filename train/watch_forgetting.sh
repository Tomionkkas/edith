#!/bin/bash
# Score general English against the newest checkpoint, while training runs.
#
# A run's own validation set is drawn from its training distribution and
# cannot see forgetting: stage 2's Marvel val loss improved the whole way
# while general perplexity went 24 -> 162. This watches from outside, so it
# also works for a run already in flight.
#
# Reads latest.pt, which the trainer writes atomically (.tmp then rename), so
# there is no torn read. Granularity is --ckpt-every, not --eval-every.
#
#   setsid nohup bash train/watch_forgetting.sh 360 > forgetting.log 2>&1 &

set -u
cd /root/marvel-slm
PY=/venv/main/bin/python
EVERY=${1:-360}
CKPT=checkpoints/stage3/latest.pt

echo "[$(date -u '+%H:%M:%S')] watching general_val.bin every ${EVERY}s"
echo "  reference: stage 2 finished at general perplexity 33.28"

while pgrep -f "trainer.py --stage 3" > /dev/null 2>&1; do
  if [ -f "$CKPT" ]; then
    step=$($PY -c "import torch,sys
try:
    print(torch.load('$CKPT', map_location='cpu', weights_only=False)['step'])
except Exception:
    print('?')" 2>/dev/null)
    line=$($PY train/eval_ppl.py --ckpt "$CKPT" --bin data/packed/general_val.bin \
             --windows 100 --batch 2 2>&1 | tail -1)
    echo "[$(date -u '+%H:%M:%S')] step ${step} | ${line}"
  else
    echo "[$(date -u '+%H:%M:%S')] no checkpoint yet"
  fi
  sleep "$EVERY"
done
echo "[$(date -u '+%H:%M:%S')] trainer gone; monitor exiting"
