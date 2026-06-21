#!/bin/bash
# Wait until the correlative-microscopy sweep fully releases the GPU, then launch the
# symmc-flow revision battery (batch 16). Box is SHARED: we must NOT run while the
# correlative finetune is active, and must not fire during its brief inter-config gaps.
# Condition to launch: NO finetune_ma_roma process AND >9 GiB free, sustained for
# CLEAR_NEEDED consecutive 60 s checks. Runs in tmux so it survives disconnects.
set -u
cd /workspace/symmc-flow
LOG=gpu_results/revision/_watch.log
mkdir -p gpu_results/revision
CLEAR_NEEDED=5            # 5 consecutive clear minutes => sweep truly done, not a gap
clear=0
echo "WATCH START $(date) — waiting for correlative sweep to release GPU" | tee $LOG
while true; do
  if pgrep -f finetune_ma_roma >/dev/null 2>&1; then
    busy=1
  else
    busy=0
  fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  free=${free:-0}
  if [ "$busy" -eq 0 ] && [ "$free" -gt 9000 ]; then
    clear=$((clear+1))
  else
    clear=0
  fi
  echo "$(date) busy=$busy free=${free}MiB clear=$clear/$CLEAR_NEEDED" >> $LOG
  if [ "$clear" -ge "$CLEAR_NEEDED" ]; then
    echo "GPU FREE & CORRELATIVE DONE — launching battery $(date)" | tee -a $LOG
    break
  fi
  sleep 60
done
exec bash scripts/run_revision3.sh
