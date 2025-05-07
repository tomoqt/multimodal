#!/usr/bin/env bash
LOGFILE="train.log"
while true; do
  echo "[$(date '+%F %T')] Starting training…" | tee -a "$LOGFILE"
  torchrun --nproc_per_node=1 \
       training/train_autoregressive.py \
       --config configs/real_config.yaml \
    2>&1 | tee -a "$LOGFILE"
  EXIT=$?
  echo "[$(date '+%F %T')] Exited with code $EXIT" | tee -a "$LOGFILE"
  if [ $EXIT -eq 0 ]; then
    echo "[$(date '+%F %T')] Finished cleanly." | tee -a "$LOGFILE"
    break
  fi
  echo "[$(date '+%F %T')] Crash detected; restarting in 10s…" | tee -a "$LOGFILE"
  sleep 10
done
