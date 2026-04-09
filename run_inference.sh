#!/bin/bash
set -euo pipefail

python3 test_inference.py \
  --config configs/test_config.yaml \
  --checkpoint checkpoints/epoch_1.pt \
  --strategies greedy,beam,sampling,nucleus \
  --max_examples 20
