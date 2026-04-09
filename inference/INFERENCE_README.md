# Inference (Lean Core)

Supported decoding strategies:

- `greedy`
- `beam`
- `sampling`
- `nucleus`

The looped/Entropix strategies were intentionally removed in this branch.

## Quick Usage

```bash
python3 test_inference.py \
  --checkpoint checkpoints/epoch_1.pt \
  --config configs/real_config.yaml \
  --strategies greedy,beam,sampling,nucleus \
  --max_examples 50
```

Results are written to `inference_results/core_eval.csv` by default.
