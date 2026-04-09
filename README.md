# MultiModal (Lean Core)

This branch is a lean overhaul focused on the core pretraining path:

- raw dataset -> tokenized train/val/test splits
- optional IR encoder (`convnext` or `regular` CNN)
- decoder-only transformer for SMILES generation
- minimal train/inference scripts without looped decoding or Entropix

## What Was Removed

- looped decoder architecture
- Entropix and loop-specific inference/eval scripts
- heavy training scaffolding (distributed/optimizer variants/wandb-specific code paths)

## Repository Layout

- `data/create_tokenized_dataset_faster.py`:
  Raw parquet -> tokenized text + IR arrays (`.npy`)
- `data/build_vocab.py`:
  Builds source vocabulary (`vocab.txt` + `vocab.json`)
- `models/`:
  `MultiModalToSMILESModel`, `SMILESDecoder`, optional ConvNeXt IR encoder
- `training/core_dataset.py`:
  Dataset + collate for tokenized splits
- `training/core_train.py`:
  Minimal train-step and autoregressive loss utilities
- `training/train_autoregressive.py`:
  Lean pretraining entrypoint
- `test_inference.py`:
  Lean inference/evaluation entrypoint
- `tests/`:
  Core test suite for raw-data preprocessing, dataset loading, model forward, and train-step smoke test

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Raw Data -> Tokenized Data

```bash
python3 data/create_tokenized_dataset_faster.py \
  --analytical_data data_extraction/multimodal_spectroscopic_dataset \
  --out_path data/tokenized_baseline \
  --h_nmr --c_nmr --ir --formula
```

Then build vocabulary:

```bash
python3 data/build_vocab.py
```

## Run Core Tests

```bash
python3 -m unittest discover -s tests -v
```

## Train

```bash
python3 training/train_autoregressive.py --config configs/real_config.yaml
```

Single-node 8-GPU DDP launch:

```bash
./run_train_8gpu.sh configs/real_config_8gpu.yaml
```

Equivalent raw torchrun:

```bash
torchrun --nproc_per_node=8 training/train_autoregressive.py --config configs/real_config_8gpu.yaml
```

100M-model preset (8-GPU):

```bash
./run_train_8gpu.sh configs/real_config_8gpu_100m.yaml
```

## Throughput Sweep (8 GPU)

Run an automated per-GPU batch-size sweep and get ETA for 50 epochs:

```bash
./run_sweep_8gpu.sh configs/real_config_8gpu_100m.yaml \
  --batch-sizes 24,32,40,48,56,64,72,80,96 \
  --warmup-steps 20 \
  --steps 80
```

The sweep writes JSON results to `sweep_results_8gpu.json` and prints:

- best stable batch size
- measured global `tok_s`
- estimated wall-clock for `50` epochs (default uses `tokens_per_epoch=30342999` for full-dataset train split)

Notes:

- Device auto-select is `cuda` -> `mps` -> `cpu` by availability.
- The core dataset path pretokenizes into fixed-length tensors by default (`data.pretokenize: true`) to keep batch shapes static and improve MPS throughput.
- Persistent token caches are enabled by default (`data.use_disk_cache: true`) and saved under `data.cache_dir`.
- Optional IR feature cache mode (`data.ir_cache_mode: pt`) stores split-level IR tensors in the same cache directory.
- Warm caches explicitly with:
  `python3 training/build_token_cache.py --config configs/real_config.yaml`
- DDP is enabled automatically when launched with `torchrun` (`WORLD_SIZE>1`).
- Training logs include `tok_s` (non-pad target tokens per second).

## Evaluate

```bash
python3 test_inference.py \
  --checkpoint checkpoints/epoch_1.pt \
  --config configs/real_config.yaml \
  --strategies greedy,beam,sampling,nucleus
```
