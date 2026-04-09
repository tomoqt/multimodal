#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/real_config_8gpu_100m.yaml}"
shift || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29517}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "[launch] sweep config=${CONFIG_PATH}"
echo "[launch] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE}"
echo "[launch] master=${MASTER_ADDR}:${MASTER_PORT}"

python3 scripts/sweep_batch_sizes.py \
  --config "${CONFIG_PATH}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  "$@"
