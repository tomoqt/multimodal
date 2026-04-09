#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/real_config_8xh100_100m.yaml}"
shift || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

NNODES="${NNODES:-8}"
NODE_RANK="${NODE_RANK:?NODE_RANK is required}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

echo "[launch] config=${CONFIG_PATH}"
echo "[launch] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE}"
echo "[launch] master=${MASTER_ADDR}:${MASTER_PORT}"

torchrun \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  training/train_autoregressive.py \
  --config "${CONFIG_PATH}" \
  "$@"
