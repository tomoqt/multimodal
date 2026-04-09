#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

CONFIG_PATH="${1:-configs/official_nmiracle_like_4xa100_50ep.yaml}"
shift || true

ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/artifacts/official_nmiracle_like_8xa100}"
DATA_DIR="${DATA_DIR:-data/tokenized_official_nmiracle_like/data}"
EVAL_PROTOCOLS="${EVAL_PROTOCOLS:-nmiracle_like}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
BEAM_WIDTH="${BEAM_WIDTH:-10}"
N_BEST="${N_BEST:-10}"
BEAM_BATCH_SIZE="${BEAM_BATCH_SIZE:-8}"
GREEDY_BATCH_SIZE="${GREEDY_BATCH_SIZE:-32}"
MAX_LEN="${MAX_LEN:-128}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | grep -c '^GPU ' || true)}"
if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" == "0" ]]; then
  NPROC_PER_NODE=1
fi

mkdir -p "${ARTIFACT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

OUTPUT_DIR="$(python3 - <<'PY' "${CONFIG_PATH}"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg["checkpoint"]["output_dir"])
PY
)"

mkdir -p "${OUTPUT_DIR}"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
if [[ -z "${RESUME_CHECKPOINT}" ]]; then
  latest_checkpoint="$(ls -1t "${OUTPUT_DIR}"/epoch_*.pt 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    RESUME_CHECKPOINT="${latest_checkpoint}"
  fi
fi

echo "[start] $(date --iso-8601=seconds)"
echo "[launch] repo_root=${REPO_ROOT}"
echo "[launch] config=${CONFIG_PATH}"
echo "[launch] data_dir=${DATA_DIR}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] artifact_dir=${ARTIFACT_DIR}"
echo "[launch] nproc_per_node=${NPROC_PER_NODE}"

train_cmd=(
  torchrun
  --standalone
  --nproc_per_node "${NPROC_PER_NODE}"
  training/train_autoregressive.py
  --config "${CONFIG_PATH}"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "[launch] resuming_from=${RESUME_CHECKPOINT}"
  train_cmd+=(--checkpoint "${RESUME_CHECKPOINT}")
else
  echo "[launch] resuming_from=none"
fi
if [[ "$#" -gt 0 ]]; then
  train_cmd+=("$@")
fi

"${train_cmd[@]}" 2>&1 | tee "${ARTIFACT_DIR}/official_nmiracle_like_50ep_train.log"

latest_checkpoint="$(ls -1t "${OUTPUT_DIR}"/epoch_*.pt 2>/dev/null | head -n 1 || true)"
if [[ -z "${latest_checkpoint}" ]]; then
  echo "[error] no checkpoint found in ${OUTPUT_DIR} after training" >&2
  exit 1
fi

echo "[eval] checkpoint=${latest_checkpoint}"
python3 scripts/eval_public_protocols.py \
  --checkpoint "${latest_checkpoint}" \
  --config "${CONFIG_PATH}" \
  --data_dir "${DATA_DIR}" \
  --protocols "${EVAL_PROTOCOLS}" \
  --split "${EVAL_SPLIT}" \
  --beam_width "${BEAM_WIDTH}" \
  --n_best "${N_BEST}" \
  --max_len "${MAX_LEN}" \
  --decode_strategy beam \
  --beam_batch_size "${BEAM_BATCH_SIZE}" \
  --greedy_batch_size "${GREEDY_BATCH_SIZE}" \
  --output_json "${ARTIFACT_DIR}/official_nmiracle_like_public_eval_beam${BEAM_WIDTH}.json" \
  2>&1 | tee "${ARTIFACT_DIR}/official_nmiracle_like_public_eval.log"

echo "[done] $(date --iso-8601=seconds)"
