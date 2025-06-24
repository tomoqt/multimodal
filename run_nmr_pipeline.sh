#!/usr/bin/env bash
# ===============================================================
# run_nmr_pipeline.sh
# ---------------------------------------------------------------
# Convenience launcher that can (optionally) run the two training
# stages we have implemented:
#   1) Supervised fine-tuning (training/train_sft_nmr.py)
#   2) GRPO RL post-training (training/train_grpo_nmr.py)
#
# USAGE EXAMPLES
#   bash run_nmr_pipeline.sh                       # run both stages with defaults
#   bash run_nmr_pipeline.sh --no-sft              # skip SFT, only GRPO
#   bash run_nmr_pipeline.sh --no-grpo             # only run SFT
#   bash run_nmr_pipeline.sh --model mistralai/Mistral-7B-v0.1 \
#        --sft-data path/to/sft_data \
#        --grpo-data path/to/grpo_data \
#        --hf-repo myuser/nmr-model \
#        --extra-sft "--num_train_epochs 3" \
#        --extra-grpo "--num_train_epochs 1 --beta 0.05"
#
# OPTIONS
#   --no-sft                 Skip SFT stage
#   --no-grpo                Skip GRPO stage
#   --model  <name_or_path>  Base model to fine-tune (passed to both stages)
#   --sft-data <dir>         Directory containing src/tgt pairs for SFT
#   --grpo-data <dir>        Directory containing src/tgt pairs for GRPO
#   --sft-out  <dir>         Checkpoint location for SFT
#   --grpo-out <dir>         Checkpoint location for GRPO
#   --hf-repo  <repo_id>     Push final stage(s) to HuggingFace Hub repo
#   --hf-token <token>       HF token (if not already logged-in)
#   --extra-sft "<args>"     Extra args passed verbatim to train_sft_nmr.py
#   --extra-grpo "<args>"    Extra args passed verbatim to train_grpo_nmr.py
#   --accel-args "<args>"    Extra flags passed to `accelerate launch` (e.g. "--multi_gpu")
# ===============================================================
set -euo pipefail

# -------- default values --------
RUN_SFT=1
RUN_GRPO=1
MODEL_NAME="gpt2"
SFT_DATA="data/reshaped_tokenized_data/data"
GRPO_DATA="data/reshaped_tokenized_data/data"
SFT_OUT="checkpoints_sft_nmr"
GRPO_OUT="checkpoints_grpo_nmr"
HF_REPO=""
HF_TOKEN=""
EXTRA_SFT=""
EXTRA_GRPO=""
ACCEL_ARGS=""

# -------- arg parsing --------
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-sft)          RUN_SFT=0; shift ;;
    --no-grpo)         RUN_GRPO=0; shift ;;
    --model)           MODEL_NAME="$2"; shift 2 ;;
    --sft-data)        SFT_DATA="$2"; shift 2 ;;
    --grpo-data)       GRPO_DATA="$2"; shift 2 ;;
    --sft-out)         SFT_OUT="$2"; shift 2 ;;
    --grpo-out)        GRPO_OUT="$2"; shift 2 ;;
    --hf-repo)         HF_REPO="$2"; shift 2 ;;
    --hf-token)        HF_TOKEN="$2"; shift 2 ;;
    --extra-sft)       EXTRA_SFT="$2"; shift 2 ;;
    --extra-grpo)      EXTRA_GRPO="$2"; shift 2 ;;
    --accel-args)      ACCEL_ARGS="$2"; shift 2 ;;
    -h|--help)
      grep -E "^#( |$)" "$0" | sed -E 's/^# ?//'; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# -------- helper for HF flags --------
function hf_flags() {
  if [[ -n "$HF_REPO" ]]; then
    echo "--push_to_hub --hf_repo_id $HF_REPO"
    if [[ -n "$HF_TOKEN" ]]; then
      echo "--hf_token $HF_TOKEN"
    fi
  fi
}

# -------- run SFT --------
if [[ $RUN_SFT -eq 1 ]]; then
  echo "==================== Stage 1: SFT ===================="
  accelerate launch $ACCEL_ARGS training/train_sft_nmr.py \
    --model_name "$MODEL_NAME" \
    --data_dir "$SFT_DATA" \
    --output_dir "$SFT_OUT" \
    $(hf_flags) \
    $EXTRA_SFT
fi

# -------- run GRPO --------
if [[ $RUN_GRPO -eq 1 ]]; then
  echo "==================== Stage 2: GRPO ===================="
  accelerate launch $ACCEL_ARGS training/train_grpo_nmr.py \
    --model_name "$MODEL_NAME" \
    --data_dir "$GRPO_DATA" \
    --output_dir "$GRPO_OUT" \
    $(hf_flags) \
    $EXTRA_GRPO
fi

echo "Pipeline finished." 