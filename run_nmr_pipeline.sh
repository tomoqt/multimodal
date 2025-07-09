#!/usr/bin/env bash
# ===============================================================
# run_nmr_pipeline.sh
# ---------------------------------------------------------------
# Convenience launcher that can (optionally) run the two training
# stages we have implemented:
#   0) Pre-tokenization for SFT (scripts/pretokenize_sft_data.py)
#   1) Supervised fine-tuning (training/train_sft_nmr.py)
#   2) GRPO RL post-training (training/train_grpo_nmr.py)
#
# USAGE EXAMPLES
#   bash run_nmr_pipeline.sh                       # run all stages with defaults
#   bash run_nmr_pipeline.sh --no-sft              # skip SFT, only GRPO (lora)
#   bash run_nmr_pipeline.sh --no-grpo             # only run SFT (full-finetune)
#   bash run_nmr_pipeline.sh --sft-peft --grpo-full-finetune # sft-lora, grpo-full
#   bash run_nmr_pipeline.sh --model mistralai/Mistral-7B-v0.1 \
#        --sft-data path/to/sft_data \
#        --grpo-data path/to/grpo_data \
#        --hf-repo myuser/nmr-model \
#        --extra-sft "--num_train_epochs 3" \
#        --extra-grpo "--num_train_epochs 1 --beta 0.05" \
#        --lora-r 32
#
# OPTIONS
#   --no-sft                 Skip SFT stage
#   --no-grpo                Skip GRPO stage
#   --skip-sft-tokenization  Skip the SFT pre-tokenization step (assumes it's done)
#   --model  <name_or_path>  Base model to fine-tune (passed to both stages)
#   --sft-data <dir>         Directory containing src/tgt pairs for SFT
#   --grpo-data <dir>        Directory containing src/tgt pairs for GRPO
#   --sft-out  <dir>         Checkpoint location for SFT
#   --grpo-out <dir>         Checkpoint location for GRPO
#   --hf-repo  <repo_id>     Push final stage(s) to HuggingFace Hub repo
#   --hf-token <token>       HF token (if not already logged-in)
#   --sft-peft               Use PEFT/LoRA for SFT stage (default: full finetune)
#   --grpo-full-finetune     Use full fine-tuning for GRPO (default: PEFT/LoRA)
#   --lora-r <int>           LoRA 'r' parameter
#   --lora-alpha <int>       LoRA 'alpha' parameter
#   --lora-dropout <float>   LoRA 'dropout'
#   --extra-sft "<args>"     Extra args passed verbatim to train_sft_nmr.py
#   --extra-grpo "<args>"    Extra args passed verbatim to train_grpo_nmr.py
#   --accel-args "<args>"    Extra flags passed to `accelerate launch` (e.g. "--multi_gpu")
#   --verbose                Pass verbose flag to training stages
# ===============================================================

#./run_nmr_pipeline.sh --sft-peft --accel-args "--config_file fsdp_config.yaml"


set -euo pipefail

# -------- default values --------
RUN_SFT=1
RUN_GRPO=1
SKIP_SFT_TOKENIZATION=0
MODEL_NAME="futurehouse/ether0"
SFT_DATA="data/reshaped_tokenized_data/data"
SFT_TOKENIZED_DATA="data/tokenized_sft_data/$(basename "$MODEL_NAME")" # Model-specific tokenized data
GRPO_DATA="data/reshaped_tokenized_data/data"
SFT_OUT="checkpoints_sft_nmr"
GRPO_OUT="checkpoints_grpo_nmr"
HF_REPO=""
HF_TOKEN=""
SFT_PEFT=0
GRPO_FULL_FINETUNE=0
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.00
EXTRA_SFT=""
EXTRA_GRPO=""
ACCEL_ARGS=""
VERBOSE=0

# -------- arg parsing --------
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-sft)               RUN_SFT=0; shift ;;
    --no-grpo)              RUN_GRPO=0; shift ;;
    --skip-sft-tokenization) SKIP_SFT_TOKENIZATION=1; shift ;;
    --model)                MODEL_NAME="$2"; SFT_TOKENIZED_DATA="data/tokenized_sft_data/$(basename "$2")"; shift 2 ;;
    --sft-data)             SFT_DATA="$2"; shift 2 ;;
    --grpo-data)            GRPO_DATA="$2"; shift 2 ;;
    --sft-out)              SFT_OUT="$2"; shift 2 ;;
    --grpo-out)             GRPO_OUT="$2"; shift 2 ;;
    --hf-repo)              HF_REPO="$2"; shift 2 ;;
    --hf-token)             HF_TOKEN="$2"; shift 2 ;;
    --sft-peft)             SFT_PEFT=1; shift ;;
    --grpo-full-finetune)   GRPO_FULL_FINETUNE=1; shift ;;
    --lora-r)               LORA_R="$2"; shift 2 ;;
    --lora-alpha)           LORA_ALPHA="$2"; shift 2 ;;
    --lora-dropout)         LORA_DROPOUT="$2"; shift 2 ;;
    --extra-sft)            EXTRA_SFT="$2"; shift 2 ;;
    --extra-grpo)           EXTRA_GRPO="$2"; shift 2 ;;
    --accel-args)           ACCEL_ARGS="$2"; shift 2 ;;
    --verbose)              VERBOSE=1; shift ;;
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

# -------- propagate verbose flag to extra args --------
if [[ $VERBOSE -eq 1 ]]; then
  EXTRA_SFT="$EXTRA_SFT --verbose"
  EXTRA_GRPO="$EXTRA_GRPO --verbose"
fi

# -------- construct PEFT args string --------
PEFT_ARGS="--lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT"
SFT_PEFT_FLAG=""
if [[ $SFT_PEFT -eq 1 ]]; then
  SFT_PEFT_FLAG="--use_peft"
fi
GRPO_PEFT_FLAG=""
if [[ $GRPO_FULL_FINETUNE -eq 1 ]]; then
  GRPO_PEFT_FLAG="--full_finetune"
fi

# -------- determine model for GRPO stage --------
GRPO_MODEL_NAME="$MODEL_NAME"
if [[ $RUN_SFT -eq 1 ]]; then
  echo "SFT stage is enabled. GRPO will use its output from '$SFT_OUT'."
  GRPO_MODEL_NAME="$SFT_OUT"
fi

# -------- run SFT Tokenization --------
if [[ $RUN_SFT -eq 1 && $SKIP_SFT_TOKENIZATION -eq 0 ]]; then
  echo "==================== Stage 0: SFT Pre-tokenization ===================="
  # This is a CPU-bound task, run with python directly, not accelerate
  python scripts/pretokenize_sft_data.py \
    --model_name "$MODEL_NAME" \
    --data_dir "$SFT_DATA" \
    --output_dir "$SFT_TOKENIZED_DATA"
    # max_seq_length is defaulted in the script
fi

# -------- run SFT --------
if [[ $RUN_SFT -eq 1 ]]; then
  echo "==================== Stage 1: SFT ===================="
  accelerate launch $ACCEL_ARGS training/train_sft_nmr.py \
    --model_name "$MODEL_NAME" \
    --tokenized_data_dir "$SFT_TOKENIZED_DATA" \
    --output_dir "$SFT_OUT" \
    $SFT_PEFT_FLAG \
    $PEFT_ARGS \
    $(hf_flags) \
    $EXTRA_SFT
fi

# -------- run GRPO --------
if [[ $RUN_GRPO -eq 1 ]]; then
  echo "==================== Stage 2: GRPO ===================="
  accelerate launch $ACCEL_ARGS training/train_grpo_nmr.py \
    --model_name "$GRPO_MODEL_NAME" \
    --data_dir "$GRPO_DATA" \
    --output_dir "$GRPO_OUT" \
    $GRPO_PEFT_FLAG \
    $PEFT_ARGS \
    $(hf_flags) \
    $EXTRA_GRPO
fi

echo "Pipeline finished." 