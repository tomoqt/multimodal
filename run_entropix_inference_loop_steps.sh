#!/bin/bash

CHECKPOINTS=("100k.pt")
LOOP_STEPS=(1 5)
BASE_CHECKPOINT_PATH="/home/consorzio/Technoscience/Research/multimodal/checkpoints"
BASE_OUTPUT_DIR="inference_results_entropix_loop_steps"
CALIBRATION_DIR="calibration_results"
# Set default threshold levels (can be "conservative", "moderate", or "aggressive")
THRESHOLD_LEVEL_ENTROPY=${THRESHOLD_LEVEL_ENTROPY:-"aggressive"}
THRESHOLD_LEVEL_VARENTROPY=${THRESHOLD_LEVEL_VARENTROPY:-"aggressive"}
# Whether to force recalibration even if calibration files exist (default: no)
FORCE_RECALIBRATE=${FORCE_RECALIBRATE:-"no"}

# Ensure the base directories exist
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$CALIBRATION_DIR"

# Validate threshold levels
if [[ ! "$THRESHOLD_LEVEL_ENTROPY" =~ ^(conservative|moderate|aggressive)$ ]]; then
  echo "Error: Invalid threshold level '$THRESHOLD_LEVEL_ENTROPY'. Must be one of: conservative, moderate, aggressive"
  exit 1
fi

if [[ ! "$THRESHOLD_LEVEL_VARENTROPY" =~ ^(conservative|moderate|aggressive)$ ]]; then
  echo "Error: Invalid threshold level '$THRESHOLD_LEVEL_VARENTROPY'. Must be one of: conservative, moderate, aggressive"
  exit 1
fi

echo "Using $THRESHOLD_LEVEL_ENTROPY entropy and $THRESHOLD_LEVEL_VARENTROPY varentropy thresholds for all experiments"
echo "Force recalibration: $FORCE_RECALIBRATE"

for CKPT in "${CHECKPOINTS[@]}"; do
  # Extract checkpoint name without extension for the output directory
  CKPT_NAME=$(basename "$CKPT" .pt)
  CKPT_CALIBRATION_DIR="$CALIBRATION_DIR/ckpt_${CKPT_NAME}"
  CALIBRATION_FILE="$CKPT_CALIBRATION_DIR/threshold_recommendations.json"
  
  # Check if calibration needs to be run
  SHOULD_CALIBRATE="yes"
  if [[ -f "$CALIBRATION_FILE" ]] && [[ "$FORCE_RECALIBRATE" != "yes" ]]; then
    echo "Calibration file already exists: $CALIBRATION_FILE"
    echo "Skipping calibration (use FORCE_RECALIBRATE=yes to override)"
    SHOULD_CALIBRATE="no"
  fi
  
  if [[ "$SHOULD_CALIBRATE" == "yes" ]]; then
    echo "================================================================================"
    echo "Running calibration for checkpoint: $CKPT"
    echo "Calibration directory: $CKPT_CALIBRATION_DIR"
    echo "================================================================================"
    
    # Run calibration for this checkpoint
    python calibrate_entropix.py \
      --config configs/real_config.yaml \
      --checkpoint "$BASE_CHECKPOINT_PATH/$CKPT" \
      --output_dir "$CKPT_CALIBRATION_DIR" \
      --split test \
      --num_samples 100 \
      --max_seq_steps 90
  fi
  
  # Extract thresholds from the JSON file
  ENTROPY_THRESHOLD=$(python -c "import json; print(json.load(open('$CALIBRATION_FILE'))['entropy']['recommended_thresholds']['$THRESHOLD_LEVEL_ENTROPY'])")
  VARENTROPY_THRESHOLD=$(python -c "import json; print(json.load(open('$CALIBRATION_FILE'))['varentropy']['recommended_thresholds']['$THRESHOLD_LEVEL_VARENTROPY'])")
  
  # Validate that we have valid numeric values
  if [[ ! $ENTROPY_THRESHOLD =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ ! $VARENTROPY_THRESHOLD =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: Could not extract valid threshold values from the JSON file"
    echo "ENTROPY_THRESHOLD: $ENTROPY_THRESHOLD"
    echo "VARENTROPY_THRESHOLD: $VARENTROPY_THRESHOLD"
    exit 1
  fi
  
  echo "Extracted thresholds for $CKPT:"
  echo "  Entropy threshold ($THRESHOLD_LEVEL_ENTROPY): $ENTROPY_THRESHOLD"
  echo "  Varentropy threshold ($THRESHOLD_LEVEL_VARENTROPY): $VARENTROPY_THRESHOLD"
  
  for LOOP_STEP in "${LOOP_STEPS[@]}"; do
    OUTPUT_DIR="$BASE_OUTPUT_DIR/ckpt_${CKPT_NAME}_loopstep_${LOOP_STEP}_${THRESHOLD_LEVEL_ENTROPY}_${THRESHOLD_LEVEL_VARENTROPY}"
    
    echo "================================================================================"
    echo "Running inference for checkpoint: $CKPT"
    echo "Loop increase step: $LOOP_STEP"
    echo "Using thresholds:"
    echo "  Entropy ($THRESHOLD_LEVEL_ENTROPY): $ENTROPY_THRESHOLD"
    echo "  Varentropy ($THRESHOLD_LEVEL_VARENTROPY): $VARENTROPY_THRESHOLD"
    echo "Output directory: $OUTPUT_DIR"
    echo "================================================================================"
    
    python test_inference.py \
      --config configs/real_config.yaml \
      --checkpoint "$BASE_CHECKPOINT_PATH/$CKPT" \
      --full_dataset_test \
      --max_loops 30 \
      --max_examples 10 \
      --entropy_threshold "$ENTROPY_THRESHOLD" \
      --varentropy_threshold "$VARENTROPY_THRESHOLD" \
      --loop_increase_step "$LOOP_STEP" \
      --output_dir "$OUTPUT_DIR"
      
    echo "--------------------------------------------------------------------------------"
    echo "Finished inference for $CKPT with loop_increase_step $LOOP_STEP"
    echo "--------------------------------------------------------------------------------"
    echo ""
  done
done

echo "================================================================================"
echo "All Entropix inference runs with varying loop steps complete."
echo "Results saved in $BASE_OUTPUT_DIR"
echo "================================================================================" 