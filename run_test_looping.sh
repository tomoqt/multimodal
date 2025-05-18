#!/bin/bash

# Define checkpoints as separate array elements
CHECKPOINTS=("10k.pt" "50k.pt" "100k.pt")
LOOP_COUNTS=(1 5 15 30 45 60)
MAX_SAMPLES=1000  # Number of samples to process
BASE_CHECKPOINT_PATH="/home/consorzio/Technoscience/Research/multimodal/checkpoints"
BASE_OUTPUT_DIR="inference_results_test_looping"

# Ensure the base output directory exists
mkdir -p "$BASE_OUTPUT_DIR"

# Process each checkpoint separately
for CKPT in "${CHECKPOINTS[@]}"; do
  # Extract checkpoint name without extension for the output directory
  CKPT_NAME=$(basename "$CKPT" .pt)
  
  echo "================================================================================"
  echo "Running test_looping.py for checkpoint: $CKPT"
  echo "Loop counts: ${LOOP_COUNTS[*]}"
  echo "================================================================================"
  
  OUTPUT_DIR="$BASE_OUTPUT_DIR/ckpt_${CKPT_NAME}"
  mkdir -p "$OUTPUT_DIR"
  
  python test_looping.py \
    --config configs/real_config.yaml \
    --checkpoint "$BASE_CHECKPOINT_PATH/$CKPT" \
    --max_samples "$MAX_SAMPLES" \
    --loop_counts "${LOOP_COUNTS[@]}" \
    --split test \
    --output_dir "$OUTPUT_DIR"
  
  echo "--------------------------------------------------------------------------------"
  echo "Finished test_looping for $CKPT with loop counts: ${LOOP_COUNTS[*]}"
  echo "Results saved in $OUTPUT_DIR"
  echo "--------------------------------------------------------------------------------"
  echo ""
done

echo "================================================================================"
echo "All loop testing complete."
echo "Results saved in $BASE_OUTPUT_DIR"
echo "================================================================================" 