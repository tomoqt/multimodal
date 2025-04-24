#!/bin/bash

# Configuration
CONFIG_FILE="configs/test_config.yaml"
BASE_RUN_NAME="muon_mix_lr_sweep" # Base name for wandb runs
LRS=(1e-2 5e-3 1e-3 1e-4)

# Ensure the script exits if any command fails
set -e

echo "Starting learning rate sweep..."

for lr in "${LRS[@]}"
do
  # Format LR for run name (replace '.' with 'p', 'e-' with 'em')
  # Example: 5e-3 -> 5em3, 1e-2 -> 1em2
  lr_tag=$(echo "$lr" | sed 's/\./p/' | sed 's/e-0\?/em/' | sed 's/+//') 
  run_name="${BASE_RUN_NAME}_${lr_tag}"

  # Calculate min_lr = lr / 10
  min_lr=$(echo "scale=10; $lr / 10" | bc) 
  # Format min_lr to scientific notation if needed (optional, depends on how python script handles it)
  # min_lr=$(printf "%.1E" "$min_lr") # Example: 1.0E-05

  echo "--------------------------------------------------"
  echo "Running with LR = $lr, Min LR = $min_lr (Run Name: $run_name)"
  echo "--------------------------------------------------"

  # Run the training script with overridden parameters
  # Overriding both AdamW lr (training.learning_rate) and Muon lr (optimizer.muon.lr)
  # Also overriding the wandb base run name for clarity
  # And overriding the min_learning_rate
  python training/train_autoregressive.py \
    --config "$CONFIG_FILE" \
    training.learning_rate="$lr" \
    training.min_learning_rate="$min_lr" \
    optimizer.muon.lr="$lr" \
    wandb.base_run_name="$run_name"

  echo "Finished run for LR = $lr"
done

echo "--------------------------------------------------"
echo "Learning rate sweep completed successfully."
echo "--------------------------------------------------"

exit 0 