#!/usr/bin/env python
"""
This script evaluates the automatic loop exit capability of the MultiModalToSMILESModel.
It compares the performance of greedy decoding with automatic exit enabled against
greedy decoding with various fixed numbers of loops (e.g., 1, 10, 100) over a subset
of a specified dataset split.
"""

import argparse
import torch
import os
import json
import yaml
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm

from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from inference.inference import ModelInference, DecodingStrategy

# Import utility functions from test_inference.py.
from test_inference import load_config, get_ir_tokenizer, detect_ir_as_prompt, SimpleSpectralSmilesDataset, evaluate_similarity, combine_metrics

def parse_fixed_loops(loop_string):
    """Parses a comma-separated string of integers into a list."""
    try:
        return sorted([int(x.strip()) for x in loop_string.split(',')])
    except ValueError:
        raise argparse.ArgumentTypeError("Fixed loops must be a comma-separated list of integers (e.g., '1,10,100').")

def main():
    parser = argparse.ArgumentParser(description="Automatic Loop Exit vs Fixed Loops Test")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/test_config.yaml', help='Path to configuration YAML file')
    parser.add_argument('--max_samples', type=int, default=25, help='Maximum number of dataset samples to process')
    parser.add_argument('--fixed_loops', type=parse_fixed_loops, default='1,10,100', help='Comma-separated list of fixed loop counts to test (e.g., "1,10,100")')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Directory to save results')
    # loops_representation is not the primary focus here, but keep it for potential debugging
    parser.add_argument('--loops_representation', type=bool, default=False, help='Flag to track and return representations across loops (can be memory intensive)')
    args = parser.parse_args()

    # Load configuration
    # Ensure config loading handles potential missing keys gracefully for model init
    config_base = load_config(args.config)
    model_config = config_base.get('model', {})
    data_config = config_base.get('data', {})
    training_config = config_base.get('training', {})

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize SMILES tokenizer
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.join(current_dir, 'training/vocab.txt')
    tokenizer = SmilesTokenizer(vocab_file=vocab_path)

    # Load NMR tokenizer
    nmr_vocab_path = Path(data_config.get('tokenized_dir', 'tokenized_baseline/data')).parent / 'vocab.json'
    if not nmr_vocab_path.exists():
        raise FileNotFoundError(f"NMR vocabulary not found at {nmr_vocab_path}")
    with open(nmr_vocab_path) as f:
        nmr_tokenizer = json.load(f)

    # Load checkpoint and detect IR settings
    checkpoint = torch.load(args.checkpoint, map_location=device)
    auto_ir_as_prompt, extra_params = detect_ir_as_prompt(checkpoint, config_base)
    ir_as_prompt = model_config.get('ir_as_prompt', auto_ir_as_prompt) # Prioritize config setting if available

    # Get IR tokenizer if needed
    ir_tokenizer = None
    ir_vocab_size = None
    if ir_as_prompt:
        # Try loading from config path first, then fallback
        ir_tokenizer_path_str = data_config.get('ir_tokenizer_path')
        if ir_tokenizer_path_str:
             ir_tokenizer_path = Path(ir_tokenizer_path_str)
             if ir_tokenizer_path.exists():
                 with open(ir_tokenizer_path, 'r') as f:
                     ir_tokenizer = json.load(f)
                     ir_vocab_size = len(ir_tokenizer)
                     print(f"IR tokenizer loaded from config path: {ir_tokenizer_path}")
             else:
                 print(f"Warning: IR tokenizer path in config not found: {ir_tokenizer_path}")
        
        # Fallback if not loaded from config
        if ir_tokenizer is None:
             ir_tokenizer = get_ir_tokenizer(config_base) # Uses default logic from test_inference
             if ir_tokenizer:
                 ir_vocab_size = len(ir_tokenizer)
                 print(f"IR tokenizer loaded using fallback logic.")
             else:
                  # Raise error only if IR prompt is explicitly required by config but tokenizer fails
                  if config_base.get('data', {}).get('ir_as_prompt', False):
                     raise ValueError("IR as prompt is enabled, but IR tokenizer could not be loaded.")
                  else: # If auto-detected but failed, just disable IR prompt
                     print("Warning: Could not load IR tokenizer using fallback. Disabling IR as prompt.")
                     ir_as_prompt = False
                     ir_vocab_size = None
        
        if ir_tokenizer:
            print(f"IR tokenizer has {len(ir_tokenizer)} tokens")


    # --- Model Initialization ---
    # Get parameters from config, providing defaults if necessary
    max_loops_from_config = model_config.get('max_loops', 1) # Model's internal max loops
    automatic_loop_exit = model_config.get('automatic_loop_exit', False)
    automatic_loop_exit_threshold = model_config.get('automatic_loop_exit_threshold', 0.01)

    print("--- Model Configuration for Initialization ---")
    print(f"  Automatic Loop Exit Enabled: {automatic_loop_exit}")
    if automatic_loop_exit:
        print(f"  Automatic Loop Exit Threshold: {automatic_loop_exit_threshold}")
        print(f"  Max Loops (for auto-exit ceiling): {max_loops_from_config}")
    print("--------------------------------------------")

    # Create the model with appropriate parameters from config
    smiles_vocab_size = len(tokenizer)
    token_ids = list(nmr_tokenizer.values())
    nmr_vocab_size = max(token_ids) + 1 if token_ids else 0 # Handle empty tokenizer case
    model_kwargs = {
        'smiles_vocab_size': smiles_vocab_size,
        'nmr_vocab_size': nmr_vocab_size,
        'max_seq_length': model_config.get('max_seq_length', 512),
        'max_nmr_length': model_config.get('max_nmr_length', 128),
        'max_memory_length': model_config.get('max_memory_length', 128),
        'embed_dim': model_config.get('embed_dim', 768),
        'num_heads': model_config.get('num_heads', 8),
        'num_layers': model_config.get('num_layers', 6),
        'dropout': model_config.get('dropout', 0.1),
        'verbose': False,
        'use_stablemax': model_config.get('use_stablemax', False),
        'ir_as_prompt': ir_as_prompt,
        'ir_encoder_type': model_config.get('ir_encoder_type', 'regular'),
        'ir_vocab_size': ir_vocab_size, # Pass loaded size
        # Core loop parameters from config
        'max_loops': max_loops_from_config,
        'automatic_loop_exit': automatic_loop_exit,
        'automatic_loop_exit_threshold': automatic_loop_exit_threshold,
        # Pass representation tracking flag
        'loops_representation': args.loops_representation
    }
    # Clean up None values potentially passed for ir_vocab_size
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}


    model = MultiModalToSMILESModel(**model_kwargs).to(device)
    try:
        # Filter checkpoint state dict for keys present in the current model
        model_state_dict = model.state_dict()
        filtered_checkpoint_state_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items() if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint_state_dict, strict=False)
        
        print("Successfully loaded model state from checkpoint")
        if missing_keys:
            print(f"Warning: Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
            
    except Exception as e:
        print(f"Error loading model state: {e}")
        # Optionally, allow continuing without loaded weights for debugging structure
        cont = input("Continue without loaded weights? (y/n): ")
        if cont.lower() != 'y':
             return
    model.eval()

    # Create inference wrapper
    inference = ModelInference(model, tokenizer, device, ir_as_prompt=ir_as_prompt)

    # Create dataset
    dataset = SimpleSpectralSmilesDataset(
        data_dir=data_config.get('tokenized_dir', 'tokenized_baseline/data'),
        split=args.split,
        smiles_tokenizer=tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        max_smiles_len=model_config.get('max_seq_length', 512),
        max_nmr_len=model_config.get('max_nmr_length', 128),
        ir_as_prompt=ir_as_prompt,
        ir_tokenizer=ir_tokenizer
    )
    total_dataset_samples = len(dataset)
    num_samples = min(total_dataset_samples, args.max_samples)
    print(f"Processing {num_samples} samples from dataset split '{args.split}' (total available: {total_dataset_samples})")

    # Dictionary to hold aggregated metrics for each strategy
    results_by_strategy = {}
    # List of strategies to test
    strategies_to_test = ['Automatic'] + [f'Loops={l}' for l in args.fixed_loops]

    start_time = time.time()

    # --- Run Inference for Each Strategy ---
    for strategy_name in strategies_to_test:
        sample_metrics = []
        num_loops_for_decode = None
        
        if strategy_name == 'Automatic':
            if not automatic_loop_exit:
                 print("Skipping 'Automatic' strategy as automatic_loop_exit is False in config.")
                 continue
            # Use the model's configured max_loops; the model handles the exit internally.
            num_loops_for_decode = model.max_loops 
            print(f"Testing Greedy Decoding with Automatic Exit (max_loops={num_loops_for_decode})")
            desc = "Automatic Exit"
        else:
            # Extract fixed loop count from strategy name
            num_loops_for_decode = int(strategy_name.split('=')[1])
            print(f"Testing Greedy Decoding with Fixed Loops = {num_loops_for_decode}")
            desc = f"Fixed Loops={num_loops_for_decode}"

        for idx in tqdm(range(num_samples), desc=desc):
            target_tokens, (ir_tensor, _), nmr_tokens, _ = dataset[idx]
            
            # Ensure data is on the correct device (handle potential None for IR)
            ir_data = ir_tensor.to(device) if ir_tensor is not None else None
            nmr_tokens = nmr_tokens.to(device) if nmr_tokens is not None else None
            
            target_smiles = dataset.targets[idx] # Assumes targets are pre-loaded strings

            # Perform decoding
            # Note: The 'automatic_loop_exit' behavior is determined by how the model was *initialized*.
            # We only control the *number of loops* passed to the decode function here.
            # If model has auto_exit=True, passing model.max_loops triggers it.
            # Passing a smaller fixed number runs that fixed number (unless auto-exit triggers earlier).
            results = inference.decode(
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                strategy=DecodingStrategy.GREEDY_LOOP,
                max_len=model_config.get('max_seq_length', 512),
                num_loops=num_loops_for_decode,
                loops_representation=args.loops_representation
            )
            
            # Extract SMILES (handle potential extra outputs if loops_representation=True)
            generated_smiles = results[0] if isinstance(results, tuple) else results

            # Evaluate metrics for this sample
            metrics = evaluate_similarity(generated_smiles, target_smiles, strategy_name)
            sample_metrics.append(metrics)
            
            

        # Aggregate metrics over all samples for this strategy
        if sample_metrics:
            aggregated = combine_metrics(sample_metrics)
            results_by_strategy[strategy_name] = aggregated
        else:
             # Handle case where a strategy was skipped or produced no results
             results_by_strategy[strategy_name] = {
                 'valid_smiles': 0.0, 'exact_match': 0.0, 'avg_tanimoto': 0.0,
                 'avg_#mcs/#target': 0.0, 'avg_ecfp6_iou': 0.0
            }


    total_time = time.time() - start_time

    # --- Prepare and Display Results ---
    rows = []
    # Ensure consistent order: Automatic first, then fixed loops sorted numerically
    display_order = sorted(results_by_strategy.keys(), key=lambda x: (-1 if x == 'Automatic' else int(x.split('=')[1])))

    for strategy_name in display_order:
        metrics = results_by_strategy[strategy_name]
        rows.append({
            'Strategy': strategy_name,
            'Valid SMILES': f"{metrics.get('valid_smiles', 0.0):.2%}",
            'Exact Match': f"{metrics.get('exact_match', 0.0):.2%}",
            'Tanimoto': f"{metrics.get('avg_tanimoto', 0.0):.4f}",
            'MCS Ratio': f"{metrics.get('avg_#mcs/#target', 0.0):.4f}",
            'ECFP6 IoU': f"{metrics.get('avg_ecfp6_iou', 0.0):.4f}"
        })
        
    print("===== Aggregate Metrics Comparison =====")
    if rows:
        print(tabulate(rows, headers="keys", tablefmt="psql", showindex=False))
    else:
        print("No results to display.")
    print(f"Total time: {int(total_time // 60)} minutes {int(total_time % 60)} seconds")
    print(f"Tested on {num_samples} samples from '{args.split}' split.")

    # Save results to CSV file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"automatic_exit_comparison_{timestamp}.csv"
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(results_file, index=False)
        print(f"Saved aggregate metrics to {results_file}")
    else:
        print("No results generated, CSV file not saved.")

if __name__ == '__main__':
    main() 