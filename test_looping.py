#!/usr/bin/env python
"""
This script performs a full-dataset looping test by applying greedy loop decoding 
with various num_loops values. It processes up to a specified number of samples (default 25)
from a given dataset split and aggregates the metrics.
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
# (Make sure test_inference.py is in your PYTHONPATH or same directory.)
from test_inference import load_config, get_ir_tokenizer, detect_ir_as_prompt, SimpleSpectralSmilesDataset, evaluate_similarity, combine_metrics

def main():
    parser = argparse.ArgumentParser(description="Full Dataset Looping Test for Greedy Loop Decoding")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration YAML file')
    parser.add_argument('--max_samples', type=int, default=50, help='Maximum number of dataset samples to process')
    parser.add_argument('--max_loops', type=int, default=10, help='Maximum number of loops for greedy loop decoding')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Directory to save results')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize SMILES tokenizer
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.join(current_dir, 'training/vocab.txt')
    tokenizer = SmilesTokenizer(vocab_file=vocab_path)

    # Load NMR tokenizer (vocab.json should be located relative to tokenized_dir)
    nmr_vocab_path = Path(config['data']['tokenized_dir']).parent / 'vocab.json'
    if not nmr_vocab_path.exists():
        raise FileNotFoundError(f"NMR vocabulary not found at {nmr_vocab_path}")
    with open(nmr_vocab_path) as f:
        nmr_tokenizer = json.load(f)

    # Load checkpoint and detect IR settings
    checkpoint = torch.load(args.checkpoint, map_location=device)
    auto_ir_as_prompt, extra_params = detect_ir_as_prompt(checkpoint, config)
    ir_as_prompt = auto_ir_as_prompt

    # Get IR tokenizer if needed
    ir_tokenizer = None
    if ir_as_prompt:
        ir_tokenizer = get_ir_tokenizer(config)
        print(f"IR tokenizer has {len(ir_tokenizer)} tokens")

    # Create the model with appropriate parameters
    smiles_vocab_size = len(tokenizer)
    token_ids = list(nmr_tokenizer.values())
    nmr_vocab_size = max(token_ids) + 1
    model_kwargs = {
        'smiles_vocab_size': smiles_vocab_size,
        'nmr_vocab_size': nmr_vocab_size,
        'max_seq_length': config['model']['max_seq_length'],
        'max_nmr_length': config['model']['max_nmr_length'],
        'max_memory_length': config['model']['max_memory_length'],
        'embed_dim': config['model']['embed_dim'],
        'num_heads': config['model']['num_heads'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'verbose': False,
        'use_stablemax': config['model'].get('use_stablemax', False),
        'ir_as_prompt': ir_as_prompt,
        'ir_encoder_type': config['model'].get('ir_encoder_type', 'regular'),
        'max_loops': args.max_loops
    }
    if ir_as_prompt:
        if 'ir_vocab_size' in extra_params:
            model_kwargs['ir_vocab_size'] = extra_params['ir_vocab_size']
        else:
            model_kwargs['ir_vocab_size'] = len(ir_tokenizer)

    model = MultiModalToSMILESModel(**model_kwargs).to(device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded model state from checkpoint")
    except Exception as e:
        print(f"Error loading model state: {e}")
        return
    model.eval()

    # Create inference wrapper
    inference = ModelInference(model, tokenizer, device, ir_as_prompt=ir_as_prompt)

    # Create dataset
    dataset = SimpleSpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        split=args.split,
        smiles_tokenizer=tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length'],
        ir_as_prompt=ir_as_prompt,
        ir_tokenizer=ir_tokenizer
    )
    total_dataset_samples = len(dataset)
    num_samples = min(total_dataset_samples, args.max_samples)
    print(f"Processing {num_samples} samples from dataset split '{args.split}' (total available: {total_dataset_samples})")

    # Dictionary to hold aggregated metrics for each loop count
    loop_metric_results = {}

    start_time = time.time()
    # Loop over different num_loops values for greedy loop decoding
    for loop_count in range(args.max_loops):
        sample_metrics = []
        print(f"\nTesting Greedy Loop Decoding with num_loops = {loop_count}")
        for idx in tqdm(range(num_samples), desc=f"Loop Count {loop_count}"):
            target_tokens, (ir_tensor, _), nmr_tokens, _ = dataset[idx]
            if ir_tensor is not None:
                ir_data = ir_tensor.to(device)
            else:
                ir_data = None
            nmr_tokens = nmr_tokens.to(device)
            target_smiles = dataset.targets[idx]
            results = inference.decode(
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                strategy=DecodingStrategy.GREEDY_LOOP,
                max_len=config['model']['max_seq_length'],
                num_loops=loop_count
            )
            # Evaluate metrics for this sample
            metrics = evaluate_similarity(results, target_smiles, f"Greedy Loop (num_loops={loop_count})")
            sample_metrics.append(metrics)
        # Aggregate metrics over all samples for this loop count
        aggregated = combine_metrics(sample_metrics)
        loop_metric_results[loop_count] = aggregated

    total_time = time.time() - start_time

    # Prepare results table
    rows = []
    for loop_count, metrics in loop_metric_results.items():
        rows.append({
            'Num Loops': loop_count,
            'Valid SMILES': f"{metrics['valid_smiles']:.2%}",
            'Exact Match': f"{metrics['exact_match']:.2%}",
            'Tanimoto': f"{metrics['avg_tanimoto']:.4f}",
            'MCS Ratio': f"{metrics['avg_#mcs/#target']:.4f}",
            'ECFP6 IoU': f"{metrics['avg_ecfp6_iou']:.4f}"
        })
    print("\n===== Aggregate Metrics for Greedy Loop Decoding =====")
    print(tabulate(rows, headers="keys", tablefmt="psql", showindex=False))
    print(f"\nTotal time: {int(total_time // 60)} minutes {int(total_time % 60)} seconds")

    # Save results to CSV file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"greedy_loop_full_dataset_{timestamp}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(results_file, index=False)
    print(f"Saved aggregate metrics to {results_file}")

if __name__ == '__main__':
    main()