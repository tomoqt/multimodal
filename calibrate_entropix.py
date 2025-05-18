#!/usr/bin/env python
import argparse
import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from tqdm import tqdm
from scipy import stats
from inference.inference import ModelInference, DecodingStrategy
from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer

# Import helper functions from test_inference.py
from test_inference import load_config, load_raw_spectrum_tokens, load_raw_ir, detect_ir_as_prompt, get_ir_tokenizer
from test_inference import SimpleSpectralSmilesDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate Entropix thresholds using test dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='calibration_results', help='Directory to save calibration results')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to analyze (0 for all)')
    parser.add_argument('--max_seq_steps', type=int, default=90, help='Maximum number of sequence steps to analyze per sample')
    parser.add_argument('--ir_as_prompt', action='store_true', help='Use IR as prompt tokens')
    parser.add_argument('--no_ir_as_prompt', action='store_true', help='Do not use IR as prompt tokens')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top candidates to consider for threshold analysis')
    parser.add_argument('--max_loops', type=int, default=3, help='Maximum number of loops to use in analysis')
    return parser.parse_args()


def collect_entropy_stats(model, tokenizer, dataset, device, ir_as_prompt, num_samples=0, max_seq_steps=50, top_k=5):
    """
    Collect entropy and varentropy statistics from model predictions on a dataset.
    
    Args:
        model: The model to analyze
        tokenizer: The SMILES tokenizer
        dataset: The dataset to use
        device: The device to run on
        ir_as_prompt: Whether IR is used as prompt tokens
        num_samples: Number of samples to analyze (0 for all)
        max_seq_steps: Maximum number of sequence steps to analyze per sample
        top_k: Number of top candidates to consider
        
    Returns:
        A pandas DataFrame with entropy and varentropy statistics
    """
    model.eval()
    inference = ModelInference(model, tokenizer, device, ir_as_prompt=ir_as_prompt)

    # Data to collect
    stats = []
    
    # Determine number of samples to process
    if num_samples <= 0 or num_samples > len(dataset):
        num_samples = len(dataset)
    
    # Process dataset samples
    for idx in tqdm(range(num_samples), desc="Analyzing samples"):
        # Get sample from dataset
        target_tokens, (ir_tensor, _), nmr_tokens, _ = dataset[idx]
        if ir_tensor is not None:
            ir_data = ir_tensor.to(device)
        else:
            ir_data = None
        
        if nmr_tokens is not None:
            nmr_tokens = nmr_tokens.to(device)
        
        target_smiles = dataset.targets[idx]
        
        # Initialize sequence with BOS token
        current_token = torch.tensor([[inference.bos_token_id]], device=device)
        
        # Prepare inputs and encode
        (nmr_tokens_batch, ir_data_batch, _), batch_size = inference.prepare_inputs(
            nmr_tokens, ir_data, None
        )
        memory = inference.encode_inputs(nmr_tokens_batch, ir_data_batch, None)
        
        # Iterate through sequence steps
        sequence = [inference.bos_token_id]
        for step in range(max_seq_steps):
            # Get logits from model
            logits = model.decoder(tgt=current_token, memory=memory, nmr_tokens=nmr_tokens_batch)
            logits = logits[0, -1, :]  # Take last token predictions
            
            # Calculate log probabilities, entropy, and varentropy
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = inference.calculate_entropy(log_probs)
            varentropy = inference.calculate_varentropy(log_probs)
            
            # Get next token (greedy decoding)
            next_token = log_probs.argmax().unsqueeze(0).unsqueeze(0)
            token_id = next_token.item()
            
            # Record statistics
            token_text = tokenizer.convert_ids_to_tokens([token_id])[0] if token_id < len(tokenizer) else "<UNK>"
            stats.append({
                'sample_idx': idx,
                'target_smiles': target_smiles,
                'step': step,
                'token_id': token_id,
                'token_text': token_text,
                'entropy': entropy,
                'varentropy': varentropy
            })
            
            # Add token to sequence
            sequence.append(token_id)
            
            # Exit if EOS token is generated
            if token_id == inference.eos_token_id:
                break
                
            # Update current token for next step
            current_token = torch.cat([current_token, next_token], dim=1)
    
    # Convert to DataFrame
    return pd.DataFrame(stats)


def calculate_threshold_recommendations(df):
    """
    Calculate recommended threshold values based on quantiles.
    
    Args:
        df: DataFrame with entropy and varentropy values
        
    Returns:
        Dictionary of recommended threshold values
    """
    recommendations = {}
    
    # Calculate quantiles for entropy
    entropy_quantiles = df['entropy'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
    recommendations['entropy'] = {
        'quantiles': entropy_quantiles,
        'mean': df['entropy'].mean(),
        'std': df['entropy'].std(),
        'recommended_thresholds': {
            'conservative': entropy_quantiles[0.5],  # 50th percentile
            'moderate': entropy_quantiles[0.75],     # 75th percentile
            'aggressive': entropy_quantiles[0.9]     # 90th percentile
        }
    }
    
    # Calculate quantiles for varentropy
    varentropy_quantiles = df['varentropy'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
    recommendations['varentropy'] = {
        'quantiles': varentropy_quantiles,
        'mean': df['varentropy'].mean(),
        'std': df['varentropy'].std(),
        'recommended_thresholds': {
            'conservative': varentropy_quantiles[0.5],  # 50th percentile
            'moderate': varentropy_quantiles[0.75],     # 75th percentile
            'aggressive': varentropy_quantiles[0.9]     # 90th percentile
        }
    }
    
    # Add example Entropix parameters
    recommendations['entropix_decode_params'] = {
        'conservative': {
            'entropy_threshold': entropy_quantiles[0.5],
            'varentropy_threshold': varentropy_quantiles[0.5],
            'max_loops': 3,
            'automatic_loop_exit': False,
            'top_k': 5,
            'loop_increase_step': 1
        },
        'moderate': {
            'entropy_threshold': entropy_quantiles[0.75],
            'varentropy_threshold': varentropy_quantiles[0.75],
            'max_loops': 3,
            'automatic_loop_exit': False,
            'top_k': 5,
            'loop_increase_step': 1
        },
        'aggressive': {
            'entropy_threshold': entropy_quantiles[0.9],
            'varentropy_threshold': varentropy_quantiles[0.9],
            'max_loops': 3,
            'automatic_loop_exit': True,
            'automatic_loop_exit_threshold': 0.01,
            'top_k': 5,
            'loop_increase_step': 1
        }
    }
    
    return recommendations


def plot_distributions(df, output_dir):
    """
    Create visualizations of entropy and varentropy distributions.
    
    Args:
        df: DataFrame with entropy and varentropy values
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))
    
    # Plot histograms for entropy
    plt.subplot(2, 2, 1)
    sns.histplot(df['entropy'], kde=True)
    plt.title('Distribution of Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.axvline(df['entropy'].quantile(0.5), color='r', linestyle='--', label='50th percentile')
    plt.axvline(df['entropy'].quantile(0.75), color='g', linestyle='--', label='75th percentile')
    plt.axvline(df['entropy'].quantile(0.9), color='b', linestyle='--', label='90th percentile')
    plt.legend()
    
    # Plot histograms for varentropy
    plt.subplot(2, 2, 2)
    sns.histplot(df['varentropy'], kde=True)
    plt.title('Distribution of Varentropy')
    plt.xlabel('Varentropy')
    plt.ylabel('Frequency')
    plt.axvline(df['varentropy'].quantile(0.5), color='r', linestyle='--', label='50th percentile')
    plt.axvline(df['varentropy'].quantile(0.75), color='g', linestyle='--', label='75th percentile')
    plt.axvline(df['varentropy'].quantile(0.9), color='b', linestyle='--', label='90th percentile')
    plt.legend()
    
    # Plot joint distribution
    plt.subplot(2, 1, 2)
    sns.scatterplot(data=df, x='entropy', y='varentropy', alpha=0.5)
    plt.title('Joint Distribution of Entropy and Varentropy')
    plt.xlabel('Entropy')
    plt.ylabel('Varentropy')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'entropy_varentropy_distributions.png'))
    plt.close()
    
    # Create a heatmap to visualize the 2D density
    plt.figure(figsize=(10, 8))
    h = plt.hist2d(df['entropy'], df['varentropy'], bins=50, cmap='viridis')
    plt.colorbar(h[3])
    plt.title('2D Histogram of Entropy and Varentropy')
    plt.xlabel('Entropy')
    plt.ylabel('Varentropy')
    plt.savefig(os.path.join(output_dir, 'entropy_varentropy_heatmap.png'))
    plt.close()
    
    # Plot entropy by sequence position
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='step', y='entropy')
    plt.title('Entropy by Sequence Position')
    plt.xlabel('Sequence Position')
    plt.ylabel('Entropy')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_by_position.png'))
    plt.close()
    
    # Plot varentropy by sequence position
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='step', y='varentropy')
    plt.title('Varentropy by Sequence Position')
    plt.xlabel('Sequence Position')
    plt.ylabel('Varentropy')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'varentropy_by_position.png'))
    plt.close()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizers
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = 'training/vocab.txt'
    tokenizer = SmilesTokenizer(vocab_file=vocab_path)
    
    nmr_vocab_path = Path(config['data']['tokenized_dir']).parent / 'vocab.json'
    if not nmr_vocab_path.exists():
        raise FileNotFoundError(f"NMR vocabulary not found at {nmr_vocab_path}")
    with open(nmr_vocab_path) as f:
        nmr_tokenizer = json.load(f)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Decide whether to use ir_as_prompt
    if args.ir_as_prompt and args.no_ir_as_prompt:
        raise ValueError("Cannot specify both --ir_as_prompt and --no_ir_as_prompt")
    
    # Determine if we should use ir_as_prompt based on checkpoint, config, and command line args
    auto_ir_as_prompt, extra_params = detect_ir_as_prompt(checkpoint, config)
    
    if args.ir_as_prompt:
        ir_as_prompt = True
    elif args.no_ir_as_prompt:
        ir_as_prompt = False
    else:
        ir_as_prompt = auto_ir_as_prompt
    
    print(f"Using ir_as_prompt={ir_as_prompt}")
    
    # Get IR tokenizer if needed
    ir_tokenizer = None
    if ir_as_prompt:
        ir_tokenizer = get_ir_tokenizer(config)
        print(f"IR tokenizer has {len(ir_tokenizer)} tokens")
    
    # Initialize model
    smiles_vocab_size = len(tokenizer)
    token_ids = list(nmr_tokenizer.values())
    nmr_vocab_size = max(token_ids) + 1
    
    # Create model with appropriate parameters
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
        'max_loops': args.max_loops,
        'use_rmsnorm': config['model'].get('use_rmsnorm', True)
    }
    
    # Add ir_vocab_size if needed
    if ir_as_prompt:
        if 'ir_vocab_size' in extra_params:
            model_kwargs['ir_vocab_size'] = extra_params['ir_vocab_size']
        else:
            model_kwargs['ir_vocab_size'] = len(ir_tokenizer)
    
    # Initialize model
    model = MultiModalToSMILESModel(**model_kwargs).to(device)
    
    # Load model state
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded model state from checkpoint")
    except Exception as e:
        print(f"Error loading model state: {e}")
        print("This could be due to a mismatch between the model architecture and the checkpoint.")
        print("Please check that the model configuration matches the checkpoint.")
        return
    
    # Load dataset
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
    
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Collect entropy and varentropy statistics
    print(f"Analyzing entropy and varentropy distributions for {args.num_samples} samples...")
    stats_df = collect_entropy_stats(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        ir_as_prompt=ir_as_prompt,
        num_samples=args.num_samples,
        max_seq_steps=args.max_seq_steps,
        top_k=args.top_k
    )
    
    # Save raw statistics
    stats_df.to_csv(os.path.join(args.output_dir, 'entropy_stats.csv'), index=False)
    print(f"Saved raw statistics to {os.path.join(args.output_dir, 'entropy_stats.csv')}")
    
    # Calculate threshold recommendations
    recommendations = calculate_threshold_recommendations(stats_df)
    
    # Save recommendations
    with open(os.path.join(args.output_dir, 'threshold_recommendations.json'), 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"Saved threshold recommendations to {os.path.join(args.output_dir, 'threshold_recommendations.json')}")
    
    # Create visualizations
    print("Creating visualizations...")
    plot_distributions(stats_df, args.output_dir)
    print(f"Saved visualizations to {args.output_dir}")
    
    # Print summary
    print("\n===== Entropy Distribution =====")
    print(f"Mean: {recommendations['entropy']['mean']:.4f}")
    print(f"Std Dev: {recommendations['entropy']['std']:.4f}")
    print("Recommended thresholds:")
    print(f"  Conservative: {recommendations['entropy']['recommended_thresholds']['conservative']:.4f}")
    print(f"  Moderate: {recommendations['entropy']['recommended_thresholds']['moderate']:.4f}")
    print(f"  Aggressive: {recommendations['entropy']['recommended_thresholds']['aggressive']:.4f}")
    
    print("\n===== Varentropy Distribution =====")
    print(f"Mean: {recommendations['varentropy']['mean']:.4f}")
    print(f"Std Dev: {recommendations['varentropy']['std']:.4f}")
    print("Recommended thresholds:")
    print(f"  Conservative: {recommendations['varentropy']['recommended_thresholds']['conservative']:.4f}")
    print(f"  Moderate: {recommendations['varentropy']['recommended_thresholds']['moderate']:.4f}")
    print(f"  Aggressive: {recommendations['varentropy']['recommended_thresholds']['aggressive']:.4f}")
    
    # Print example command for using Entropix decoding
    entropy_threshold = recommendations['entropy']['recommended_thresholds']['moderate']
    varentropy_threshold = recommendations['varentropy']['recommended_thresholds']['moderate']
    print("\nExample command for Entropix decoding with moderate thresholds:")
    print(f"python inference.py --checkpoint {args.checkpoint} --config {args.config} --strategy entropix --entropy_threshold {entropy_threshold:.4f} --varentropy_threshold {varentropy_threshold:.4f} --max_loops {args.max_loops}")
    
    print("\nCalibration complete! Use these threshold values with the Entropix decoding strategy.")


if __name__ == '__main__':
    main() 