import os
import torch
import json
from pathlib import Path
from datetime import datetime
import yaml
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from train_autoregressive import (
    load_config, 
    create_data_loaders,
    MultiModalToSMILESModel,
    SmilesTokenizer
)

def greedy_decode(model, nmr_data, ir_data, c_nmr_data, tokenizer, max_len=128, device=None):
    """
    Simple greedy decoding for SMILES generation.
    Args:
        model: The MultiModalToSMILESModel instance
        nmr_data: H-NMR spectral data tuple (data, domain) or None
        ir_data: IR spectral data tuple (data, domain) or None
        c_nmr_data: C-NMR spectral data tuple (data, domain) or None
        tokenizer: SmilesTokenizer instance
        max_len: Maximum sequence length for generation
        device: torch device to use
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Get tokenizer's special token IDs
    BOS_TOKEN_ID = tokenizer.cls_token_id  # [CLS] token is used as BOS
    EOS_TOKEN_ID = tokenizer.sep_token_id  # [SEP] token is used as EOS
    PAD_TOKEN_ID = tokenizer.pad_token_id
        
    model.eval()
    with torch.no_grad():
        # Get batch size from input data
        batch_size = 1  # Default
        if nmr_data is not None:
            batch_size = nmr_data[0].size(0) if isinstance(nmr_data, tuple) else nmr_data.size(0)
        elif ir_data is not None:
            batch_size = ir_data[0].size(0) if isinstance(ir_data, tuple) else ir_data.size(0)
        elif c_nmr_data is not None:
            batch_size = c_nmr_data[0].size(0) if isinstance(c_nmr_data, tuple) else c_nmr_data.size(0)

        # Start tokens for each sequence in the batch
        current_token = torch.tensor([[BOS_TOKEN_ID]] * batch_size, device=device)
        
        # Encode
        memory = model.encoder(nmr_data, ir_data, c_nmr_data)
        
        # Initialize storage for generated tokens
        generated_sequences = [[] for _ in range(batch_size)]
        for seq in generated_sequences:
            seq.append(BOS_TOKEN_ID)
        
        # Use model's max sequence length as the limit
        max_len = min(max_len, model.decoder.max_seq_length)
        
        finished_sequences = [False] * batch_size
        
        for _ in range(max_len):
            logits = model.decoder(current_token, memory)
            next_token = logits[:, -1:].argmax(dim=-1)
            
            # Update each sequence
            for i in range(batch_size):
                if not finished_sequences[i]:
                    token = next_token[i].item()
                    generated_sequences[i].append(token)
                    if token == EOS_TOKEN_ID:
                        finished_sequences[i] = True
            
            # Stop if all sequences are finished
            if all(finished_sequences):
                break
            
            current_token = torch.cat([current_token, next_token], dim=1)
        
        # Convert to tensor
        max_seq_len = max(len(seq) for seq in generated_sequences)
        padded_sequences = []
        for seq in generated_sequences:
            # Pad sequence to max length
            padded_seq = seq + [PAD_TOKEN_ID] * (max_seq_len - len(seq))
            padded_sequences.append(padded_seq)
        
        return torch.tensor(padded_sequences, device=device)

def find_latest_checkpoint():
    """Find the most recent checkpoint directory and its best model."""
    checkpoint_dir = Path('checkpoints')
    if not checkpoint_dir.exists():
        raise FileNotFoundError("No checkpoints directory found")
    
    # Get all checkpoint directories
    checkpoint_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoint directories found")
    
    # Sort by creation time and get the latest
    latest_dir = max(checkpoint_dirs, key=lambda x: datetime.strptime(x.name, '%Y%m%d_%H%M%S'))
    model_path = latest_dir / 'best_model.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"No best_model.pt found in {latest_dir}")
    
    return model_path, latest_dir

def evaluate_predictions(predictions, targets, verbose=True):
    """
    Evaluate model predictions vs. targets using canonical SMILES.
    Returns detailed metrics and optionally prints examples.
    """
    exact_matches = 0
    valid_smiles = 0
    tanimoto_scores = []
    detailed_results = []
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        result = {
            'prediction': pred,
            'target': target,
            'valid': False,
            'exact_match': False,
            'tanimoto': 0.0
        }
        
        # Remove spaces before creating molecules
        pred_no_spaces = pred.replace(" ", "")
        target_no_spaces = target.replace(" ", "")
        
        # Convert to RDKit molecules
        mol_pred = Chem.MolFromSmiles(pred_no_spaces)
        mol_target = Chem.MolFromSmiles(target_no_spaces)
        
        if mol_pred is not None:
            valid_smiles += 1
            result['valid'] = True
            
            if mol_target is not None:
                # Get canonical SMILES
                canon_pred = Chem.MolToSmiles(mol_pred, canonical=True)
                canon_target = Chem.MolToSmiles(mol_target, canonical=True)
                
                if canon_pred == canon_target:
                    exact_matches += 1
                    result['exact_match'] = True
                
                # Calculate Tanimoto similarity
                fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2)
                fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, 2)
                tanimoto = DataStructs.TanimotoSimilarity(fp_pred, fp_target)
                tanimoto_scores.append(tanimoto)
                result['tanimoto'] = tanimoto
                
                if verbose and i < 5:  # Add debug output for first 5 examples
                    print(f"\nCanonical SMILES comparison:")
                    print(f"Target (canonical):     {canon_target}")
                    print(f"Prediction (canonical): {canon_pred}")
        
        detailed_results.append(result)
        
        # Print some examples if verbose
        if verbose and i < 5:  # Print first 5 examples
            print(f"\nExample {i+1}:")
            print(f"Target:     {target}")
            print(f"Prediction: {pred}")
            print(f"Valid: {result['valid']}")
            print(f"Exact Match: {result['exact_match']}")
            print(f"Tanimoto: {result['tanimoto']:.3f}")
    
    metrics = {
        'exact_match': exact_matches / len(predictions),
        'valid_smiles': valid_smiles / len(predictions),
        'avg_tanimoto': sum(tanimoto_scores) / len(tanimoto_scores) if tanimoto_scores else 0.0
    }
    
    return metrics, detailed_results

def test_model(config_path=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load config
    config = load_config(config_path)
    
    # Initialize tokenizer
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.join(current_dir, 'vocab.txt')
    tokenizer = SmilesTokenizer(vocab_file=vocab_path)
    
    # Create data loaders (we'll only use the test loader)
    _, _, test_loader = create_data_loaders(tokenizer, config)
    
    # Find and load latest checkpoint
    checkpoint_path, checkpoint_dir = find_latest_checkpoint()
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    # Initialize model
    model = MultiModalToSMILESModel(
        vocab_size=len(tokenizer),
        max_seq_length=config['model']['max_seq_length'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        resample_size=config['model']['resample_size']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}, "
          f"global step {checkpoint['global_step']}")
    
    # Evaluate
    print("\nRunning inference on test set...")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            tgt_tokens, ir, h_nmr, c_nmr = batch
            
            # Move data to device and handle spectral data tuples
            tgt_tokens = tgt_tokens.to(device)
            
            if ir is not None:
                if isinstance(ir, tuple):
                    ir = (ir[0].to(device), ir[1].to(device))
                else:
                    ir = ir.to(device)
            
            if h_nmr is not None:
                if isinstance(h_nmr, tuple):
                    h_nmr = (h_nmr[0].to(device), h_nmr[1].to(device))
                else:
                    h_nmr = h_nmr.to(device)
            
            if c_nmr is not None:
                if isinstance(c_nmr, tuple):
                    c_nmr = (c_nmr[0].to(device), c_nmr[1].to(device))
                else:
                    c_nmr = c_nmr.to(device)
            
            # Generate predictions
            pred_tokens = greedy_decode(model, h_nmr, ir, c_nmr, tokenizer)
            
            # Decode predictions and targets
            for i in range(pred_tokens.size(0)):
                pred_smiles = tokenizer.decode(pred_tokens[i].tolist(), skip_special_tokens=True)
                target_smiles = tokenizer.decode(tgt_tokens[i].tolist(), skip_special_tokens=True)
                
                all_predictions.append(pred_smiles)
                all_targets.append(target_smiles)
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    metrics, detailed_results = evaluate_predictions(all_predictions, all_targets)
    
    # Print summary metrics
    print("\nTest Results:")
    print(f"Number of test examples: {len(all_predictions)}")
    print(f"Exact Match: {metrics['exact_match']:.2%}")
    print(f"Valid SMILES: {metrics['valid_smiles']:.2%}")
    print(f"Average Tanimoto: {metrics['avg_tanimoto']:.3f}")
    
    # Save detailed results
    results_file = checkpoint_dir / 'detailed_test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'detailed_results': detailed_results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create a histogram of Tanimoto scores
    tanimoto_scores = [r['tanimoto'] for r in detailed_results if r['valid']]
    if tanimoto_scores:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(tanimoto_scores, bins=50)
        plt.title('Distribution of Tanimoto Scores')
        plt.xlabel('Tanimoto Similarity')
        plt.ylabel('Count')
        plt.savefig(checkpoint_dir / 'tanimoto_distribution.png')
        plt.close()
        print(f"Tanimoto score distribution plot saved to: {checkpoint_dir / 'tanimoto_distribution.png'}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test trained SMILES generation model')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    test_model(args.config) 