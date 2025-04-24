#!/usr/bin/env python
import argparse
import torch
import os
import json
from pathlib import Path
import yaml
import numpy as np
from scipy.interpolate import interp1d  # Added for IR processing
import pandas as pd
from tabulate import tabulate
import time
from tqdm import tqdm

from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from inference.inference import ModelInference, DecodingStrategy

# Import evaluation metrics functions from logging_utils
try:
    from logging_utils import evaluate_predictions, aggregate_metrics
except ImportError:
    print("Warning: Could not import logging_utils. Using local implementation.")
    import rdkit
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdFMCS
    import numpy as np
    
    def evaluate_predictions(predictions, targets):
        """Evaluate SMILES prediction quality with RDKit metrics"""
        results = []
        for pred, target in zip(predictions, targets):
            # Process to handle spaces/clean up
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")
            
            mol_pred = Chem.MolFromSmiles(pred) if pred else None
            mol_target = Chem.MolFromSmiles(target) if target else None
            
            # Initialize metrics
            result = {
                'prediction': pred,
                'target': target,
                'valid_pred': mol_pred is not None,
                'valid_target': mol_target is not None,
                'exact_match': False,  # Initialize to False, will update if valid molecules
                'tanimoto': 0.0,
                '#mcs/#target': 0.0,
                'ecfp6_iou': 0.0
            }
            
            # Skip detailed metrics if either molecule is invalid
            if not mol_pred or not mol_target:
                results.append(result)
                continue
                
            # Generate canonical SMILES for exact match comparison
            canon_pred = Chem.MolToSmiles(mol_pred, canonical=True)
            canon_target = Chem.MolToSmiles(mol_target, canonical=True)
            result['exact_match'] = canon_pred == canon_target
                
            # Calculate Tanimoto similarity with Morgan fingerprints
            fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2)
            fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, 2)
            tanimoto = DataStructs.TanimotoSimilarity(fp_pred, fp_target)
            result['tanimoto'] = tanimoto
            
            # Calculate Maximum Common Substructure (MCS) size ratio
            try:
                mcs = rdFMCS.FindMCS([mol_pred, mol_target], timeout=1)
                if mcs and mcs.numAtoms > 0:
                    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                    result['#mcs/#target'] = mcs.numAtoms / mol_target.GetNumAtoms()
                else:
                    result['#mcs/#target'] = 0.0
            except:
                result['#mcs/#target'] = 0.0
            
            # Calculate ECFP6 (Morgan r=3) Intersection-over-Union
            fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 3, nBits=2048)
            fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, 3, nBits=2048)
            
            # Convert fingerprints to numpy arrays for set operations
            arr_pred = np.zeros((1,))
            arr_target = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp_pred, arr_pred)
            DataStructs.ConvertToNumpyArray(fp_target, arr_target)
            
            intersection = np.logical_and(arr_pred, arr_target).sum()
            union = np.logical_or(arr_pred, arr_target).sum()
            if union > 0:
                result['ecfp6_iou'] = intersection / union
            
            results.append(result)
        return results

    def aggregate_metrics(results, metric='tanimoto'):
        """
        Aggregate individual result metrics by selecting the best prediction
        according to the specified metric (default: tanimoto similarity)
        """
        n_samples = len(results)
        if n_samples == 0:
            return {
                'valid_smiles': 0.0,
                'exact_match': 0.0,
                'exact_match_all': 0.0,
                'avg_tanimoto': 0.0,
                'avg_#mcs/#target': 0.0,
                'avg_ecfp6_iou': 0.0
            }
        
        # Count valid predictions
        n_valid_pred = sum(1 for r in results if r['valid_pred'])
        n_valid_target = sum(1 for r in results if r['valid_target'])
        
        # Filter for valid molecules
        valid_results = [r for r in results if r['valid_pred'] and r['valid_target']]
        
        if not valid_results:
            return {
                'valid_smiles': n_valid_pred / n_samples if n_samples > 0 else 0.0,
                'exact_match': 0.0,
                'exact_match_all': 0.0,
                'avg_tanimoto': 0.0,
                'avg_#mcs/#target': 0.0,
                'avg_ecfp6_iou': 0.0
            }
        
        # Find the best prediction according to the specified metric
        best_result = max(valid_results, key=lambda x: x[metric])
        
        return {
            'valid_smiles': n_valid_pred / n_samples if n_samples > 0 else 0.0,
            'exact_match': 1.0 if best_result['exact_match'] else 0.0,
            'exact_match_all': sum(1 for r in results if r['exact_match']) / n_samples if n_samples > 0 else 0.0,
            'avg_tanimoto': best_result['tanimoto'],
            'avg_#mcs/#target': best_result['#mcs/#target'],
            'avg_ecfp6_iou': best_result['ecfp6_iou']
        }


def load_config(config_path=None):
    """Load configuration from a YAML file with default fallbacks."""
    # Default minimal configuration
    default_config = {
        'model': {
            'max_seq_length': 512,
            'max_nmr_length': 128,
            'max_memory_length': 128,
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'use_stablemax': False,
            #'width_basis': 13
        },
        'data': {
            'tokenized_dir': 'tokenized_baseline/data',
            'ir_as_prompt': False
        }
    }
    if config_path is not None:
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            update_dict(default_config, custom_config)
    return default_config


def load_raw_spectrum_tokens(file_path, spectral_tokenizer, max_len):
    """
    Load and tokenize raw NMR spectrum data from a text file.
    Expected format: two columns with domain and intensities.
    """
    tokens_list = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    intensity = float(parts[1])
                    # Round intensity to 2 decimals and convert to string token
                    token = str(round(intensity, 2))
                    tokens_list.append(token)
                except:
                    continue
    token_str = " ".join(tokens_list)
    tokens = token_str.split()
    token_ids = [spectral_tokenizer.get(t, spectral_tokenizer.get("<UNK>")) for t in tokens]
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    return torch.tensor(token_ids, dtype=torch.long)


def load_raw_ir(file_path, ir_as_prompt=False, ir_tokenizer=None):
    """
    Load raw IR spectral data from a text file and process according to 
    the approach in build_ir_vocab.py.
    
    Expected format: two columns with domain and intensities.
    
    If ir_as_prompt is True, tokenize the values using ir_tokenizer.
    Otherwise, return raw values as a tensor.
    """
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    value = float(parts[1])
                    values.append(value)
                except:
                    continue
    
    if ir_as_prompt:
        if ir_tokenizer is None:
            raise ValueError("ir_tokenizer is required when ir_as_prompt is True")
        
        # Process IR using the approach from build_ir_vocab.py
        ir_array = np.array(values)
        interpolation_points = 400  # This must match the expected length in the model
        
        # Interpolate to fixed number of points (400-4000 cm^-1 range as in build_ir_vocab.py)
        original_x = np.linspace(400, 4000, len(ir_array))
        interpolation_x = np.linspace(400, 4000, interpolation_points)
        
        # Handle edge case of too few points
        if len(ir_array) < 2:
            print("Warning: IR data has fewer than 2 points. Using zeros.")
            interp_ir = np.zeros(interpolation_points)
        else:
            # Use interpolation with extrapolation for points outside the range
            interp = interp1d(original_x, ir_array, bounds_error=False, fill_value="extrapolate")
            interp_ir = interp(interpolation_x)
        
        # Normalize to range [0, 100] (following build_ir_vocab.py)
        min_val = min(interp_ir)
        interp_ir = interp_ir + abs(min_val)
        max_val = max(interp_ir)
        if max_val > 0:  # Avoid division by zero
            interp_ir = (interp_ir / max_val) * 100
        
        # Round to integers and convert to strings (following build_ir_vocab.py)
        interp_ir = np.round(interp_ir, decimals=0).astype(int).astype(str)
        
        # Create token list with "IR" prefix followed by space-separated intensity values
        tokens = ["IR"]  # Start with IR prefix token
        
        # Convert string tokens to token IDs
        token_ids = [ir_tokenizer.get("IR", ir_tokenizer.get("<UNK>", 0))]
        for token_str in interp_ir:
            token_id = ir_tokenizer.get(token_str, ir_tokenizer.get("<UNK>", 0))
            token_ids.append(token_id)
        
        return torch.tensor(token_ids, dtype=torch.long)
    else:
        return torch.tensor(values, dtype=torch.float32)


# Simple class to match the dataset structure used in search_and_infer.py
class SimpleSpectralSmilesDataset:
    """
    A simplified dataset loader to read source and target SMILES, and IR data from numpy files.
    """
    def __init__(self, data_dir, split='test', smiles_tokenizer=None, spectral_tokenizer=None, 
                 max_smiles_len=512, max_nmr_len=128, ir_as_prompt=False, ir_tokenizer=None):
        self.data_dir = Path(data_dir)
        self.smiles_tokenizer = smiles_tokenizer
        self.spectral_tokenizer = spectral_tokenizer
        self.max_smiles_len = max_smiles_len
        self.max_nmr_len = max_nmr_len
        self.split = split
        self.ir_as_prompt = ir_as_prompt
        self.ir_tokenizer = ir_tokenizer

        if split == "all":
            splits = ["train", "val", "test"]
            self.sources = []
            self.targets = []
            ir_data_list = []
            for s in splits:
                src_file = self.data_dir / f"src-{s}.txt"
                tgt_file = self.data_dir / f"tgt-{s}.txt"
                if src_file.exists() and tgt_file.exists():
                    with open(src_file) as f:
                        src_data = [line.strip() for line in f]
                    with open(tgt_file) as f:
                        tgt_data = [line.strip().replace(" ", "") for line in f]
                    self.sources.extend(src_data)
                    self.targets.extend(tgt_data)

                    ir_path = self.data_dir / f"ir-{s}.npy"
                    if ir_path.exists():
                        try:
                            part = np.memmap(ir_path, dtype='float32', mode='r', shape=None)
                            array_shape = part.shape
                            if len(array_shape) == 1:
                                num_samples = len(src_data)
                                feature_dim = array_shape[0] // num_samples
                                part = part.reshape(num_samples, feature_dim)
                            ir_data_list.append(part)
                        except Exception as e:
                            print(f"[Dataset] Failed to load IR data for split {s}: {e}")
            if ir_data_list:
                self.ir_data = np.concatenate(ir_data_list, axis=0)
                print(f"[Dataset] Loaded combined IR data with shape: {self.ir_data.shape}")
            else:
                self.ir_data = None
        else:
            # Original logic for a single split
            src_file = self.data_dir / f"src-{split}.txt"
            tgt_file = self.data_dir / f"tgt-{split}.txt"
            with open(src_file) as f:
                self.sources = [line.strip() for line in f]
            with open(tgt_file) as f:
                self.targets = [line.strip().replace(" ", "") for line in f]
            ir_path = self.data_dir / f"ir-{split}.npy"
            self.ir_data = None
            if ir_path.exists():
                try:
                    self.ir_data = np.memmap(ir_path, dtype='float32', mode='r', shape=None)
                    array_shape = self.ir_data.shape
                    if len(array_shape) == 1:
                        num_samples = len(self.sources)
                        feature_dim = array_shape[0] // num_samples
                        self.ir_data = self.ir_data.reshape(num_samples, feature_dim)
                    print(f"[Dataset] Loaded IR data with shape: {self.ir_data.shape}")
                except Exception as e:
                    print(f"[Dataset] Failed to load IR data: {e}")
                    self.ir_data = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # For inference we need target tokens, nmr tokens, and IR data
        target_seq = self.targets[idx]
        target_tokens = self.smiles_tokenizer.encode(
            target_seq,
            add_special_tokens=True,
            max_length=self.max_smiles_len,
            truncation=True
        )
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        source_seq = self.sources[idx]
        nmr_tokens = source_seq.split()
        nmr_token_ids = [self.spectral_tokenizer.get(token, self.spectral_tokenizer.get("<UNK>")) for token in nmr_tokens]
        if len(nmr_token_ids) > self.max_nmr_len:
            nmr_token_ids = nmr_token_ids[:self.max_nmr_len]
        nmr_tokens = torch.tensor(nmr_token_ids, dtype=torch.long)

        ir_tensor = None
        if self.ir_data is not None:
            raw_ir_data = self.ir_data[idx].copy()
            
            if self.ir_as_prompt:
                # Process IR data following build_ir_vocab.py approach
                interpolation_points = 400
                
                # If raw_ir_data is already exactly the right size, assume it's pre-processed
                # Otherwise, interpolate to get fixed dimensions
                if len(raw_ir_data) != interpolation_points:
                    # Interpolate to fixed number of points
                    original_x = np.linspace(400, 4000, len(raw_ir_data))
                    interpolation_x = np.linspace(400, 4000, interpolation_points)
                    
                    # Handle edge case of too few points
                    if len(raw_ir_data) < 2:
                        print(f"Warning: IR data for sample {idx} has fewer than 2 points. Using zeros.")
                        ir_data = np.zeros(interpolation_points)
                    else:
                        # Use interpolation with extrapolation for points outside the range
                        interp = interp1d(original_x, raw_ir_data, bounds_error=False, fill_value="extrapolate")
                        ir_data = interp(interpolation_x)
                else:
                    ir_data = raw_ir_data
                
                # Normalize to range [0, 100]
                min_val = min(ir_data)
                ir_data = ir_data + abs(min_val)
                max_val = max(ir_data)
                if max_val > 0:  # Avoid division by zero
                    ir_data = (ir_data / max_val) * 100
                
                # Round to integers and convert to strings
                ir_data = np.round(ir_data, decimals=0).astype(int).astype(str)
                
                # Convert to tokens
                token_ids = [self.ir_tokenizer.get("IR", self.ir_tokenizer.get("<UNK>", 0))]
                for token_str in ir_data:
                    token_id = self.ir_tokenizer.get(token_str, self.ir_tokenizer.get("<UNK>", 0))
                    token_ids.append(token_id)
                
                ir_tensor = torch.tensor(token_ids, dtype=torch.long)
            else:
                ir_tensor = torch.tensor(raw_ir_data, dtype=torch.float32)

        return target_tokens, (ir_tensor, None), nmr_tokens, None


def detect_ir_as_prompt(checkpoint, config=None):
    """
    Detect whether the model was trained with ir_as_prompt=True by examining
    the checkpoint or the config.
    
    Returns:
        bool: Whether ir_as_prompt should be True
        dict: Additional parameters needed (like ir_vocab_size)
    """
    # First check if it's specified in config
    if config and 'data' in config and 'ir_as_prompt' in config['data']:
        ir_as_prompt = config['data']['ir_as_prompt']
        print(f"Using ir_as_prompt={ir_as_prompt} from config")
        
        # If True, try to get ir_vocab_size from config
        extra_params = {}
        if ir_as_prompt and 'ir_vocab_size' in config['data']:
            extra_params['ir_vocab_size'] = config['data']['ir_vocab_size']
        
        return ir_as_prompt, extra_params
    
    # Otherwise try to detect from checkpoint state_dict
    ir_as_prompt = False
    extra_params = {}
    
    # Look for telltale signs in the state dict
    state_dict = checkpoint['model_state_dict']
    
    # If 'ir_embed.weight' exists, it's likely ir_as_prompt=True
    if 'ir_embed.weight' in state_dict:
        ir_as_prompt = True
        # Get ir_vocab_size from the embedding weight shape
        ir_vocab_size = state_dict['ir_embed.weight'].shape[0]
        extra_params['ir_vocab_size'] = ir_vocab_size
        print(f"Detected ir_as_prompt=True from checkpoint (ir_vocab_size={ir_vocab_size})")
    
    # Check if IR encoder weights are missing
    encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.ir_encoder')]
    if not encoder_keys and not ir_as_prompt:
        # This is a bit risky - absence of encoder doesn't guarantee ir_as_prompt
        # But it's a good hint
        print("Warning: No IR encoder found in checkpoint, but ir_as_prompt not detected.")
        print("This could indicate a mismatch between the checkpoint and the model architecture.")
    
    return ir_as_prompt, extra_params


def get_ir_tokenizer(config):
    """
    Get IR tokenizer from config or create a default one based on the 
    build_ir_vocab.py approach.
    """
    # Try to load IR tokenizer from a file if specified in config
    if config and 'data' in config and 'ir_tokenizer_path' in config['data']:
        ir_tokenizer_path = config['data']['ir_tokenizer_path']
        try:
            with open(ir_tokenizer_path, 'r') as f:
                ir_tokenizer = json.load(f)
            print(f"Loaded IR tokenizer from {ir_tokenizer_path}")
            return ir_tokenizer
        except Exception as e:
            print(f"Failed to load IR tokenizer: {e}")
    
    # Create a default IR tokenizer following build_ir_vocab.py approach
    print("Creating default IR tokenizer for integer values 0-100")
    
    # Initialize with special tokens as in build_ir_vocab.py
    vocab_set = ['<PAD>', '<UNK>', 'IR']
    
    # Add tokens for integer values 0-100
    for i in range(0, 101):
        vocab_set.append(str(i))
    
    # Create token-to-id mapping
    token_to_id = {token: idx for idx, token in enumerate(vocab_set)}
    
    print(f"Created default IR tokenizer with {len(token_to_id)} tokens")
    print(f"Special tokens: {token_to_id['<PAD>']}, {token_to_id['<UNK>']}, {token_to_id['IR']}")
    
    return token_to_id


def evaluate_similarity(predictions, target, method_name=""):
    """
    Evaluate similarity metrics between generated SMILES and target.
    
    Args:
        predictions: List of predicted SMILES strings
        target: Target SMILES string (ground truth)
        method_name: Name of the decoding method
        
    Returns:
        Dictionary of metrics
    """
    # Create a list of the same target for each prediction
    targets = [target] * len(predictions)
    
    # Evaluate predictions using functions from logging_utils
    detailed_results = evaluate_predictions(predictions, targets)
    metrics = aggregate_metrics(detailed_results)
    
    # Print metrics for this method
    print(f"\n----- Similarity Metrics for {method_name} -----")
    print(f"Valid SMILES Rate: {metrics['valid_smiles']:.2%}")
    print(f"Exact Match Rate: {metrics['exact_match']:.2%}")
    print(f"Tanimoto Similarity: {metrics['avg_tanimoto']:.4f}")
    print(f"MCS Ratio: {metrics['avg_#mcs/#target']:.4f}")
    print(f"ECFP6 IoU: {metrics['avg_ecfp6_iou']:.4f}")
    
    # Add method name to metrics
    metrics['method'] = method_name
    
    return metrics


def combine_metrics(metrics_list):
    """
    Combine metrics from multiple test examples into one aggregate result.
    
    Args:
        metrics_list: List of metrics dictionaries
        
    Returns:
        Dictionary of combined metrics
    """
    # Skip empty list
    if not metrics_list:
        return None
    
    # Initialize result with keys from the first metrics dict
    keys = metrics_list[0].keys()
    combined = {k: [] for k in keys if not k == 'method'}
    
    # Add method name if it exists in the first metrics dict
    if 'method' in metrics_list[0]:
        combined['method'] = metrics_list[0]['method']
    
    # Collect all values
    for metrics in metrics_list:
        for k, v in metrics.items():
            if k != 'method':
                # Ensure the key exists in combined before appending
                if k not in combined:
                    combined[k] = []
                combined[k].append(v)
    
    # Calculate averages - handle non-numeric values correctly
    result = {}
    for k, v in combined.items():
        # Skip empty lists
        if not v:
            continue
            
        # Check if values are numeric or strings
        if all(isinstance(x, (int, float, bool, np.number)) for x in v if x is not None):
            # For numeric values, calculate mean, ignoring None values
            valid_values = [x for x in v if x is not None]
            if valid_values:
                result[k] = np.mean(valid_values)
            else:
                result[k] = None # Or 0.0, depending on desired behavior for all None
        else:
            # For non-numeric values (like strings), use the first non-None value
            # This assumes these values should be the same across all metrics
            first_valid = next((item for item in v if item is not None), None)
            result[k] = first_valid
    
    # Add method name back if it exists
    if 'method' in combined:
        result['method'] = combined['method']
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Test different inference mechanisms')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--raw_nmr', type=str, default=None, help='Path to raw NMR spectrum file')
    parser.add_argument('--raw_ir', type=str, default=None, help='Path to raw IR spectrum file')
    parser.add_argument('--dataset_test', action='store_true', help='Test on a dataset sample instead of raw files')
    parser.add_argument('--full_dataset_test', action='store_true', help='Test on the entire dataset and report aggregate metrics')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (only if --dataset_test or --full_dataset_test)')
    parser.add_argument('--index', type=int, default=0, help='Index in dataset to test (only if --dataset_test)')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of examples to process in parallel (only with --full_dataset_test)')
    parser.add_argument('--max_examples', type=int, default=None, help='Maximum number of test examples to process (only with --full_dataset_test)')
    parser.add_argument('--strategies', type=str, default='greedy,beam,sampling,nucleus,entropix', help='Comma-separated list of decoding strategies to test (greedy,beam,sampling,nucleus,entropix,greedy_loop)')
    parser.add_argument('--ir_as_prompt', action='store_true', help='Use IR as prompt tokens')
    parser.add_argument('--no_ir_as_prompt', action='store_true', help='Do not use IR as prompt tokens')
    parser.add_argument('--entropy_threshold', type=float, default=0.6939, help='Entropy threshold for Entropix decoding')
    parser.add_argument('--varentropy_threshold', type=float, default=1.3781, help='Varentropy threshold for Entropix decoding')
    parser.add_argument('--max_loops', type=int, default=10, help='Maximum number of middle layer loops for high entropy states')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Directory to save inference results')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizers
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.join(current_dir, 'training/vocab.txt')
    tokenizer = SmilesTokenizer(vocab_file=vocab_path)
    
    nmr_vocab_path = Path(config['data']['tokenized_dir']).parent / 'vocab.json'
    if not nmr_vocab_path.exists():
        raise FileNotFoundError(f"NMR vocabulary not found at {nmr_vocab_path}")
    with open(nmr_vocab_path) as f:
        nmr_tokenizer = json.load(f)
    
    # Load checkpoint first to detect ir_as_prompt
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
        'ir_as_prompt': ir_as_prompt
    }
    # Added to ensure the correct IR encoder type and max_loops are used
    model_kwargs['ir_encoder_type'] = config['model'].get('ir_encoder_type', 'regular')
    model_kwargs['max_loops'] = args.max_loops
    
    # Add ir_vocab_size if needed
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
        print("This could be due to a mismatch between the model architecture and the checkpoint.")
        print("Please check that the model configuration matches the checkpoint.")
        return
    
    model.eval()
    
    # Determine which strategies to use based on args.strategies
    if args.strategies.lower() == 'all':
        strategies = ["greedy", "beam", "sampling", "nucleus", "entropix", "greedy_loop"]
    else:
        strategies = [s.strip().lower() for s in args.strategies.split(',')]
        valid_strategies = ["greedy", "beam", "sampling", "nucleus", "entropix", "greedy_loop"]
        for s in strategies:
            if s not in valid_strategies:
                raise ValueError(f"Invalid strategy '{s}'. Valid options are: {', '.join(valid_strategies)}")
    
    print(f"Testing strategies: {', '.join(strategies)}")
    
    # Create inference wrapper
    inference = ModelInference(model, tokenizer, device, ir_as_prompt=ir_as_prompt)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test on full dataset
    if args.full_dataset_test:
        print(f"\n===== Testing on full dataset (split={args.split}) =====")
        
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
        
        # Determine number of examples to process
        num_examples = len(dataset)
        if args.max_examples is not None:
            num_examples = min(num_examples, args.max_examples)
        
        print(f"Processing {num_examples} examples from dataset")
        
        # Dictionary to collect metrics for each strategy
        all_strategy_metrics = {strategy: [] for strategy in strategies}
        
        # Process examples in batches
        batch_size = args.batch_size
        total_batches = (num_examples + batch_size - 1) // batch_size
        
        # Track time
        start_time = time.time()
        
        # Process each batch
        for batch_idx in tqdm(range(total_batches)):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_examples)
            batch_size_actual = batch_end - batch_start
            
            # Process each example in the batch
            for i in range(batch_start, batch_end):
                target_tokens, (ir_tensor, _), nmr_tokens, _ = dataset[i]
                if ir_tensor is not None:
                    ir_data = ir_tensor.to(device)
                else:
                    ir_data = None
                nmr_tokens = nmr_tokens.to(device)
                
                target_smiles = dataset.targets[i]
                
                # Process each strategy for this example
                for strategy in strategies:
                    start_decode_time = time.time()
                    results = None
                    metrics = None
                    
                    if strategy == "greedy":
                        results = inference.decode(
                            nmr_tokens=nmr_tokens,
                            ir_data=ir_data,
                            strategy=DecodingStrategy.GREEDY,
                            max_len=config['model']['max_seq_length']
                        )
                        decode_time = time.time() - start_decode_time
                        metrics = evaluate_similarity(results, target_smiles, "Greedy Decoding")
                    
                    elif strategy == "beam":
                        results = inference.decode(
                            nmr_tokens=nmr_tokens,
                            ir_data=ir_data,
                            strategy=DecodingStrategy.BEAM,
                            max_len=config['model']['max_seq_length'],
                            beam_width=5
                        )
                        decode_time = time.time() - start_decode_time
                        metrics = evaluate_similarity(results, target_smiles, "Beam Search")
                    
                    elif strategy == "sampling":
                        results = inference.decode(
                            nmr_tokens=nmr_tokens,
                            ir_data=ir_data,
                            strategy=DecodingStrategy.SAMPLING,
                            max_len=config['model']['max_seq_length'],
                            temperature=1.0
                        )
                        decode_time = time.time() - start_decode_time
                        metrics = evaluate_similarity(results, target_smiles, "Sampling")
                    
                    elif strategy == "nucleus":
                        results = inference.decode(
                            nmr_tokens=nmr_tokens,
                            ir_data=ir_data,
                            strategy=DecodingStrategy.NUCLEUS,
                            max_len=config['model']['max_seq_length'],
                            temperature=1.0,
                            top_p=0.9
                        )
                        decode_time = time.time() - start_decode_time
                        metrics = evaluate_similarity(results, target_smiles, "Nucleus Sampling")
                    
                    elif strategy == "entropix":
                        results = inference.decode(
                            nmr_tokens=nmr_tokens,
                            ir_data=ir_data,
                            strategy=DecodingStrategy.ENTROPIX,
                            max_len=config['model']['max_seq_length'],
                            top_k=5,
                            entropy_threshold=args.entropy_threshold,
                            varentropy_threshold=args.varentropy_threshold,
                            max_loops=args.max_loops
                        )
                        decode_time = time.time() - start_decode_time
                        metrics = evaluate_similarity(results, target_smiles, "Entropix")
                        
                    elif strategy == "greedy_loop":
                        # Time the entire loop process for greedy_loop
                        loop_times = []
                        all_loop_results = []
                        for loop_count in range(args.max_loops):
                            loop_start_time = time.time()
                            loop_results = inference.decode(
                                nmr_tokens=nmr_tokens,
                                ir_data=ir_data,
                                strategy=DecodingStrategy.GREEDY_LOOP,
                                max_len=config['model']['max_seq_length'],
                                num_loops=loop_count
                            )
                            loop_duration = time.time() - loop_start_time
                            loop_times.append(loop_duration)
                            all_loop_results.append(loop_results) # Store results for each loop count if needed
                        
                        # Use the results from the last loop for evaluation
                        results = all_loop_results[-1] if all_loop_results else []
                        # Total time is the time for the last iteration or sum? Let's use last iteration time for consistency?
                        # Or maybe sum makes more sense as it represents total computation? Let's use the total time for the whole process.
                        decode_time = time.time() - start_decode_time # Total time for all loops
                        metrics = evaluate_similarity(results, target_smiles, f"Greedy Loop (max_loops={args.max_loops})")
                        # We could also average loop_times, but let's stick to total time for now.
                        
                    # Add timing information to metrics
                    if metrics is not None:
                        metrics['duration'] = decode_time
                        all_strategy_metrics[strategy].append(metrics)
                        
            # Show progress after each batch
            elapsed_time = time.time() - start_time
            examples_processed = batch_end
            examples_remaining = num_examples - batch_end
            examples_per_second = examples_processed / elapsed_time if elapsed_time > 0 else 0
            
            # Estimate time remaining
            if examples_per_second > 0:
                time_remaining = examples_remaining / examples_per_second
                time_remaining_str = f"{int(time_remaining // 60)}m {int(time_remaining % 60)}s"
            else:
                time_remaining_str = "Unknown"
            
            # Print progress
            print(f"\rProcessed {examples_processed}/{num_examples} examples. "
                  f"Speed: {examples_per_second:.2f} ex/s. Est. time remaining: {time_remaining_str}", 
                  end="")
        
        # Calculate aggregate metrics for each strategy
        aggregate_metrics = {}
        for strategy, metrics_list in all_strategy_metrics.items():
            aggregate_metrics[strategy] = combine_metrics(metrics_list)
        
        # Print aggregate metrics
        print("\n\n===== Aggregate Metrics =====")
        metrics_rows = []
        for strategy, metrics in aggregate_metrics.items():
            if metrics:
                # Format duration if available
                duration_str = f"{metrics.get('duration', 0.0):.4f}s" if metrics.get('duration') is not None else "N/A"
                
                metrics_rows.append({
                    'Method': strategy.capitalize().replace('_', ' '),
                    'Valid SMILES': f"{metrics['valid_smiles']:.2%}",
                    'Exact Match': f"{metrics['exact_match']:.2%}",
                    'Tanimoto': f"{metrics['avg_tanimoto']:.4f}",
                    'MCS Ratio': f"{metrics['avg_#mcs/#target']:.4f}",
                    'ECFP6 IoU': f"{metrics['avg_ecfp6_iou']:.4f}",
                    'Avg Time': duration_str  # Add timing info
                })
        
        # Convert to DataFrame for nice printing
        metrics_df = pd.DataFrame(metrics_rows)
        
        # Print table
        print(tabulate(metrics_df, headers='keys', tablefmt='psql', showindex=False))
        
        # Save metrics to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"inference_results_{timestamp}.csv"
        # Add the Avg Time column to the DataFrame before saving
        metrics_df_save = metrics_df.copy() # Avoid modifying the printed df
        metrics_df_save['Avg Time (s)'] = [metrics.get('duration', None) for metrics in aggregate_metrics.values() if metrics] # Add raw seconds for CSV
        metrics_df_save.drop(columns=['Avg Time'], inplace=True) # Remove formatted string version
        metrics_df_save.to_csv(results_file, index=False)
        print(f"\nSaved results to {results_file}")
        
        # Also save raw metrics (numbers only) for further analysis
        raw_metrics = {}
        for strategy, metrics in aggregate_metrics.items():
            if metrics:
                raw_metrics[strategy] = {
                    'valid_smiles': metrics['valid_smiles'],
                    'exact_match': metrics['exact_match'],
                    'avg_tanimoto': metrics['avg_tanimoto'],
                    'avg_#mcs/#target': metrics['avg_#mcs/#target'],
                    'avg_ecfp6_iou': metrics['avg_ecfp6_iou'],
                    'avg_duration': metrics.get('duration', None) # Add timing info
                }
        
        raw_file = output_dir / f"inference_raw_metrics_{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump(raw_metrics, f, indent=2)
        print(f"Saved raw metrics to {raw_file}")
        
        # Print total time
        total_time = time.time() - start_time
        print(f"\nTotal time: {int(total_time // 60)} minutes {int(total_time % 60)} seconds")
        print(f"Average time per example: {total_time / num_examples:.2f} seconds")
        
        # Done with full dataset test
        return
                
    # Load spectral data - use the same approach as search_and_infer.py
    nmr_tokens = None
    ir_data = None
    target_smiles = None
    
    # Either use dataset or raw files
    if args.dataset_test:
        print(f"Testing on dataset sample from split '{args.split}', index {args.index}")
        # Create dataset like in search_and_infer.py
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
        
        if args.index >= len(dataset):
            raise ValueError(f"Index {args.index} out of range for dataset of size {len(dataset)}")
            
        target_tokens, (ir_tensor, _), nmr_tokens, _ = dataset[args.index]
        if ir_tensor is not None:
            ir_data = ir_tensor.to(device)
        nmr_tokens = nmr_tokens.to(device)
        
        target_smiles = dataset.targets[args.index]
        print(f"Target SMILES: {target_smiles}")
        
    else:
        # Use raw spectral files
        if args.raw_nmr:
            print(f"Loading NMR data from {args.raw_nmr}")
            nmr_tokens = load_raw_spectrum_tokens(
                args.raw_nmr,
                nmr_tokenizer,
                config['model']['max_nmr_length']
            ).to(device)
        
        if args.raw_ir:
            print(f"Loading IR data from {args.raw_ir}")
            ir_data = load_raw_ir(
                args.raw_ir,
                ir_as_prompt=ir_as_prompt,
                ir_tokenizer=ir_tokenizer
            ).to(device)
        
        # Target SMILES is unknown when using raw files
        target_smiles = None
    
    if nmr_tokens is None and ir_data is None:
        print("Warning: No input spectral data provided. Results may not be meaningful.")
    
    # Dictionary to store metrics for all methods
    all_metrics = {}
    all_times = {} # Dictionary to store timing for single sample test
    
    # Test different decoding strategies
    print("\n===== Testing Different Decoding Strategies =====")
    
    # Apply selected strategies
    if "greedy" in strategies:
        # 1. Greedy decoding
        print("\n1. Greedy Decoding")
        start_decode_time = time.time()
        greedy_results = inference.decode(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            strategy=DecodingStrategy.GREEDY,
            max_len=config['model']['max_seq_length']
        )
        decode_time = time.time() - start_decode_time
        all_times["Greedy"] = decode_time
        print(f"  Time: {decode_time:.4f}s")
        for i, result in enumerate(greedy_results):
            print(f"  Result {i+1}: {result}")
        
        # Evaluate greedy results if we have a target
        if target_smiles:
            greedy_metrics = evaluate_similarity(greedy_results, target_smiles, "Greedy Decoding")
            all_metrics["Greedy"] = greedy_metrics
    
    if "beam" in strategies:
        # 2. Beam search
        print("\n2. Beam Search (width=5)")
        start_decode_time = time.time()
        beam_results = inference.decode(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            strategy=DecodingStrategy.BEAM,
            max_len=config['model']['max_seq_length'],
            beam_width=5
        )
        decode_time = time.time() - start_decode_time
        all_times["Beam"] = decode_time
        print(f"  Time: {decode_time:.4f}s")
        for i, result in enumerate(beam_results):
            print(f"  Result {i+1}: {result}")
        
        # Evaluate beam search results if we have a target
        if target_smiles:
            beam_metrics = evaluate_similarity(beam_results, target_smiles, "Beam Search")
            all_metrics["Beam"] = beam_metrics
    
    if "sampling" in strategies:
        # 3. Sampling with temperature
        print("\n3. Sampling (temperature=1.0)")
        start_decode_time = time.time()
        sampling_results = inference.decode(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            strategy=DecodingStrategy.SAMPLING,
            max_len=config['model']['max_seq_length'],
            temperature=1.0
        )
        decode_time = time.time() - start_decode_time
        all_times["Sampling"] = decode_time
        print(f"  Time: {decode_time:.4f}s")
        for i, result in enumerate(sampling_results):
            print(f"  Result {i+1}: {result}")
        
        # Evaluate sampling results if we have a target
        if target_smiles:
            sampling_metrics = evaluate_similarity(sampling_results, target_smiles, "Sampling")
            all_metrics["Sampling"] = sampling_metrics
    
    if "nucleus" in strategies:
        # 4. Nucleus sampling (top-p)
        print("\n4. Nucleus Sampling (top-p=0.9)")
        start_decode_time = time.time()
        nucleus_results = inference.decode(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            strategy=DecodingStrategy.NUCLEUS,
            max_len=config['model']['max_seq_length'],
            temperature=1.0,
            top_p=0.9
        )
        decode_time = time.time() - start_decode_time
        all_times["Nucleus"] = decode_time
        print(f"  Time: {decode_time:.4f}s")
        for i, result in enumerate(nucleus_results):
            print(f"  Result {i+1}: {result}")
        
        # Evaluate nucleus sampling results if we have a target
        if target_smiles:
            nucleus_metrics = evaluate_similarity(nucleus_results, target_smiles, "Nucleus Sampling")
            all_metrics["Nucleus"] = nucleus_metrics
    
    if "entropix" in strategies:
        # 5. Entropix tree search
        print(f"\n5. Entropix Tree Search (entropy_threshold={args.entropy_threshold}, varentropy_threshold={args.varentropy_threshold}, max_loops={args.max_loops})")
        start_decode_time = time.time()
        entropix_results = inference.decode(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            strategy=DecodingStrategy.ENTROPIX,
            max_len=config['model']['max_seq_length'],
            top_k=5,
            entropy_threshold=args.entropy_threshold,
            varentropy_threshold=args.varentropy_threshold,
            max_loops=args.max_loops
        )
        decode_time = time.time() - start_decode_time
        all_times["Entropix"] = decode_time
        print(f"  Time: {decode_time:.4f}s")
        for i, result in enumerate(entropix_results):
            print(f"  Result {i+1}: {result}")
        
        # Evaluate entropix results if we have a target
        if target_smiles:
            entropix_metrics = evaluate_similarity(entropix_results, target_smiles, "Entropix")
            all_metrics["Entropix"] = entropix_metrics
    
    if "greedy_loop" in strategies:
        print("\nX. Greedy Loop Decoding with Varying Layer Loop Counts")
        greedy_loop_results = []  # Initialize list to store loop results
        loop_times = []
        overall_start_time = time.time()
        # Test with different numbers of loops
        for loop_count in range(args.max_loops):
            loop_start_time = time.time()
            results = inference.decode(
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                strategy=DecodingStrategy.GREEDY_LOOP,
                max_len=config['model']['max_seq_length'],
                num_loops=loop_count
            )
            loop_duration = time.time() - loop_start_time
            loop_times.append(loop_duration)
            greedy_loop_results.append(results)  # Store results from current loop count
            print(f"\nResults for Greedy Loop Decoding with num_loops = {loop_count}: (Time: {loop_duration:.4f}s)")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: {result}")
            if target_smiles:
                metrics = evaluate_similarity(results, target_smiles, f"Greedy Loop (num_loops={loop_count})")
                # Store metrics for each loop count separately if needed
                all_metrics[f"GreedyLoop_{loop_count}"] = metrics 
        
        overall_decode_time = time.time() - overall_start_time
        all_times["GreedyLoop"] = overall_decode_time # Store overall time for the strategy
        print(f"\nOverall time for Greedy Loop strategy (up to {args.max_loops} loops): {overall_decode_time:.4f}s")
        
        # Use last iteration result for comparison tables and final metrics
        selected_greedy_loop_results = greedy_loop_results[-1] if greedy_loop_results else []
        if target_smiles and f"GreedyLoop_{args.max_loops - 1}" in all_metrics:
             # Use metrics from the last loop count for the main "GreedyLoop" entry
            all_metrics["GreedyLoop"] = all_metrics[f"GreedyLoop_{args.max_loops - 1}"]


    # Compare results
    print("\n===== Results Comparison =====")
    all_results = {}
    if "greedy" in strategies:
        all_results["Greedy"] = greedy_results[0]
    if "beam" in strategies:
        all_results["Beam"] = beam_results[0]
    if "sampling" in strategies:
        all_results["Sampling"] = sampling_results[0]
    if "nucleus" in strategies:
        all_results["Nucleus"] = nucleus_results[0]
    if "entropix" in strategies:
        all_results["Entropix"] = entropix_results[0]
    if "greedy_loop" in strategies:
        all_results["GreedyLoop"] = selected_greedy_loop_results
    
    for method, result in all_results.items():
        print(f"{method}: {result}")
    
    # Compare metrics in a table if we have a target
    if target_smiles and all_metrics:
        print("\n===== Metrics Comparison =====")
        
        # Convert metrics to DataFrame for nice printing
        metrics_df = pd.DataFrame([
            {
                'Method': method,
                'Valid SMILES': f"{metrics['valid_smiles']:.2%}",
                'Exact Match': f"{metrics['exact_match']:.2%}",
                'Tanimoto': f"{metrics['avg_tanimoto']:.4f}",
                'MCS Ratio': f"{metrics['avg_#mcs/#target']:.4f}",
                'ECFP6 IoU': f"{metrics['avg_ecfp6_iou']:.4f}",
                'Time (s)': f"{all_times.get(method, 0.0):.4f}" # Add timing info
            }
            for method, metrics in all_metrics.items() if method in all_times # Ensure method has timing info
        ])
        
        # Print table
        print(tabulate(metrics_df, headers='keys', tablefmt='psql', showindex=False))


if __name__ == '__main__':
    main() 