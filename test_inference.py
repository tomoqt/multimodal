#!/usr/bin/env python
import argparse
import torch
import os
import json
from pathlib import Path
import yaml
import numpy as np
from scipy.interpolate import interp1d  # Added for IR processing

from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from inference import ModelInference, DecodingStrategy


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
            'width_basis': 13
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


def main():
    parser = argparse.ArgumentParser(description='Test different inference mechanisms')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--raw_nmr', type=str, default=None, help='Path to raw NMR spectrum file')
    parser.add_argument('--raw_ir', type=str, default=None, help='Path to raw IR spectrum file')
    parser.add_argument('--dataset_test', action='store_true', help='Test on a dataset sample instead of raw files')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (only if --dataset_test)')
    parser.add_argument('--index', type=int, default=0, help='Index in dataset to test (only if --dataset_test)')
    parser.add_argument('--ir_as_prompt', action='store_true', help='Use IR as prompt tokens')
    parser.add_argument('--no_ir_as_prompt', action='store_true', help='Do not use IR as prompt tokens')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizers
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.join(current_dir, 'vocab.txt')
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
    
    # Load spectral data - use the same approach as search_and_infer.py
    nmr_tokens = None
    ir_data = None
    
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
        
        print(f"Target SMILES: {dataset.targets[args.index]}")
        
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
    
    if nmr_tokens is None and ir_data is None:
        print("Warning: No input spectral data provided. Results may not be meaningful.")
    
    # Create inference wrapper (pass ir_as_prompt to ensure consistent handling)
    inference = ModelInference(model, tokenizer, device, ir_as_prompt=ir_as_prompt)
    
    # Test different decoding strategies
    print("\n===== Testing Different Decoding Strategies =====")
    
    # 1. Greedy decoding
    print("\n1. Greedy Decoding")
    greedy_results = inference.decode(
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        strategy=DecodingStrategy.GREEDY,
        max_len=config['model']['max_seq_length']
    )
    for i, result in enumerate(greedy_results):
        print(f"  Result {i+1}: {result}")
    
    # 2. Beam search
    print("\n2. Beam Search (width=5)")
    beam_results = inference.decode(
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        strategy=DecodingStrategy.BEAM,
        max_len=config['model']['max_seq_length'],
        beam_width=5
    )
    for i, result in enumerate(beam_results):
        print(f"  Result {i+1}: {result}")
    
    # 3. Sampling with temperature
    print("\n3. Sampling (temperature=1.0)")
    sampling_results = inference.decode(
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        strategy=DecodingStrategy.SAMPLING,
        max_len=config['model']['max_seq_length'],
        temperature=1.0
    )
    for i, result in enumerate(sampling_results):
        print(f"  Result {i+1}: {result}")
    
    # 4. Nucleus sampling (top-p)
    print("\n4. Nucleus Sampling (top-p=0.9)")
    nucleus_results = inference.decode(
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        strategy=DecodingStrategy.NUCLEUS,
        max_len=config['model']['max_seq_length'],
        temperature=1.0,
        top_p=0.9
    )
    for i, result in enumerate(nucleus_results):
        print(f"  Result {i+1}: {result}")
    
    # Compare results
    print("\n===== Results Comparison =====")
    print(f"Greedy: {greedy_results[0]}")
    print(f"Beam search: {beam_results[0]}")
    print(f"Sampling: {sampling_results[0]}")
    print(f"Nucleus: {nucleus_results[0]}")


if __name__ == '__main__':
    main() 