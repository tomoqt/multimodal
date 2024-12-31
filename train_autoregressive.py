# =======================
# File: train_autoregressive.py
# =======================
"""
Script to train a MultiModalToSMILESModel using either:
- Preprocessed .pt files created by create_training_data.py, or
- Parquet files (loaded in one go, no row-group chunking).

Key Steps:
1) Loads spectral + SMILES data
2) Tokenizes SMILES
3) Demonstrates a basic training loop with teacher forcing
4) Shows a minimal inference (greedy decode) function
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from datetime import datetime
from sklearn.model_selection import train_test_split
import json
from rdkit import Chem
from pathlib import Path
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
from tqdm import tqdm
import pandas as pd
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse
import time
import pyarrow.parquet as pq

# Import our custom tokenizer
from models.smiles_tokenizer import SmilesTokenizer
from models.multimodal_to_smiles import MultiModalToSMILESModel

current_dir = os.path.dirname(os.path.realpath(__file__))
vocab_path = os.path.join(current_dir, 'vocab.txt')
tokenizer = SmilesTokenizer(vocab_file=vocab_path)

# -------------------------------------------------------------------------
# Warmup + Cosine Annealing With Restarts Scheduler
# -------------------------------------------------------------------------
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Combines linear warmup with cosine annealing and warm restarts.
    """
    def __init__(self, optimizer, warmup_steps, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_0_initial = T_0
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.completed_warmup = False
        self.n_restarts = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        # Cosine annealing with warm restarts
        if not self.completed_warmup:
            self.completed_warmup = True
            self.T_cur = 0
        
        # Check for restart
        if self.T_cur >= self.T_0:
            self.T_cur = 0
            self.T_0 = self.T_0 * self.T_mult
            self.n_restarts += 1
        
        progress = self.T_cur / self.T_0
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        self.T_cur += 1
        
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._last_lr = self.get_lr()
        
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr


# -------------------------------------------------------------------------
# Dataset / DataLoader for .pt Binaries
# -------------------------------------------------------------------------
class SpectralSmilesDataset(Dataset):
    """
    A PyTorch Dataset that reads sharded .pt files.
    Handles both preprocessed and raw data formats.
    """
    def __init__(self, data_dir, tokenizer, max_len=128, preprocessed=True):
        print("\n[Dataset] Initializing SpectralSmilesDataset...")
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocessed = preprocessed
        
        # Load shard paths
        print("[Dataset] Looking for shard files...")
        pattern = "*_processed.pt" if preprocessed else "*_rg*.pt"
        self.shard_paths = sorted(self.data_dir.glob(pattern))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found in {data_dir} with pattern {pattern}")
        print(f"[Dataset] Found {len(self.shard_paths)} shard files")
            
        # Try to load cached index
        cache_name = "index_cache_processed.pt" if preprocessed else "index_cache.pt"
        cache_path = self.data_dir / cache_name
        if cache_path.exists():
            print("[Dataset] Loading cached index map...")
            self.index_map = torch.load(cache_path)
            print(f"[Dataset] Loaded index map with {len(self.index_map)} entries")
        else:
            print("[Dataset] Building new index map (this may take a while)...")
            self.index_map = []
            for shard_path in tqdm(self.shard_paths, desc="Loading shards"):
                shard_data = torch.load(shard_path, map_location='cpu')
                self.index_map.extend((shard_path, row_idx) for row_idx in shard_data.keys())
            torch.save(self.index_map, cache_path)
            print(f"[Dataset] Created and cached index map with {len(self.index_map)} entries")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        shard_path, row_idx = self.index_map[idx]
        shard_data = torch.load(shard_path, map_location='cpu')
        data = shard_data[row_idx]

        # Tokenize SMILES
        smiles = data['smiles']
        start_time = time.perf_counter()
        tokens = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )
        encode_time = time.perf_counter() - start_time
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Tokenization took {encode_time:.4f} seconds")

        # Get spectral data - each should be a tuple of (data, domain)
        spectra = data['spectra']
        ir = spectra.get('ir')  # Should be (data, domain) tuple
        h_nmr = spectra.get('h_nmr')  # Should be (data, domain) tuple
        c_nmr = spectra.get('c_nmr')  # Should be (data, domain) tuple

        return tokens, ir, h_nmr, c_nmr


# -------------------------------------------------------------------------
# Dataset / DataLoader for Parquet Files (No Chunking)
# -------------------------------------------------------------------------
class ParquetSpectralDataset(Dataset):
    """
    A PyTorch Dataset that reads multiple Parquet files using row groups.
    Maintains an index mapping to efficiently load data from specific row groups.
    """
    def __init__(self, data_dir, tokenizer, max_len=128):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load metadata if needed
        meta_path = self.data_dir / "meta_data/meta_data_dict.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        with open(meta_path) as f:
            self.meta_data = json.load(f)

        # Store domain information from metadata
        self.ir_domain = torch.tensor(self.meta_data["ir_spectra"]["dimensions"], dtype=torch.float32)
        self.h_nmr_domain = torch.tensor(self.meta_data["h_nmr_spectra"]["dimensions"], dtype=torch.float32)
        self.c_nmr_domain = torch.tensor(self.meta_data["c_nmr_spectra"]["dimensions"], dtype=torch.float32)

        # Find all parquet files
        print("[Dataset] Looking for parquet files...")
        self.parquet_files = sorted(self.data_dir.glob("*.parquet"))
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        print(f"[Dataset] Found {len(self.parquet_files)} parquet files")

        # Create index mapping for row groups
        print("[Dataset] Building index mapping...")
        self.index_map = []
        
        for file_idx, pq_file in enumerate(tqdm(self.parquet_files, desc="Indexing parquet files")):
            try:
                pf = pq.ParquetFile(pq_file)
                for rg_idx in range(pf.num_row_groups):
                    # Get row count for this row group
                    row_count = pf.read_row_group(rg_idx, columns=['smiles']).num_rows
                    # Add entries for each row in this row group
                    self.index_map.extend(
                        (file_idx, rg_idx, row_idx) 
                        for row_idx in range(row_count)
                    )
            except Exception as e:
                print(f"Warning: Error reading file {pq_file}: {str(e)}")
                continue
        
        if not self.index_map:
            raise ValueError("No valid data found in parquet files")
        
        print(f"[Dataset] Created index map with {len(self.index_map)} entries")

        # Cache for the currently loaded row group
        self._current_file_idx = None
        self._current_rg_idx = None
        self._current_rg_data = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, rg_idx, row_idx = self.index_map[idx]
        
        # Load the row group if it's not already loaded
        if self._current_file_idx != file_idx or self._current_rg_idx != rg_idx:
            pf = pq.ParquetFile(self.parquet_files[file_idx])
            self._current_rg_data = pf.read_row_group(
                rg_idx,
                columns=['smiles', 'ir_spectra', 'h_nmr_spectra', 
                        'c_nmr_spectra']
            ).to_pandas()
            self._current_file_idx = file_idx
            self._current_rg_idx = rg_idx

        row_data = self._current_rg_data.iloc[row_idx]

        # Tokenize SMILES
        smiles = row_data['smiles']
        tokens = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )

        def to_tensor(x, spectrum_type):
            """Convert spectrum to tensor with correct domain information"""
            if x is None:
                return None
            
            try:
                # Handle 1D spectra normally
                tensor = torch.tensor(x, dtype=torch.float32)
                if tensor.dim() == 1:
                    if spectrum_type == 'ir':
                        return (tensor, self.ir_domain)
                    elif spectrum_type == 'h_nmr':
                        return (tensor, self.h_nmr_domain)
                    elif spectrum_type == 'c_nmr':
                        return (tensor, self.c_nmr_domain)
                return tensor
                
            except Exception as e:
                print(f"Warning: Error converting {spectrum_type} data to tensor: {e}")
                return None

        # Convert spectra to tuples of (data, domain)
        ir_spectra = to_tensor(row_data['ir_spectra'], 'ir')
        h_nmr_spectra = to_tensor(row_data['h_nmr_spectra'], 'h_nmr')
        c_nmr_spectra = to_tensor(row_data['c_nmr_spectra'], 'c_nmr')

        return tokens, ir_spectra, h_nmr_spectra, c_nmr_spectra


# -------------------------------------------------------------------------
# Collate Function - Moved outside to be picklable
# -------------------------------------------------------------------------
def collate_fn(batch):
    """
    Custom collate: pad tokens, preserve spectral data tuples.
    """
    # Unzip the batch into separate lists
    all_tokens, all_ir, all_h_nmr, all_c_nmr = zip(*batch)
    
    # Helper function to stack spectral data tuples
    def maybe_stack_with_domain(items):
        if items[0] is not None:
            # Stack data tensors along batch dimension
            data = torch.stack([item[0] for item in items], dim=0)
            # Use first domain tensor (they're all the same)
            domain = items[0][1]
            return (data, domain)
        return None

    # Stack spectral data preserving tuple structure
    ir_batch = maybe_stack_with_domain(all_ir) if all_ir[0] is not None else None
    h_nmr_batch = maybe_stack_with_domain(all_h_nmr) if all_h_nmr[0] is not None else None
    c_nmr_batch = maybe_stack_with_domain(all_c_nmr) if all_c_nmr[0] is not None else None

    # Pad tokens
    max_len = max(len(t) for t in all_tokens)
    padded_tokens = []
    for seq in all_tokens:
        pad_amount = max_len - len(seq)
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        if pad_amount > 0:
            pad_tensor = torch.full((pad_amount,), tokenizer.pad_token_id, dtype=torch.long)
            seq_tensor = torch.cat([seq_tensor, pad_tensor], dim=0)
        padded_tokens.append(seq_tensor)
    token_batch = torch.stack(padded_tokens, dim=0)

    return token_batch, ir_batch, h_nmr_batch, c_nmr_batch


def create_data_loaders(tokenizer, config):
    """
    Create train/val/test DataLoaders using parameters from config.
    """
    print("\n[DataLoader] Creating data loaders...")
    print(f"[DataLoader] Using {'Parquet' if config['data']['use_parquet'] else 'Binary'} dataset")
    
    if config['data']['use_parquet']:
        dataset = ParquetSpectralDataset(
            config['data']['data_dir'], 
            tokenizer, 
            config['model']['max_seq_length']
        )
    else:
        dataset = SpectralSmilesDataset(
            config['data']['binary_dir'], 
            tokenizer, 
            config['model']['max_seq_length'],
            preprocessed=config['data'].get('preprocessed', False)
        )
    
    print(f"[DataLoader] Total dataset size: {len(dataset)}")
    
    # Split indices
    print("[DataLoader] Splitting dataset into train/val/test...")
    all_indices = list(range(len(dataset)))
    train_val_indices, test_indices = train_test_split(
        all_indices, 
        test_size=config['data'].get('test_size', 20), 
        random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=config['data'].get('val_size', 0.1), 
        random_state=42
    )
    
    print(f"[DataLoader] Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create loaders with simple collate function
    print("[DataLoader] Creating train loader...")
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print("[DataLoader] Creating validation loader...")
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print("[DataLoader] Creating test loader...")
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=config['training'].get('test_batch_size', 1),
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print("[DataLoader] Data loaders created successfully")
    return train_loader, val_loader, test_loader


# -------------------------------------------------------------------------
# Setup: Config, Tokenizer, Model, Data Loaders
# -------------------------------------------------------------------------
def load_config(config_path=None):
    """Load config from yaml file, falling back to defaults if not specified"""
    default_config = {
        'model': {
            'max_seq_length': 128,
            'embed_dim': 768,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'resample_size': 1000
        },
        'training': {
            'batch_size': 32,
            'test_batch_size': 1,
            'num_epochs': 1,
            'learning_rate': 1.0e-4,
            'min_learning_rate': 1.0e-6,
            'validation_frequency': 500,
            'logging_frequency': 100,
            'save_frequency': 1000
        },
        'scheduler': {
            'warmup_steps': 100,
            'T0': 5,
            'T_mult': 2
        },
        'data': {
            'use_parquet': False,
            'data_dir': "data_extraction/multimodal_spectroscopic_dataset",
            'binary_dir': "training_binaries",
            'preprocessed': False,
            'test_size': 20,
            'val_size': 0.1
        },
        'wandb': {
            'project': "smiles-generation",
            'base_run_name': "smiles_gen",
            'log_examples': True
        }
    }
    
    if config_path:
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
            # Recursively update default config with custom values
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            update_dict(default_config, custom_config)
    
    return default_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train SMILES generation model')
    parser.add_argument('--config', type=str, help='Path to config file')
    return parser.parse_args()

def get_domain_ranges(meta_data):
    """Extract domain ranges from metadata"""
    ir_range = [
        min(meta_data["ir_spectra"]["dimensions"]),
        max(meta_data["ir_spectra"]["dimensions"])
    ]
    h_nmr_range = [
        min(meta_data["h_nmr_spectra"]["dimensions"]),
        max(meta_data["h_nmr_spectra"]["dimensions"])
    ]
    c_nmr_range = [
        min(meta_data["c_nmr_spectra"]["dimensions"]),
        max(meta_data["c_nmr_spectra"]["dimensions"])
    ]
    hsqc_h_range = [
        min(meta_data["hsqc_nmr_spectrum"]["dimensions"]["h"]),
        max(meta_data["hsqc_nmr_spectrum"]["dimensions"]["h"])
    ]
    hsqc_c_range = [
        min(meta_data["hsqc_nmr_spectrum"]["dimensions"]["c"]),
        max(meta_data["hsqc_nmr_spectrum"]["dimensions"]["c"])
    ]
    return ir_range, h_nmr_range, c_nmr_range, hsqc_h_range, hsqc_c_range

# Add this main function to contain the training code
def main():
    print("\n[Main] Starting training script...")
    args = parse_args()
    
    print("[Main] Loading configuration...")
    config = load_config(args.config)
    print(f"[Main] Loaded config with {len(config)} sections")

    print("\n[Main] Setting up model parameters...")
    max_seq_length = config['model']['max_seq_length']
    batch_size = config['training']['batch_size']
    embed_dim = config['model']['embed_dim']
    num_heads = config['model']['num_heads']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    resample_size = config['model']['resample_size']

    PAD_TOKEN_ID = tokenizer.pad_token_id
    BOS_TOKEN_ID = tokenizer.cls_token_id
    EOS_TOKEN_ID = tokenizer.sep_token_id

    print("\n[Main] Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"[Main] Found {torch.cuda.device_count()} CUDA devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  - {torch.cuda.get_device_name(i)}")
    else:
        print("[Main] No CUDA devices found, using CPU")
    print(f"[Main] Using device: {device}")

    print("\n[Main] Initializing model...")
    
    # Get domain ranges from dataset's metadata
    if config['data']['use_parquet']:
        meta_path = Path(config['data']['data_dir']) / "meta_data/meta_data_dict.json"
        with open(meta_path) as f:
            meta_data = json.load(f)
        domain_ranges = get_domain_ranges(meta_data)
    else:
        domain_ranges = None  # Use defaults for binary dataset
    
    model = MultiModalToSMILESModel(
        vocab_size=len(tokenizer),
        max_seq_length=max_seq_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        resample_size=resample_size,
        domain_ranges=domain_ranges,
        verbose=False
    ).to(device)
    print("[Main] Model initialized successfully")

    print("\n[Main] Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        tokenizer=tokenizer,
        config=config
    )

    print("\n[Main] Initializing wandb...")
    run_name = (
        f"{config['wandb']['base_run_name']}_"
        f"enc_d{embed_dim}_"
        f"enc_h{num_heads}_"
        f"dec_l{num_layers}_"
        f"bs{batch_size}_"
        f"lr{config['training']['learning_rate']}_warm{config['scheduler']['warmup_steps']}_"
        f"{datetime.now().strftime('%m%d_%H%M')}"
    )

    wandb.init(
        project=config['wandb']['project'],
        name=run_name,
        config=config
    )
    print("[Main] wandb initialized successfully")

    print("\n[Main] Setting up training components...")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_steps=config['scheduler']['warmup_steps'],
        T_0=config['scheduler']['T0'] * len(train_loader),
        T_mult=config['scheduler']['T_mult'],
        eta_min=config['training']['min_learning_rate']
    )

    print("\n[Main] Creating checkpoint directory...")
    save_dir = Path('checkpoints') / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Main] Checkpoint directory created at {save_dir}")

    print("\n[Main] Starting training loop...")
    NUM_EPOCHS = config['training']['num_epochs']
    validation_frequency = config['training']['validation_frequency']
    verbose = False
    
    # Add timing stats
    batch_times = []
    data_loading_times = []
    forward_times = []
    backward_times = []

    best_val_loss = float('inf')
    epoch_loss = 0
    num_batches = 0
    global_step = 0

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def evaluate_predictions(predictions, targets):
        """
        Evaluate model predictions vs. targets:
        - Exact match
        - Valid SMILES
        - Average Tanimoto (Morgan fingerprint)
        """
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
        
        exact_matches = 0
        valid_smiles = 0
        tanimoto_scores = []
        
        for pred, target in zip(predictions, targets):
            if pred == target:
                exact_matches += 1
            
            mol_pred = Chem.MolFromSmiles(pred)
            mol_target = Chem.MolFromSmiles(target)
            
            if mol_pred is not None:
                valid_smiles += 1
                if mol_target is not None:
                    fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2)
                    fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, 2)
                    tanimoto_scores.append(DataStructs.TanimotoSimilarity(fp_pred, fp_target))
        
        return {
            'exact_match': exact_matches / len(predictions),
            'valid_smiles': valid_smiles / len(predictions),
            'avg_tanimoto': sum(tanimoto_scores) / len(tanimoto_scores) if tanimoto_scores else 0.0
        }


    def greedy_decode(model, nmr_data, ir_data, c_nmr_data, max_len=128):
        """
        Simple greedy decoding for SMILES generation.
        """
        model.eval()
        with torch.no_grad():
            # Start token
            current_token = torch.tensor([[BOS_TOKEN_ID]], device=device)
            
            # Handle spectral data tuples properly
            # For single batch inference, we need to add batch dimension if not present
            if nmr_data is not None:
                if isinstance(nmr_data, tuple):
                    # If it's already a tuple of (data, domain), keep as is
                    pass
                else:
                    # Add batch dimension if needed
                    if nmr_data.dim() == 1:
                        nmr_data = nmr_data.unsqueeze(0)
                    # Create domain tensor if needed (should be provided by dataset)
                    raise ValueError("NMR data must be provided as (data, domain) tuple")
                    
            if ir_data is not None:
                if isinstance(ir_data, tuple):
                    pass
                else:
                    if ir_data.dim() == 1:
                        ir_data = ir_data.unsqueeze(0)
                    raise ValueError("IR data must be provided as (data, domain) tuple")
                    
            if c_nmr_data is not None:
                if isinstance(c_nmr_data, tuple):
                    pass
                else:
                    if c_nmr_data.dim() == 1:
                        c_nmr_data = c_nmr_data.unsqueeze(0)
                    raise ValueError("C-NMR data must be provided as (data, domain) tuple")
            
            # Encode
            memory = model.encoder(nmr_data, ir_data, c_nmr_data)
            
            generated_tokens = [BOS_TOKEN_ID]
            # Use the decoder's max_seq_length as the limit
            max_len = min(max_len, model.decoder.max_seq_length)
            
            for _ in range(max_len):
                logits = model.decoder(current_token, memory)
                next_token = logits[:, -1:].argmax(dim=-1)
                generated_tokens.append(next_token.item())
                current_token = torch.cat([current_token, next_token], dim=1)
                
                if next_token.item() == EOS_TOKEN_ID:
                    break
                
            return torch.tensor([generated_tokens])


    def validate(model, val_loader, criterion, tokenizer):
        model.eval()
        total_loss = 0
        num_batches = 0
        exact_matches = 0
        total_sequences = 0
        
        with torch.no_grad():
            for tgt_tokens, ir, h_nmr, c_nmr in val_loader:
                tgt_tokens = tgt_tokens.to(device)
                
                # Handle spectral data tuples
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
                
                T = tgt_tokens.shape[1]
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), 1)
                
                logits = model(h_nmr, ir, c_nmr, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))
                
                # Greedy decode for comparison
                pred_tokens = greedy_decode(model, h_nmr, ir, c_nmr)
                
                # Compare predictions
                for pred, target in zip(pred_tokens, tgt_tokens):
                    pred_smiles = tokenizer.decode(pred.tolist(), skip_special_tokens=True)
                    target_smiles = tokenizer.decode(target.tolist(), skip_special_tokens=True)
                    if pred_smiles == target_smiles:
                        exact_matches += 1
                
                total_loss += loss.item()
                num_batches += 1
                total_sequences += tgt_tokens.size(0)
        
        return {
            'val_loss': total_loss / num_batches,
            'val_exact_match': exact_matches / total_sequences
        }


    def log_validation_results(val_metrics, global_step):
        wandb.log({
            "val_loss": val_metrics['val_loss'],
            "val_exact_match": val_metrics['val_exact_match']
        }, step=global_step)


    def evaluate_on_test(model, test_loader, tokenizer, device):
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for tgt_tokens, ir, h_nmr, c_nmr in test_loader:
                # Move target tokens to device
                tgt_tokens = tgt_tokens.to(device)
                
                # Handle spectral data tuples properly
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
                
                # Use greedy decode with proper tensor handling
                pred_tokens = greedy_decode(model, h_nmr, ir, c_nmr)
                
                # Decode predictions and targets
                pred_smiles = tokenizer.decode(pred_tokens[0].tolist(), skip_special_tokens=True)
                target_smiles = tokenizer.decode(tgt_tokens[0].tolist(), skip_special_tokens=True)
                
                all_predictions.append(pred_smiles)
                all_targets.append(target_smiles)

        return all_predictions, all_targets


    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    model.train()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            if verbose:
                data_load_start = time.time()
            
            # Unpack the batch data correctly
            tgt_tokens, ir, h_nmr, c_nmr = batch
            
            # Get the batch data
            tgt_tokens = tgt_tokens.to(device)

            # Handle spectral data tuples
            if ir is not None:
                if isinstance(ir, tuple):
                    # For 1D spectra: (data, domain)
                    ir = (ir[0].to(device), ir[1].to(device))
                else:
                    ir = ir.to(device)

            if h_nmr is not None:
                print('h_nmr',h_nmr)
                if isinstance(h_nmr, tuple):
                    h_nmr = (h_nmr[0].to(device), h_nmr[1].to(device))
                else:
                    h_nmr = h_nmr.to(device)

            if c_nmr is not None:
                print('c_nmr',c_nmr)

                if isinstance(c_nmr, tuple):
                    c_nmr = (c_nmr[0].to(device), c_nmr[1].to(device))
                else:
                    c_nmr = c_nmr.to(device)

            if verbose:
                data_load_time = time.time() - data_load_start
                data_loading_times.append(data_load_time)
                forward_start = time.time()

            # Forward pass
            T = tgt_tokens.shape[1]
            # Print shapes of spectral tensors
            print("\nSpectral tensor shapes:")
            if h_nmr is not None:
                if isinstance(h_nmr, tuple):
                    print(f"H-NMR data: {h_nmr[0].shape}")
                else:
                    print(f"H-NMR: {h_nmr.shape}")
            if ir is not None:
                if isinstance(ir, tuple):
                    print(f"IR data: {ir[0].shape}")
                else:
                    print(f"IR: {ir.shape}")
            if c_nmr is not None:
                if isinstance(c_nmr, tuple):
                    print(f"C-NMR data: {c_nmr[0].shape}")
                else:
                    print(f"C-NMR: {c_nmr.shape}")
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), 1)
            logits = model(h_nmr, ir, c_nmr, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))

            if verbose:
                forward_time = time.time() - forward_start
                forward_times.append(forward_time)
                backward_start = time.time()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose:
                backward_time = time.time() - backward_start
                backward_times.append(backward_time)
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)

                if batch_idx % 10 == 0:  # Print timing stats every 10 batches
                    print(f"\n[Timing Stats] Batch {batch_idx}")
                    print(f"  Data loading: {np.mean(data_loading_times[-10:]):.3f}s")
                    print(f"  Forward pass: {np.mean(forward_times[-10:]):.3f}s")
                    print(f"  Backward pass: {np.mean(backward_times[-10:]):.3f}s")
                    print(f"  Total batch time: {np.mean(batch_times[-10:]):.3f}s")

            current_lr = scheduler.get_lr()[0]
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Update progress bar with timing info if verbose
            if verbose:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}",
                    'batch_time': f"{batch_time:.3f}s",
                    'data_time': f"{data_load_time:.3f}s"
                })
            else:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}"
                })

            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": current_lr,
                "epoch": epoch + 1,
                "global_step": global_step,
            }, step=global_step)

            if batch_idx % config['training']['logging_frequency'] == 0:
                model.eval()
                with torch.no_grad():
                    # Take first item but maintain tuple structure
                    h_nmr_batch = (h_nmr[0][0:1], h_nmr[1]) if h_nmr is not None else None
                    ir_batch = (ir[0][0:1], ir[1]) if ir is not None else None
                    c_nmr_batch = (c_nmr[0][0:1], c_nmr[1]) if c_nmr is not None else None
                    
                    pred_tokens = greedy_decode(model,
                                                  h_nmr_batch,
                                                  ir_batch,
                                                  c_nmr_batch)
                    pred_smiles = tokenizer.decode(pred_tokens[0].tolist(), skip_special_tokens=True)
                    target_smiles = tokenizer.decode(tgt_tokens[0].tolist(), skip_special_tokens=True)
                    
                    wandb.log({
                        "example_prediction": wandb.Table(
                            columns=["Target SMILES", "Predicted SMILES"],
                            data=[[target_smiles, pred_smiles]]
                        )
                    }, step=global_step)
                model.train()

            # Periodic validation
            if global_step % validation_frequency == 0:
                val_metrics = validate(model, val_loader, criterion, tokenizer)
                log_validation_results(val_metrics, global_step)
                pbar.set_postfix({
                    'train_loss': f"{loss.item():.4f}",
                    'val_loss': f"{val_metrics['val_loss']:.4f}",
                    'val_exact': f"{val_metrics['val_exact_match']:.2%}"
                })
                
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': val_metrics,
                    }, save_dir / 'best_model.pt')
                model.train()

    # -------------------------------------------------------------------------
    # Final Evaluation on Test Set
    # -------------------------------------------------------------------------
    print("\nRunning final evaluation on test set...")
    all_predictions, all_targets = evaluate_on_test(model, test_loader, tokenizer, device)

    results = evaluate_predictions(all_predictions, all_targets)
    wandb.log({
        "test_exact_match": results['exact_match'],
        "test_valid_smiles": results['valid_smiles'],
        "test_avg_tanimoto": results['avg_tanimoto']
    })

    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nTest Results:")
    print(f"Exact Match: {results['exact_match']:.2%}")
    print(f"Valid SMILES: {results['valid_smiles']:.2%}")
    print(f"Avg Tanimoto: {results['avg_tanimoto']:.3f}")

    wandb.finish()

# Add this guard at the bottom of the file
if __name__ == '__main__':
    # Optional: Add freeze_support() if you plan to create executables
    # from multiprocessing import freeze_support
    # freeze_support()
    
    main()
