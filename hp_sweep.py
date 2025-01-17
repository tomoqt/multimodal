# =======================
# File: train_autoregressive.py
# =======================
"""
Script to perform hyperparameter sweeps for the MultiModalToSMILESModel using SELFIES tokenization.
Key Steps:
1) Loads spectral + SMILES data from .bin and .npy index
2) Tokenizes SMILES into SELFIES
3) Performs quick LR sweep using Ezmup
4) Trains final model with best LR
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
from rdkit import RDLogger
from rdkit.Chem import rdFMCS
from pathlib import Path
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse
import time
from pprint import pprint
from copy import deepcopy
from logging_utils import evaluate_predictions, aggregate_metrics, log_results
import selfies as sf

# Import our custom tokenizer and utils
from models.selfies_tokenizer import SelfiesTokenizer
from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_utils import process_smiles_to_selfies
from utils.ezmup import Ezmup

# Disable RDKit logging
RDLogger.DisableLog("rdApp.*")

current_dir = os.path.dirname(os.path.realpath(__file__))
vocab_path = os.path.join(current_dir, 'selfies_vocab.txt')

# Debug print to check paths
print("\n[Debug] Looking for SELFIES vocabulary:")
print(f"Current directory: {current_dir}")
print(f"Trying vocab path: {vocab_path}")
if os.path.exists(vocab_path):
    print(f"Found vocabulary file!")
    # Print first few lines to verify content
    with open(vocab_path) as f:
        print("First 5 lines of vocabulary:")
        for i, line in enumerate(f):
            if i < 5:
                print(f"  {line.strip()}")
else:
    print(f"WARNING: Vocabulary file not found at {vocab_path}")

tokenizer = SelfiesTokenizer(vocab_file=vocab_path)

# -------------------------------------------------------------------------
# Linear Warmup + Cosine/Constant LR Scheduler
# -------------------------------------------------------------------------
class LinearWarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by either constant LR or cosine decay.
    Linearly increases learning rate from 0 to max_lr over `warmup_steps`,
    then either maintains constant LR or uses cosine decay from max_lr to min_lr.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, decay_type='cosine', min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_type = decay_type
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / float(max(1, self.warmup_steps))
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            if self.decay_type == 'constant':
                # Constant learning rate after warmup
                return self.base_lrs
            else:  # cosine decay
                # Cosine decay
                progress = (self.last_epoch - self.warmup_steps) / float(
                    max(1, self.total_steps - self.warmup_steps)
                )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [
                    self.min_lr + (base_lr - self.min_lr) * cosine_decay 
            for base_lr in self.base_lrs
        ]


# -------------------------------------------------------------------------
# Dataset / DataLoader for Memory-Mapped Binary Files
# -------------------------------------------------------------------------
class SpectralSmilesDataset(Dataset):
    """
    A PyTorch Dataset that reads from tokenized text files and numpy arrays:
      - src-{split}.txt  (source sequences with NMR data)
      - tgt-{split}.txt  (target SMILES sequences) 
      - ir-{split}.npy   (IR spectra data)
    """
    def __init__(
        self, 
        data_dir, 
        smiles_tokenizer, 
        spectral_tokenizer, 
        split='train', 
        max_smiles_len=512,  # Separate length limit for SMILES
        max_nmr_len=128      # Separate length limit for NMR
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.smiles_tokenizer = smiles_tokenizer
        self.spectral_tokenizer = spectral_tokenizer
        self.max_smiles_len = max_smiles_len
        self.max_nmr_len = max_nmr_len
        self.split = split

        # Load source (NMR) and target sequences
        with open(self.data_dir / f"src-{split}.txt") as f:
            self.sources = [line.strip() for line in f]
        with open(self.data_dir / f"tgt-{split}.txt") as f:
            self.targets = [line.strip() for line in f]

        # Load IR data using numpy.memmap instead of pickle
        ir_path = self.data_dir / f"ir-{split}.npy"
        self.ir_data = None
        if ir_path.exists():
            try:
                # Use memmap to load the IR data
                self.ir_data = np.memmap(
                    ir_path,
                    dtype='float32',
                    mode='r',
                    shape=None  # Let numpy figure out the shape
                )
                # Get the actual shape from the memmap
                array_shape = self.ir_data.shape
                # Reshape if needed (should be 2D: [num_samples, features])
                if len(array_shape) == 1:
                    # Calculate number of samples based on total size and feature dimension
                    num_samples = len(self.sources)
                    feature_dim = array_shape[0] // num_samples
                    self.ir_data = self.ir_data.reshape(num_samples, feature_dim)
                
                print(f"[Dataset] Loaded IR data with shape: {self.ir_data.shape}")
            except Exception as e:
                print(f"[Warning] Failed to load IR data: {e}")
                self.ir_data = None

        print(f"[Dataset] SpectralSmilesDataset initialized for {split}:")
        print(f"          Found {len(self.sources)} samples")

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        """
        Returns:
          (target_tokens, (ir_data, None), nmr_tokens, None)
        """
        # Get target sequence and convert to SELFIES using standardized function
        target_seq = self.targets[idx]
        selfies = process_smiles_to_selfies(target_seq, idx)
        if selfies is None:
            # Fallback to a simple token if conversion fails
            selfies = "[C]"
        
        target_tokens = self.smiles_tokenizer.encode(
            selfies,
            add_special_tokens=True,
            max_length=self.max_smiles_len,
            truncation=True
        )
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        # Get source sequence (NMR data) - use spectral tokenizer
        source_seq = self.sources[idx]
        # Split into tokens and convert to IDs using the spectral vocabulary
        nmr_tokens = source_seq.split()
        nmr_token_ids = [self.spectral_tokenizer.get(token, self.spectral_tokenizer["<UNK>"]) 
                        for token in nmr_tokens]
        if len(nmr_token_ids) > self.max_nmr_len:
            nmr_token_ids = nmr_token_ids[:self.max_nmr_len]
        nmr_tokens = torch.tensor(nmr_token_ids, dtype=torch.long)

        # Get IR data if available - modify to handle memmap
        ir_data = None
        if self.ir_data is not None:
            # Copy the data from memmap to a regular tensor
            ir_data = torch.tensor(self.ir_data[idx].copy(), dtype=torch.float32)

        return (
            target_tokens,
            (ir_data, None),
            nmr_tokens,
            None
        )

    # Add cleanup method to properly close memmap file
    def __del__(self):
        if hasattr(self, 'ir_data') and self.ir_data is not None:
            del self.ir_data


# -------------------------------------------------------------------------
# Collate Function
# -------------------------------------------------------------------------
def collate_fn(batch):
    """
    Custom collate function that handles:
    - Padding target tokens (SMILES)
    - Padding NMR tokens
    - Stacking IR data
    """
    target_tokens, ir_tuples, nmr_tokens, _ = zip(*batch)

    # Pad target tokens (SMILES)
    max_target_len = max(len(seq) for seq in target_tokens)
    padded_target_tokens = []
    for seq in target_tokens:
        pad_amount = max_target_len - len(seq)
        if pad_amount > 0:
            pad_tensor = torch.full((pad_amount,), tokenizer.pad_token_id, dtype=torch.long)
            seq = torch.cat([seq, pad_tensor], dim=0)
        padded_target_tokens.append(seq)
    target_batch = torch.stack(padded_target_tokens, dim=0)

    # Pad NMR tokens
    max_nmr_len = max(len(seq) for seq in nmr_tokens)
    padded_nmr_tokens = []
    for seq in nmr_tokens:
        pad_amount = max_nmr_len - len(seq)
        if pad_amount > 0:
            pad_tensor = torch.full((pad_amount,), spectral_tokenizer["<PAD>"], dtype=torch.long)
            seq = torch.cat([seq, pad_tensor], dim=0)
        padded_nmr_tokens.append(seq)
    nmr_batch = torch.stack(padded_nmr_tokens, dim=0)

    # Stack IR data if available
    ir_batch = None
    if ir_tuples[0] is not None:
        # Extract just the IR tensors from the tuples (first element)
        ir_tensors = [t[0] for t in ir_tuples if t[0] is not None]
        if ir_tensors:
            ir_batch = torch.stack(ir_tensors, dim=0)

    return target_batch, ir_batch, nmr_batch, None

def create_data_loaders(tokenizer, config):
    print("\n[DataLoader] Creating data loaders...")
    if config['data']['use_parquet']:
        dataset = ParquetSpectralDataset(
            data_dir=config['data']['data_dir'],
            tokenizer=tokenizer,
            max_len=config['model']['max_seq_length']
        )
    else:
        dataset = SpectralSmilesDataset(
            data_dir=config['data']['binary_dir'],
            smiles_tokenizer=tokenizer,
            spectral_tokenizer=tokenizer,
            split='train',
            max_smiles_len=config['model']['max_seq_length'],
            max_nmr_len=config['model']['max_seq_length']
        )
    
    print(f"[DataLoader] Total dataset size: {len(dataset)}")
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
# Config + Argparse
# -------------------------------------------------------------------------
def load_config(config_path=None):
    """Load config from yaml file, falling back to defaults if not specified"""
    default_config = {
        'model': {
            'max_seq_length': 512,      # Max SMILES sequence length
            'max_nmr_length': 128,      # Max NMR sequence length
            'max_memory_length': 128,   # Max memory/IR sequence length
            'embed_dim': 768,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'resample_size': 1000,
            'use_concat': True
        },
        'training': {
            'batch_size': 32,
            'test_batch_size': 1,
            'num_epochs': 1,
            'learning_rate': 1.0e-4,
            'min_learning_rate': 1.0e-6,
            'validation_frequency': 500,
            'logging_frequency': 100,
            'save_frequency': 1000,
            'generate_during_training': False,
            'save_local': False
        },
        'scheduler': {
            'type': 'constant',  # or 'cosine'
            'warmup_steps': 100
        },
        'data': {
            'tokenized_dir': "tokenized_baseline/data",  # Path to tokenized data
            'num_workers': 0
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


# -------------------------------------------------------------------------
# Main Training Script
# -------------------------------------------------------------------------
def main():
    print("\n[Main] Starting training script...")
    args = parse_args()
    
    print("[Main] Loading configuration...")
    config = load_config(args.config)
    print(f"[Main] Loaded config with {len(config)} sections")

    # -------------------------------------------------------
    # (Optional) Demonstration of prime-based "width" for MuP
    # We'll set a default so we can still show usage of Ezmup,
    # but this time we'll *sweep learning rates*, not widths.
    # -------------------------------------------------------
    config['model']['embed_dim'] = 47 * 8   # 47 is our "width_basis"
    config['model']['num_heads'] = 47       # for demonstration
    # If you had a feedforward dimension, you could do 47 * 32, etc.

    # Prepare model hyperparameters
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

    print("\n[Main] Initializing model (with prime-based dimension for MuP)...")
    if config['data']['use_parquet']:
        meta_path = Path(config['data']['data_dir']) / "meta_data/meta_data_dict.json"
        with open(meta_path) as f:
            meta_data = json.load(f)
        domain_ranges = get_domain_ranges(meta_data)
    else:
        domain_ranges = None

    # Instantiate your model
    model = MultiModalToSMILESModel(
        vocab_size=len(tokenizer),
        max_seq_length=max_seq_length,
        embed_dim=embed_dim,    
        num_heads=num_heads,    
        num_layers=num_layers,  
        dropout=dropout,
        resample_size=resample_size,
        domain_ranges=domain_ranges,
        verbose=False,
        use_concat=config['model']['use_concat']
    ).to(device)
    print("[Main] Model initialized successfully")

    # -------------------------------------------------------
    # Wrap model with Ezmup (though we'll keep width *fixed*).
    # We'll do the LR sweep instead of a width sweep.
    # -------------------------------------------------------
    print("\n[Main] Wrapping model with Ezmup (width_basis=47)...")
    mup_engine = Ezmup(width_basis=47, model=model)

    # For demonstration, let's fix the width to something small, e.g. 32:
    # (so we don't have to do full-scale training).
    print("[Main] Setting initial model width to 32 for MuP demonstration...")
    mup_engine.change_width_as(32)

    print("\n[Main] Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(tokenizer=tokenizer, config=config)

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

    # ---------------------------------------------------------------------
    # Function: quick_train_eval_lr
    #   We'll do a very short training loop for each candidate LR, then
    #   evaluate on the validation set to find the best LR.
    # ---------------------------------------------------------------------
    def quick_train_eval_lr(candidate_lr, n_steps=200):
        """
        Sets the optimizer to candidate_lr, does a short training loop,
        returns the final validation loss. 
        """
        # Re-init the optimizer each time with the new LR
        optimizer = mup_engine.get_optimizer(lr=candidate_lr)
        scheduler = LinearWarmupCosineDecay(
            optimizer,
            warmup_steps=config['scheduler']['warmup_steps'],
            total_steps=config['scheduler']['T0'] * len(train_loader),
            decay_type=config['scheduler']['T_mult'],
            min_lr=config['training']['min_learning_rate']
        )
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

        model.train()
        step_count = 0

        # Short training loop
        for batch in train_loader:
            tgt_tokens, ir, h_nmr, c_nmr = batch
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

            T = tgt_tokens.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), 1)
            logits = model(h_nmr, ir, c_nmr, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step_count += 1
            if step_count >= n_steps:
                break

        # Evaluate on validation set
        model.eval()
        total_loss = 0.0
        total_batches = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                tgt_tokens, ir, h_nmr, c_nmr = batch
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

                T = tgt_tokens.size(1)
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), 1)
                
                logits = model(
                    h_nmr, ir, c_nmr,
                    target_seq=tgt_tokens[:, :-1],
                    target_mask=mask[:-1, :-1]
                )
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))
                
                # Get predictions and immediately move to CPU
                pred_tokens = logits.argmax(dim=-1).cpu().tolist()
                tgt_tokens = tgt_tokens[:, 1:].cpu().tolist()
                
                # Decode predictions, including SEP token as sequence end marker
                for pred_seq in pred_tokens:
                    # Find the first occurrence of SEP token if it exists
                    try:
                        sep_idx = pred_seq.index(tokenizer.sep_token_id)
                        # Include SEP token in sequence but don't decode it
                        pred_seq = pred_seq[:sep_idx]  # Don't include SEP in final string
                    except ValueError:
                        # No SEP token found, use full sequence
                        pass
                        
                    # Decode the sequence (SEP was used to mark end but isn't included)
                    decoded = tokenizer.decode(pred_seq).strip()
                    predictions.append(decoded)

                # Decode targets similarly
                for tgt_seq in tgt_tokens:
                    try:
                        sep_idx = tgt_seq.index(tokenizer.sep_token_id)
                        tgt_seq = tgt_seq[:sep_idx]  # Don't include SEP in final string
                    except ValueError:
                        pass
                    decoded = tokenizer.decode(tgt_seq).strip()
                    targets.append(decoded)
                
                # Clear GPU tensors we don't need anymore
                del logits, mask
                if ir is not None:
                    del ir
                if h_nmr is not None:
                    del h_nmr
                if c_nmr is not None:
                    del c_nmr
                
                total_loss += loss.item()
                total_batches += 1

                # Clear some memory
                torch.cuda.empty_cache()
        
        val_loss = total_loss / max(total_batches, 1)
        
        # Calculate molecular metrics using logging_utils
        detailed_results = evaluate_predictions(predictions, targets)
        metrics = aggregate_metrics(detailed_results)
        
        # Store all examples, not just the first 10
        combined_metrics = {
            'val_loss': val_loss,
            'valid_smiles_rate': metrics['valid_smiles'],
            'exact_match_rate': metrics['exact_match'],
            'tanimoto_similarity': metrics['avg_tanimoto'],
            'mcs_ratio': metrics['avg_#mcs/#target'],
            'ecfp6_iou': metrics['avg_ecfp6_iou'],
            'predictions': predictions[:10],  # Store first 10 predictions
            'targets': targets[:10],         # Store first 10 targets
            'num_samples': len(predictions)
        }

        print(f"[quick_train_eval_lr] LR={candidate_lr}, val_loss={val_loss:.4f}")
        print(f"                      valid_smiles={metrics['valid_smiles']:.2%}")
        print(f"                      exact_match={metrics['exact_match']:.2%}")
        print(f"                      avg_tanimoto={metrics['avg_tanimoto']:.4f}")
        
        return combined_metrics

    # ---------------------------------------------------------------------
    # Step: Sweep over candidate LRs
    # ---------------------------------------------------------------------
    lrs_to_try = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  # an example list
    best_val_loss = float('inf')
    best_lr = None

    print("\n[Main] Starting mini HP sweep over learning rates:", lrs_to_try)
    for lr_candidate in lrs_to_try:
        val_metrics = quick_train_eval_lr(lr_candidate, n_steps=100)
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_lr = lr_candidate
            
            # Log the best results so far to wandb
            wandb.log({
                "sweep_val_loss": val_metrics['val_loss'],
                "sweep_valid_smiles": val_metrics['valid_smiles_rate'],
                "sweep_exact_match": val_metrics['exact_match_rate'],
                "sweep_tanimoto": val_metrics['tanimoto_similarity'],
                "sweep_mcs_ratio": val_metrics['mcs_ratio'],
                "sweep_ecfp6_iou": val_metrics['ecfp6_iou'],
                "sweep_lr": lr_candidate
            })
            
    print(f"[Main] Best LR found: {best_lr} (val_loss={best_val_loss:.4f})")

    # Initialize wandb table for examples
    columns = ["step", "prediction", "target", "exact_match", "tanimoto", "mcs_ratio", "ecfp6_iou"]
    examples_table = wandb.Table(columns=columns)

    # Optionally, we can now upscale the width if desired:
    print(f"\n[Main] For final training, let's scale width up to 1024 (example)...")
    mup_engine.change_width_as(1024)

    # Build the final optimizer with best LR
    optimizer = mup_engine.get_optimizer(lr=best_lr)
    scheduler = LinearWarmupCosineDecay(
        optimizer,
        warmup_steps=config['scheduler']['warmup_steps'],
        total_steps=config['scheduler']['T0'] * len(train_loader),
        decay_type=config['scheduler']['T_mult'],
        min_lr=config['training']['min_learning_rate']
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    print("\n[Main] Creating checkpoint directory...")
    save_dir = Path('checkpoints') / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Main] Checkpoint directory created at {save_dir}")

    print("\n[Main] Final Training (width=1024, LR=", best_lr, ")")
    global_step = 0
    best_val_loss = float('inf')
    NUM_EPOCHS = config['training']['num_epochs']
    validation_frequency = config['training']['validation_frequency']
    logging_frequency = config['training']['logging_frequency']

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, total=len(train_loader), desc="Training", dynamic_ncols=True)
        for batch in pbar:
            tgt_tokens, ir, h_nmr, c_nmr = batch
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

            T = tgt_tokens.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), 1)
            logits = model(h_nmr, ir, c_nmr, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % logging_frequency == 0:
                current_lr = scheduler.get_lr()[0]  # Get current learning rate
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "progress": global_step / total_training_steps  # Add progress tracking
                }, step=global_step)

            # Periodic validation
            if global_step % validation_frequency == 0:
                print(f"\nRunning validation at step {global_step}...")
                val_metrics = quick_train_eval_lr(best_lr, n_steps=0)  # n_steps=0 means just evaluate
                
                # Create a new table for each validation step
                examples_table = wandb.Table(columns=columns)
                
                # Log results - sample 10 random examples for logging
                if val_metrics['predictions']:
                    # Randomly sample 10 indices
                    num_examples = len(val_metrics['predictions'])
                    sample_indices = np.random.choice(
                        num_examples, 
                        min(10, num_examples), 
                        replace=False
                    )
                    
                    for idx in sample_indices:
                        pred = val_metrics['predictions'][idx]
                        tgt = val_metrics['targets'][idx]
                        # Calculate metrics for this pair
                        pair_results = evaluate_predictions([pred], [tgt])[0]
                        examples_table.add_data(
                            global_step,
                            pred,
                            tgt,
                            pair_results['exact_match'],
                            pair_results['tanimoto'],
                            pair_results['#mcs/#target'],
                            pair_results['ecfp6_iou']
                        )
                
                # Log metrics
                wandb.log({
                    "val_loss": val_metrics['val_loss'],
                    "val_valid_smiles": val_metrics['valid_smiles_rate'],
                    "val_exact_matches": val_metrics['exact_match_rate'],
                    "val_tanimoto": val_metrics['tanimoto_similarity'],
                    "val_mcs_ratio": val_metrics['mcs_ratio'],
                    "val_ecfp6_iou": val_metrics['ecfp6_iou'],
                    "val_examples": examples_table,  # Log new table each time
                    "global_step": global_step
                }, step=global_step)
                
                print(f"[Val] Loss: {val_metrics['val_loss']:.4f}")

                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    print(f"New best validation loss: {best_val_loss:.4f}")
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_metrics['val_loss'],
                    }
                    torch.save(checkpoint, save_dir / 'best_model.pt')
                    print(f"Saved checkpoint at {save_dir}/best_model.pt")

                    # Save to W&B as an artifact
                    artifact = wandb.Artifact(
                        "model", 
                        type="model",
                        description=f"Checkpoint at step {global_step}, val_loss {best_val_loss:.4f}"
                    )
                    wandb.log_artifact(artifact, aliases=["latest", f"step_{global_step}"])
                    print("[Main] Checkpoint artifact saved to W&B.")

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} completed | Average Loss: {avg_epoch_loss:.4f}")
        wandb.log({"train_loss": avg_epoch_loss, "epoch": epoch+1})

    # Final test set evaluation
    print("\n[Main] Evaluating on test set...")
    final_test_metrics = quick_train_eval_lr(best_lr, n_steps=0)  # n_steps=0 means just evaluate
    wandb.log({
        "test_loss": final_test_metrics['val_loss'],
        "test_valid_smiles": final_test_metrics['valid_smiles_rate'],
        "test_exact_matches": final_test_metrics['exact_match_rate'],
        "test_tanimoto": final_test_metrics['tanimoto_similarity'],
        "test_mcs_ratio": final_test_metrics['mcs_ratio'],
        "test_ecfp6_iou": final_test_metrics['ecfp6_iou']
    }, step=global_step)
    print(f"[Test] Loss: {final_test_metrics['val_loss']:.4f}")

    # Save final model
    final_model_path = save_dir / 'final_model.pt'
    torch.save({
        'epoch': NUM_EPOCHS,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_metrics': final_test_metrics
    }, final_model_path)
    print(f"\n[Main] Final model saved to {final_model_path}")

    wandb.finish()


if __name__ == '__main__':
    main()
