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
5) Demonstrates using ezmup to do a small HP sweep on *learning rate*
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

# ----------------------------
#  Import Ezmup from ezmup
# ----------------------------
from utils.ezmup import Ezmup

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
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        # Cosine annealing with warm restarts
        if not self.completed_warmup:
            self.completed_warmup = True
            self.T_cur = 0
        
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
# Dataset / DataLoader for Memory-Mapped Binary Files
# -------------------------------------------------------------------------
class SpectralSmilesDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len=128):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len

        index_path = self.data_dir / "spectra_index.npy"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found at {index_path}")
        self.index = np.load(index_path)

        spectra_bin_path = self.data_dir / "spectra_data.bin"
        if not spectra_bin_path.exists():
            raise FileNotFoundError(f"Spectra data file not found at {spectra_bin_path}")
        self.spectra_mmap = np.memmap(spectra_bin_path, dtype=np.float32, mode='r')

        smiles_bin_path = self.data_dir / "smiles_data.bin"
        if not smiles_bin_path.exists():
            raise FileNotFoundError(f"SMILES data file not found at {smiles_bin_path}")
        self.smiles_mmap = np.memmap(smiles_bin_path, dtype=np.uint8, mode='r')

        print(f"[Dataset] MemoryMappedSpectralDataset initialized:")
        print(f"          Index shape = {self.index.shape}")
        print(f"          Found {len(self.index)} samples total.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        (
            ir_data_off, ir_data_len,
            ir_dom_off,  ir_dom_len,
            hnm_data_off, hnm_data_len,
            hnm_dom_off,  hnm_dom_len,
            cnm_data_off, cnm_data_len,
            cnm_dom_off,  cnm_dom_len,
            smiles_off,   smiles_len
        ) = self.index[idx]

        # IR
        if ir_data_off == -1 or ir_data_len == 0:
            ir_tuple = None
        else:
            ir_data = self.spectra_mmap[ir_data_off : ir_data_off + ir_data_len]
            if ir_dom_off == -1 or ir_dom_len == 0:
                ir_dom = None
            else:
                ir_dom = self.spectra_mmap[ir_dom_off : ir_dom_off + ir_dom_len]
            ir_data_t = torch.from_numpy(ir_data.copy()) if ir_data_len > 0 else None
            ir_dom_t  = torch.from_numpy(ir_dom.copy())  if ir_dom is not None else None
            ir_tuple  = (ir_data_t, ir_dom_t)

        # H-NMR
        if hnm_data_off == -1 or hnm_data_len == 0:
            h_nmr_tuple = None
        else:
            h_nmr_data = self.spectra_mmap[hnm_data_off : hnm_data_off + hnm_data_len]
            if hnm_dom_off == -1 or hnm_dom_len == 0:
                h_nmr_dom = None
            else:
                h_nmr_dom = self.spectra_mmap[hnm_dom_off : hnm_dom_off + hnm_dom_len]
            h_nmr_tuple = (
                torch.from_numpy(h_nmr_data.copy()),
                torch.from_numpy(h_nmr_dom.copy()) if h_nmr_dom is not None else None
            )

        # C-NMR
        if cnm_data_off == -1 or cnm_data_len == 0:
            c_nmr_tuple = None
        else:
            c_nmr_data = self.spectra_mmap[cnm_data_off : cnm_data_off + cnm_data_len]
            if cnm_dom_off == -1 or cnm_dom_len == 0:
                c_nmr_dom = None
            else:
                c_nmr_dom = self.spectra_mmap[cnm_dom_off : cnm_dom_off + cnm_dom_len]
            c_nmr_tuple = (
                torch.from_numpy(c_nmr_data.copy()),
                torch.from_numpy(c_nmr_dom.copy()) if c_nmr_dom is not None else None
            )

        # SMILES
        if smiles_off == -1 or smiles_len == 0:
            smiles_str = ""
        else:
            smiles_bytes = self.smiles_mmap[smiles_off : smiles_off + smiles_len]
            smiles_str = smiles_bytes.tobytes().decode('utf-8')

        # Tokenize
        tokens = self.tokenizer.encode(
            smiles_str,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens, ir_tuple, h_nmr_tuple, c_nmr_tuple


# -------------------------------------------------------------------------
# Dataset / DataLoader for Parquet Files
# -------------------------------------------------------------------------
class ParquetSpectralDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len=128):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        meta_path = self.data_dir / "meta_data/meta_data_dict.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        with open(meta_path) as f:
            self.meta_data = json.load(f)

        self.ir_domain = torch.tensor(self.meta_data["ir_spectra"]["dimensions"], dtype=torch.float32)
        self.h_nmr_domain = torch.tensor(self.meta_data["h_nmr_spectra"]["dimensions"], dtype=torch.float32)
        self.c_nmr_domain = torch.tensor(self.meta_data["c_nmr_spectra"]["dimensions"], dtype=torch.float32)

        print("[Dataset] Looking for parquet files...")
        self.parquet_files = sorted(self.data_dir.glob("*.parquet"))
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        print(f"[Dataset] Found {len(self.parquet_files)} parquet files")

        print("[Dataset] Loading parquet files...")
        dfs = []
        for file in tqdm(self.parquet_files, desc="Loading parquet files"):
            df = pd.read_parquet(
                file,
                columns=['smiles', 'ir_spectra', 'h_nmr_spectra', 'c_nmr_spectra'],
                engine='pyarrow'
            )
            dfs.append(df)
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"[Dataset] Loaded {len(self.data)} total rows")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['smiles']
        tokens = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )

        def to_tensor(x, spectrum_type):
            if x is None:
                return None
            try:
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

        ir_spectra = to_tensor(row['ir_spectra'], 'ir')
        h_nmr_spectra = to_tensor(row['h_nmr_spectra'], 'h_nmr')
        c_nmr_spectra = to_tensor(row['c_nmr_spectra'], 'c_nmr')

        return tokens, ir_spectra, h_nmr_spectra, c_nmr_spectra


# -------------------------------------------------------------------------
# Collate function
# -------------------------------------------------------------------------
def collate_fn(batch):
    all_tokens, all_ir, all_h_nmr, all_c_nmr = zip(*batch)
    
    def maybe_stack_with_domain(items):
        if items[0] is not None:
            data = torch.stack([item[0] for item in items], dim=0)
            domain = items[0][1]
            return (data, domain)
        return None

    ir_batch = maybe_stack_with_domain(all_ir) if all_ir[0] is not None else None
    h_nmr_batch = maybe_stack_with_domain(all_h_nmr) if all_h_nmr[0] is not None else None
    c_nmr_batch = maybe_stack_with_domain(all_c_nmr) if all_c_nmr[0] is not None else None

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
            tokenizer=tokenizer,
            max_len=config['model']['max_seq_length']
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
    default_config = {
        'model': {
            'max_seq_length': 128,
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
            'learning_rate': 1.0e-4,   # default
            'min_learning_rate': 1.0e-6,
            'validation_frequency': 500,
            'logging_frequency': 100,
            'save_frequency': 1000,
            'generate_during_training': False
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
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_steps=config['scheduler']['warmup_steps'],
            T_0=config['scheduler']['T0'] * len(train_loader),
            T_mult=config['scheduler']['T_mult'],
            eta_min=config['training']['min_learning_rate']
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
        total_loss = 0
        count = 0
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

                T = tgt_tokens.shape[1]
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), 1)
                v_logits = model(h_nmr, ir, c_nmr, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])
                v_loss = criterion(v_logits.reshape(-1, v_logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))
                
                total_loss += v_loss.item()
                count += 1

        avg_val_loss = total_loss / max(count, 1)
        print(f"[quick_train_eval_lr] LR={candidate_lr}, val_loss={avg_val_loss:.4f}")
        return avg_val_loss

    # ---------------------------------------------------------------------
    # Step: Sweep over candidate LRs
    # ---------------------------------------------------------------------
    lrs_to_try = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  # an example list
    best_val_loss = float('inf')
    best_lr = None

    print("\n[Main] Starting mini HP sweep over learning rates:", lrs_to_try)
    for lr_candidate in lrs_to_try:
        val_loss = quick_train_eval_lr(lr_candidate, n_steps=100)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr = lr_candidate
    print(f"[Main] Best LR found: {best_lr} (val_loss={best_val_loss:.4f})")

    # Optionally, we can now upscale the width if desired:
    print(f"\n[Main] For final training, let's scale width up to 1024 (example)...")
    mup_engine.change_width_as(1024)

    # Build the final optimizer with best LR
    optimizer = mup_engine.get_optimizer(lr=best_lr)
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_steps=config['scheduler']['warmup_steps'],
        T_0=config['scheduler']['T0'] * len(train_loader),
        T_mult=config['scheduler']['T_mult'],
        eta_min=config['training']['min_learning_rate']
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

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        epoch_loss = 0
        num_batches = 0

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

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % validation_frequency == 0:
                # Quick val
                model.eval()
                total_val_loss = 0
                val_count = 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        vtgt_tokens, vir, vh_nmr, vc_nmr = vbatch
                        vtgt_tokens = vtgt_tokens.to(device)
                        if vir is not None:
                            if isinstance(vir, tuple):
                                vir = (vir[0].to(device), vir[1].to(device))
                            else:
                                vir = vir.to(device)
                        if vh_nmr is not None:
                            if isinstance(vh_nmr, tuple):
                                vh_nmr = (vh_nmr[0].to(device), vh_nmr[1].to(device))
                            else:
                                vh_nmr = vh_nmr.to(device)
                        if vc_nmr is not None:
                            if isinstance(vc_nmr, tuple):
                                vc_nmr = (vc_nmr[0].to(device), vc_nmr[1].to(device))
                            else:
                                vc_nmr = vc_nmr.to(device)

                        T = vtgt_tokens.shape[1]
                        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=vtgt_tokens.device), 1)
                        v_logits = model(vh_nmr, vir, vc_nmr, target_seq=vtgt_tokens[:, :-1], target_mask=mask[:-1, :-1])
                        v_loss = criterion(v_logits.reshape(-1, v_logits.size(-1)), vtgt_tokens[:, 1:].reshape(-1))
                        total_val_loss += v_loss.item()
                        val_count += 1

                avg_val_loss = total_val_loss / max(val_count, 1)
                print(f"[Validation @ step {global_step}] val_loss={avg_val_loss:.4f}")
                wandb.log({"val_loss": avg_val_loss, "global_step": global_step})

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"  New best val_loss: {best_val_loss:.4f}")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss
                    }, save_dir / 'best_model.pt')

        print(f"  [Epoch {epoch+1}] avg_train_loss={(epoch_loss/num_batches):.4f}")
        wandb.log({"train_loss": epoch_loss/num_batches, "epoch": epoch+1})

    # -------------------------------------------------------
    # Final Save
    # -------------------------------------------------------
    final_model_path = save_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"\n[Main] Final model saved to {final_model_path}")

    wandb.finish()


if __name__ == '__main__':
    main()
