# =======================
# File: train_autoregressive.py
# =======================
"""
Script to perform learning rate sweep for MultiModalToSMILESModel.
Key Steps:
1) Loads spectral + SMILES data
2) Performs quick training with different learning rates
3) Saves the best learning rate found
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
import math
from tqdm import tqdm
import yaml
import argparse
import itertools

# Import our custom tokenizer
from models.smiles_tokenizer import SmilesTokenizer
from models.multimodal_to_smiles import MultiModalToSMILESModel

current_dir = os.path.dirname(os.path.realpath(__file__))
vocab_path = os.path.join(current_dir, 'vocab.txt')
tokenizer = SmilesTokenizer(vocab_file=vocab_path)

# -------------------------------------------------------------------------
# Linear Warmup + Cosine/Constant LR Scheduler
# -------------------------------------------------------------------------
class LinearWarmupCosineDecay(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by either constant LR or cosine decay.
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

    def __del__(self):
        if hasattr(self, 'spectra_mmap'):
            del self.spectra_mmap
        if hasattr(self, 'smiles_mmap'):
            del self.smiles_mmap

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
    
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

# -------------------------------------------------------------------------
# Config + Argparse
# -------------------------------------------------------------------------
def load_config(config_path=None):
    default_config = {
        'model': {
            'max_seq_length': 128,
            'embed_dim': 768,      # Fixed value
            'num_heads': 8,        # Fixed value
            'num_layers': 6,       # Fixed value
            'dropout': 0.1,        # Fixed value
            'resample_size': 1000,
            'use_concat': True
        },
        'training': {
            'batch_size': 32,
            'test_batch_size': 1,
            'num_epochs': 1,
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],  # LRs to try
            'min_learning_rate': 1.0e-6,
            'validation_frequency': 500,
            'logging_frequency': 100,
            'save_frequency': 1000,
            'generate_during_training': False
        },
        'scheduler': {
            'warmup_steps': 100,  # Fixed value
            'T0': 5,
            'T_mult': 2
        },
        'data': {
            'binary_dir': "training_binaries",
            'test_size': 20,
            'val_size': 0.1
        },
        'wandb': {
            'project': "smiles-generation-sweep",
            'base_run_name': "lr_sweep",
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
    parser = argparse.ArgumentParser(description='Learning rate sweep for SMILES generation model')
    parser.add_argument('--config', type=str, help='Path to config file')
    return parser.parse_args()

# -------------------------------------------------------------------------
# Main Sweep Script
# -------------------------------------------------------------------------
def main():
    print("\n[Main] Starting learning rate sweep...")
    args = parse_args()
    
    print("[Main] Loading configuration...")
    config = load_config(args.config)
    print(f"[Main] Loaded config with {len(config)} sections")

    print("\n[Main] Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"[Main] Found {torch.cuda.device_count()} CUDA devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  - {torch.cuda.get_device_name(i)}")
    else:
        print("[Main] No CUDA devices found, using CPU")
    print(f"[Main] Using device: {device}")

    print("\n[Main] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(tokenizer=tokenizer, config=config)

    print("\n[Main] Initializing wandb...")
    wandb.init(
        project=config['wandb']['project'],
        name=f"{config['wandb']['base_run_name']}_{datetime.now().strftime('%m%d_%H%M')}",
        config=config
    )

    # ---------------------------------------------------------------------
    # Function: quick_train_eval
    #   Does a short training loop with given learning rate and evaluates
    # ---------------------------------------------------------------------
    def quick_train_eval(learning_rate, n_steps=100):
        """Quick training and evaluation with given learning rate"""
        model = MultiModalToSMILESModel(
            vocab_size=len(tokenizer),
            max_seq_length=config['model']['max_seq_length'],
            embed_dim=config['model']['embed_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            resample_size=config['model']['resample_size'],
            verbose=False,
            use_concat=config['model']['use_concat']
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = LinearWarmupCosineDecay(
            optimizer,
            warmup_steps=config['scheduler']['warmup_steps'],
            total_steps=n_steps,
            decay_type='cosine',
            min_lr=config['training']['min_learning_rate']
        )
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        model.train()
        step_count = 0
        train_loss = 0

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

            train_loss += loss.item()
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

        avg_train_loss = train_loss / max(step_count, 1)
        avg_val_loss = total_loss / max(count, 1)
        
        # Clear memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        
        return avg_train_loss, avg_val_loss

    # ---------------------------------------------------------------------
    # Run learning rate sweep
    # ---------------------------------------------------------------------
    print("\n[Main] Starting learning rate sweep...")
    print(f"Learning rates to try: {config['training']['learning_rate']}")

    best_val_loss = float('inf')
    best_lr = None
    results = []

    for lr in tqdm(config['training']['learning_rate'], desc="Learning rates"):
        print(f"\nTrying learning rate: {lr}")
        
        train_loss, val_loss = quick_train_eval(lr)
        
        # Log to wandb
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': lr
        })
        
        # Save results
        results.append({
            'learning_rate': lr,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr = lr
            print(f"\nNew best learning rate found!")
            print(f"Learning rate: {lr}")
            print(f"Validation loss: {val_loss:.4f}")

    # ---------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------
    print("\n[Main] Saving sweep results...")
    save_dir = Path('sweep_results') / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save all results
    with open(save_dir / 'lr_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save best learning rate
    best_result = {
        'best_lr': best_lr,
        'best_val_loss': best_val_loss,
        'model_config': {
            'embed_dim': config['model']['embed_dim'],
            'num_heads': config['model']['num_heads'],
            'num_layers': config['model']['num_layers'],
            'dropout': config['model']['dropout']
        }
    }
    with open(save_dir / 'best_lr.json', 'w') as f:
        json.dump(best_result, f, indent=2)

    print(f"\n[Main] Best learning rate found: {best_lr}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nResults saved to {save_dir}")

    wandb.finish()

if __name__ == '__main__':
    main()
