# =======================
# File: train_autoregressive.py
# =======================
"""
Script to train a MultiModalToSMILESModel from memory-mapped binary data.
Key Steps:
1) Loads spectral + SMILES data from .bin and .npy index
2) Tokenizes SMILES
3) Basic training loop with teacher forcing
4) Minimal inference (greedy decode) function
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

# Import our custom tokenizer
from models.smiles_tokenizer import SmilesTokenizer
from models.multimodal_to_smiles import MultiModalToSMILESModel

# Disable RDKit logging
RDLogger.DisableLog("rdApp.*")

current_dir = os.path.dirname(os.path.realpath(__file__))
vocab_path = os.path.join(current_dir, 'vocab.txt')
tokenizer = SmilesTokenizer(vocab_file=vocab_path)


# -------------------------------------------------------------------------
# Linear Warmup + Constant LR Scheduler
# -------------------------------------------------------------------------
class LinearWarmupConstantLR(torch.optim.lr_scheduler._LRScheduler):
    """Simple scheduler with linear warmup followed by constant learning rate"""
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / float(self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Constant learning rate after warmup
            return self.base_lrs


# -------------------------------------------------------------------------
# Dataset / DataLoader for Memory-Mapped Binary Files
# -------------------------------------------------------------------------
class SpectralSmilesDataset(Dataset):
    """
    A PyTorch Dataset that reads from memory-mapped binary files:
      - spectra_data.bin  (float32 arrays for IR/H-NMR/C-NMR + domains)
      - smiles_data.bin   (raw UTF-8 bytes for SMILES)
      - spectra_index.npy (offsets/lengths for each row)
    """
    def __init__(self, data_dir, tokenizer, max_len=128):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load index file
        index_path = self.data_dir / "spectra_index.npy"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found at {index_path}")
        self.index = np.load(index_path)  # shape: (num_rows, 14)

        # Memory-map the data files
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
        """
        Retrieve data & domain for IR, H-NMR, C-NMR + SMILES tokens.
        Returns:
          (tokens, (ir_data, ir_dom), (h_nmr_data, h_nmr_dom), (c_nmr_data, c_nmr_dom))
        """
        row = self.index[idx]

        # Extract all offsets and lengths
        ir_data_off, ir_data_len = row[0], row[1]
        ir_dom_off,  ir_dom_len  = row[2], row[3]
        hnm_data_off, hnm_data_len = row[4], row[5]
        hnm_dom_off,  hnm_dom_len  = row[6], row[7]
        cnm_data_off, cnm_data_len = row[8], row[9]
        cnm_dom_off,  cnm_dom_len  = row[10], row[11]
        smiles_off, smiles_len = row[12], row[13]

        # Helper function to load data slice
        def load_slice(offset, length):
            if offset == -1 or length == 0:
                return None
            data = self.spectra_mmap[offset : offset + length]
            return torch.from_numpy(data.copy())

        # Load all data and domains
        ir_data_t = load_slice(ir_data_off, ir_data_len)
        ir_dom_t  = load_slice(ir_dom_off, ir_dom_len)
        h_nmr_t = load_slice(hnm_data_off, hnm_data_len)
        h_nmr_dom_t = load_slice(hnm_dom_off, hnm_dom_len)
        c_nmr_t = load_slice(cnm_data_off, cnm_data_len)
        c_nmr_dom_t = load_slice(cnm_dom_off, cnm_dom_len)

        # Load and decode SMILES
        if smiles_off == -1 or smiles_len == 0:
            smiles_str = ""
        else:
            smiles_bytes = self.smiles_mmap[smiles_off : smiles_off + smiles_len]
            smiles_str = smiles_bytes.tobytes().decode('utf-8')

        # Tokenize SMILES
        tokens = self.tokenizer.encode(
            smiles_str,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )
        tokens = torch.tensor(tokens, dtype=torch.long)

        return (
            tokens,
            (ir_data_t, ir_dom_t),
            (h_nmr_t, h_nmr_dom_t),
            (c_nmr_t, c_nmr_dom_t)
        )


# -------------------------------------------------------------------------
# Collate Function
# -------------------------------------------------------------------------
def collate_fn(batch):
    """
    Custom collate: pad tokens, then group IR, H-NMR, C-NMR as (data, domain) if available.
    """
    all_tokens, all_ir, all_h_nmr, all_c_nmr = zip(*batch)

    # Pad tokens
    max_len = max(len(seq) for seq in all_tokens)
    padded_tokens = []
    for seq in all_tokens:
        pad_amount = max_len - len(seq)
        seq_tensor = seq
        if pad_amount > 0:
            pad_tensor = torch.full((pad_amount,), tokenizer.pad_token_id, dtype=torch.long)
            seq_tensor = torch.cat([seq_tensor, pad_tensor], dim=0)
        padded_tokens.append(seq_tensor)
    token_batch = torch.stack(padded_tokens, dim=0)

    def stack_data_and_domain(items):
        """Stack data and domain tensors separately"""
        if all(x[0] is None for x in items):
            return None
        
        # Get valid shapes from first non-None item
        first_valid = next((x for x in items if x[0] is not None), None)
        if first_valid is None:
            return None

        # Prepare data and domain lists
        data_list = []
        domain_list = []
        
        for data_t, domain_t in items:
            # Handle data
            if data_t is None:
                data_t = torch.zeros_like(first_valid[0])
            data_list.append(data_t)
            
            # Handle domain
            if domain_t is None:
                domain_t = torch.zeros_like(first_valid[1]) if first_valid[1] is not None else None
            domain_list.append(domain_t)

        # Stack data
        data_batch = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=0.0)
        
        # Stack domain if available
        domain_batch = None
        if domain_list[0] is not None:
            domain_batch = torch.nn.utils.rnn.pad_sequence(domain_list, batch_first=True, padding_value=0.0)
            
        return (data_batch, domain_batch)

    # Stack each modality
    ir_batch = stack_data_and_domain(all_ir)
    h_nmr_batch = stack_data_and_domain(all_h_nmr)
    c_nmr_batch = stack_data_and_domain(all_c_nmr)

    return token_batch, ir_batch, h_nmr_batch, c_nmr_batch


def create_data_loaders(tokenizer, config):
    print("\n[DataLoader] Creating data loaders...")

    # Always load from memory-mapped binaries
    dataset = SpectralSmilesDataset(
        data_dir=config['data']['binary_dir'],
        tokenizer=tokenizer,
        max_len=config['model']['max_seq_length']
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

    # Create loaders
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
# Config, Arg Parsing, etc.
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
            'warmup_steps': 100
        },
        'data': {
            'binary_dir': "training_binaries",  # Path to .bin and .npy index
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


# -------------------------------------------------------------------------
# Main Training Script
# -------------------------------------------------------------------------
def main():
    print("\n[Main] Starting training script...")
    args = parse_args()

    print("[Main] Loading configuration...")
    config = load_config(args.config)
    print("[Main] Configuration loaded.")

    print("\n[Main] Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"[Main] Found {torch.cuda.device_count()} CUDA devices.")
        for i in range(torch.cuda.device_count()):
            print(f"  - {torch.cuda.get_device_name(i)}")
    else:
        print("[Main] No CUDA devices found, using CPU.")
    print(f"[Main] Using device: {device}")

    print("\n[Main] Initializing model...")
    from models.multimodal_to_smiles import MultiModalToSMILESModel
    model = MultiModalToSMILESModel(
        vocab_size=len(tokenizer),
        max_seq_length=config['model']['max_seq_length'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        verbose=False
    ).to(device)


    print("\n[Main] Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        tokenizer=tokenizer,
        config=config
    )

    print("\n[Main] Initializing wandb...")
    run_name = (
        f"{config['wandb']['base_run_name']}_"
        f"d{config['model']['embed_dim']}_"
        f"h{config['model']['num_heads']}_"
        f"l{config['model']['num_layers']}_"
        f"bs{config['training']['batch_size']}_"
        f"lr{config['training']['learning_rate']}_"
        f"warm{config['scheduler']['warmup_steps']}_"
        f"{datetime.now().strftime('%m%d_%H%M')}"
    )

    wandb.init(
        project=config['wandb']['project'],
        name=run_name,
        config=config
    )

    print("[Main] Calculating model size...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    wandb.run.summary.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": param_size_mb
    })
    print(f"[Main] Total parameters: {total_params:,}")
    print(f"[Main] Trainable parameters: {trainable_params:,}")
    print(f"[Main] Model size: {param_size_mb:.2f} MB")

    print("\n[Main] Setting up training components...")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = LinearWarmupConstantLR(optimizer, warmup_steps=config['scheduler']['warmup_steps'])

    print("\n[Main] Creating checkpoint directory...")
    save_dir = Path('checkpoints') / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Main] Checkpoint directory: {save_dir}")

    NUM_EPOCHS = config['training']['num_epochs']
    validation_frequency = config['training']['validation_frequency']
    logging_frequency = config['training']['logging_frequency']
    best_val_loss = float('inf')
    global_step = 0

    # Helper for validation
    def validate(model, loader, criterion, tokenizer, device):
        model.eval()
        total_loss = 0.0
        total_batches = 0
        all_details = []
        
        with torch.no_grad():
            for tokens, ir, h_nmr, c_nmr in loader:
                tokens = tokens.to(device)
                if ir is not None:
                    ir = (ir[0].to(device), ir[1].to(device) if ir[1] is not None else None)
                if h_nmr is not None:
                    h_nmr = (h_nmr[0].to(device), h_nmr[1].to(device) if h_nmr[1] is not None else None)
                if c_nmr is not None:
                    c_nmr = (c_nmr[0].to(device), c_nmr[1].to(device) if c_nmr[1] is not None else None)

                T = tokens.size(1)
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
                
                logits = model(
                    nmr_data=h_nmr,
                    ir_data=ir,
                    c_nmr_data=c_nmr,
                    target_seq=tokens[:, :-1],
                    target_mask=mask[:-1, :-1]
                )
                loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
                
                total_loss += loss.item()
                total_batches += 1

                # Convert logits to predictions for logging
                pred_tokens = logits.argmax(dim=-1).cpu().tolist()
                tgt_tokens = tokens[:, 1:].cpu().tolist()  # Skip BOS token

                # Decode from token IDs to SMILES strings
                preds_decoded = [tokenizer.decode(p) for p in pred_tokens]
                targets_decoded = [tokenizer.decode(t) for t in tgt_tokens]

                # Evaluate predictions vs. targets
                details = evaluate_predictions(preds_decoded, targets_decoded, verbose=False)
                all_details.extend(details)
        
        val_loss = total_loss / max(total_batches, 1)
        metrics_dict = aggregate_metrics(all_details)
        
        # Keep a subset of valid results for logging
        valid_set = [d for d in all_details if d['valid'] and d['valid_target']]
        metrics_dict['valid_set'] = valid_set[:100]  # store up to 100 examples
        metrics_dict['val_loss'] = val_loss

        return metrics_dict

    wandb_table = None

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("\n[Main] Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, total=len(train_loader), desc="Training", dynamic_ncols=True)
        for batch in pbar:
            tokens, ir, h_nmr, c_nmr = batch
            tokens = tokens.to(device)
            if ir is not None:
                ir = (ir[0].to(device), ir[1].to(device) if ir[1] is not None else None)
            if h_nmr is not None:
                h_nmr = (h_nmr[0].to(device), h_nmr[1].to(device) if h_nmr[1] is not None else None)
            if c_nmr is not None:
                c_nmr = (c_nmr[0].to(device), c_nmr[1].to(device) if c_nmr[1] is not None else None)

            T = tokens.size(1)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
            
            logits = model(
                nmr_data=h_nmr,
                ir_data=ir,
                c_nmr_data=c_nmr,
                target_seq=tokens[:, :-1],
                target_mask=mask[:-1, :-1]
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % logging_frequency == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": scheduler.get_lr()[0],
                    "epoch": epoch + 1,
                    "global_step": global_step
                }, step=global_step)

            # Periodic validation
            if global_step % validation_frequency == 0:
                print(f"\nRunning validation at step {global_step}...")
                val_metrics = validate(model, val_loader, criterion, tokenizer, device)
                
                # Initialize wandb table if needed
                if wandb_table is None and val_metrics['valid_set']:
                    columns = ["global_step"] + list(val_metrics['valid_set'][0].keys())
                    wandb_table = wandb.Table(columns=columns)
                
                # Log results
                log_results(val_metrics, global_step, wandb_table, prefix="val")
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
                    if config['training'].get('save_local', False):
                        torch.save(checkpoint, save_dir / 'best_model.pt')
                        print(f"Saved checkpoint locally at {save_dir}/best_model.pt")

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

    # Final test set evaluation
    print("\n[Main] Evaluating on test set...")
    final_test_loss = validate(model, test_loader, criterion, tokenizer, device)
    wandb.log({"test_loss": final_test_loss['val_loss']}, step=global_step)
    print(f"[Test] Loss: {final_test_loss['val_loss']:.4f}")

    # Optionally save final checkpoint
    if config['training'].get('save_local', False):
        final_ckpt = {
            'epoch': NUM_EPOCHS,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': final_test_loss['val_loss']
        }
        torch.save(final_ckpt, save_dir / 'final_model.pt')
        print(f"Final checkpoint saved at {save_dir}/final_model.pt")

    print("[Main] Training script completed.")
    wandb.finish()


if __name__ == '__main__':
    main()
