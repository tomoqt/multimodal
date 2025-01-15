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
        # Get target sequence (SMILES) - use SMILES tokenizer
        target_seq = self.targets[idx]
        target_tokens = self.smiles_tokenizer.encode(
            target_seq,
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
def collate_fn(batch, spectral_tokenizer):
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


def load_vocabularies(config):
    """Load both SMILES and NMR vocabularies and return their sizes"""
    # Load NMR vocabulary
    nmr_vocab_path = Path(config['data']['tokenized_dir']).parent / "vocab.json"
    if not nmr_vocab_path.exists():
        raise FileNotFoundError(f"NMR vocabulary not found at {nmr_vocab_path}")

    with open(nmr_vocab_path) as f:
        nmr_tokenizer = json.load(f)
    
    # Check the actual range of token IDs
    token_ids = list(nmr_tokenizer.values())
    min_id = min(token_ids)
    max_id = max(token_ids)
    nmr_vocab_size = max_id + 1  # Adjust vocab size to accommodate highest token ID
    
    print(f"[Vocab] Loaded NMR vocabulary with {len(nmr_tokenizer)} tokens")
    print(f"[Vocab] NMR token ID range: [{min_id}, {max_id}]")
    print(f"[Vocab] Setting NMR vocab size to: {nmr_vocab_size}")

    # SMILES vocabulary size comes from the tokenizer
    smiles_vocab_size = len(tokenizer)
    print(f"[Vocab] SMILES vocabulary has {smiles_vocab_size} tokens")

    return smiles_vocab_size, nmr_vocab_size, nmr_tokenizer


def create_data_loaders(smiles_tokenizer, nmr_tokenizer, config):
    print("\n[DataLoader] Creating data loaders...")

    # Create a collate function with the spectral tokenizer
    collate_with_tokenizer = lambda batch: collate_fn(batch, nmr_tokenizer)

    # Create datasets for each split with separate length limits
    train_dataset = SpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split='train',
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length']
    )

    val_dataset = SpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split='val',
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length']
    )

    test_dataset = SpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split='test',
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length']
    )

    print(f"[DataLoader] Dataset sizes:")
    print(f"          Train: {len(train_dataset)}")
    print(f"          Val: {len(val_dataset)}")
    print(f"          Test: {len(test_dataset)}")

    # Create data loaders with the wrapped collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        collate_fn=collate_with_tokenizer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        collate_fn=collate_with_tokenizer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training'].get('test_batch_size', 1),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        collate_fn=collate_with_tokenizer
    )

    return train_loader, val_loader, test_loader


# -------------------------------------------------------------------------
# Config, Arg Parsing, etc.
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


# -------------------------------------------------------------------------
# Main Training Script
# -------------------------------------------------------------------------
def main():
    print("\n[Main] Starting training script...")
    args = parse_args()

    print("[Main] Loading configuration...")
    config = load_config(args.config)
    print("[Main] Configuration loaded.")

    # Load vocabularies first
    print("\n[Main] Loading vocabularies...")
    smiles_vocab_size, nmr_vocab_size, nmr_tokenizer = load_vocabularies(config)

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
    model = MultiModalToSMILESModel(
        smiles_vocab_size=smiles_vocab_size,
        nmr_vocab_size=nmr_vocab_size,
        max_seq_length=config['model']['max_seq_length'],
        max_nmr_length=config['model']['max_nmr_length'],
        max_memory_length=config['model']['max_memory_length'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        verbose=False
    ).to(device)

    print("\n[Main] Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        smiles_tokenizer=tokenizer,
        nmr_tokenizer=nmr_tokenizer,  # Pass the loaded tokenizer
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
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id
    )
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])

    # Calculate total training steps (batches per epoch * num epochs)
    total_training_steps = len(train_loader) * config['training']['num_epochs']
    print(f"[Main] Total training steps: {total_training_steps:,}")
    
    # Initialize scheduler
    scheduler = LinearWarmupCosineDecay(
        optimizer,
        warmup_steps=config['scheduler']['warmup_steps'],
        total_steps=total_training_steps,
        decay_type=config['scheduler'].get('type', 'constant'),
        min_lr=config['training'].get('min_learning_rate', 1e-6)
    )
    
    print(f"[Main] Using {config['scheduler'].get('type', 'constant')} scheduler with:")
    print(f"      - Warmup steps: {config['scheduler']['warmup_steps']}")
    print(f"      - Total steps: {total_training_steps}")
    if config['scheduler'].get('type') == 'cosine':
        print(f"      - Min LR: {config['training'].get('min_learning_rate', 1e-6)}")

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
        predictions = []
        targets = []
        
        with torch.no_grad():
            for target_tokens, ir_data, nmr_tokens, _ in loader:
                target_tokens = target_tokens.to(device)
                if ir_data is not None:
                    ir_data = ir_data.to(device)
                if nmr_tokens is not None:
                    nmr_tokens = nmr_tokens.to(device)

                T = target_tokens.size(1)
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=target_tokens.device), 1)
                
                logits = model(
                    nmr_tokens=nmr_tokens,
                    ir_data=ir_data,
                    target_seq=target_tokens[:, :-1],
                    target_mask=mask[:-1, :-1]
                )
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens[:, 1:].reshape(-1))
                
                # Get predictions and immediately move to CPU
                pred_tokens = logits.argmax(dim=-1).cpu().tolist()
                tgt_tokens = target_tokens[:, 1:].cpu().tolist()
                
                # Decode predictions, including SEP token as it marks sequence end
                for pred_seq in pred_tokens:
                    # Find the first occurrence of SEP token if it exists
                    try:
                        sep_idx = pred_seq.index(tokenizer.sep_token_id)
                        # Include SEP token in the sequence
                        pred_seq = pred_seq[:sep_idx + 1]
                    except ValueError:
                        # No SEP token found, use full sequence
                        pass
                        
                    # Decode the sequence including SEP
                    decoded = tokenizer.decode(pred_seq).strip()
                    predictions.append(decoded)

                # Decode targets similarly
                for tgt_seq in tgt_tokens:
                    try:
                        sep_idx = tgt_seq.index(tokenizer.sep_token_id)
                        tgt_seq = tgt_seq[:sep_idx + 1]  # Include SEP
                    except ValueError:
                        pass
                    decoded = tokenizer.decode(tgt_seq).strip()
                    targets.append(decoded)
                
                # Clear GPU tensors we don't need anymore
                del logits, mask
                if ir_data is not None:
                    del ir_data
                if nmr_tokens is not None:
                    del nmr_tokens
                
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
            'predictions': predictions[:10],  # Store all predictions
            'targets': targets[:10],         # Store all targets
            'num_samples': len(predictions)
        }
        
        return combined_metrics

    # Initialize wandb table outside the validation loop
    columns = ["step", "prediction", "target", "exact_match", "tanimoto", "mcs_ratio", "ecfp6_iou"]
    examples_table = wandb.Table(columns=columns)

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
            target_tokens, ir_data, nmr_tokens, _ = batch
            
            target_tokens = target_tokens.to(device)
            if ir_data is not None:
                ir_data = ir_data.to(device)
            if nmr_tokens is not None:
                nmr_tokens = nmr_tokens.to(device)

            T = target_tokens.size(1)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=target_tokens.device), 1)
            
            logits = model(
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                target_seq=target_tokens[:, :-1],
                target_mask=mask[:-1, :-1]
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Note: Scheduler steps every batch, not every epoch
            
            # Clear memory after backward pass
            del logits, mask
            if ir_data is not None:
                del ir_data
            if nmr_tokens is not None:
                del nmr_tokens
            
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
                val_metrics = validate(model, val_loader, criterion, tokenizer, device)
                
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
