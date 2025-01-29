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
from rdkit import RDLogger
from pathlib import Path
import numpy as np
import math
from tqdm import tqdm
import yaml
import argparse
import itertools
from logging_utils import evaluate_predictions, aggregate_metrics, log_results
from utils.ezmup import Ezmup

# Disable RDKit logging
RDLogger.DisableLog("rdApp.*")

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
# Dataset / DataLoader for Tokenized Text Files
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
            # Remove spaces when loading SMILES sequences
            self.targets = [line.strip().replace(" ", "") for line in f]

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

        # Add debug printing for NMR tokens
        print(f"\n[Debug] Inspecting first 3 NMR sequences from {split} split:")
        for i in range(min(3, len(self.sources))):
            nmr_seq = self.sources[i]
            nmr_tokens = nmr_seq.split()
            print(f"\nSequence {i+1}:")
            print("Raw tokens:", nmr_tokens[:10], "..." if len(nmr_tokens) > 10 else "")
            print("Token IDs:", [spectral_tokenizer.get(token, spectral_tokenizer["<UNK>"]) for token in nmr_tokens[:10]], 
                  "..." if len(nmr_tokens) > 10 else "")
            print("Sequence length:", len(nmr_tokens))

        # Add debug printing for SMILES sequences
        print(f"\n[Debug] Inspecting first 3 SMILES sequences from {split} split:")
        for i in range(min(3, len(self.targets))):
            smiles_seq = self.targets[i]
            print(f"\nSMILES {i+1}:")
            print("Raw SMILES (before tokenization):", smiles_seq)
            tokens = smiles_tokenizer.encode(
                smiles_seq,
                add_special_tokens=True,
                max_length=max_smiles_len,
                truncation=True
            )
            decoded = smiles_tokenizer.decode(tokens)
            decoded_no_spaces = decoded.replace(" ", "")
            print("Tokenized IDs:", tokens)
            print("Decoded (space-separated tokens):", decoded)
            print("Decoded (no spaces):", decoded_no_spaces)
            print("Token count:", len(tokens))

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

def create_data_loaders(tokenizer, config):
    print("\n[DataLoader] Creating data loaders...")

    # Load vocabularies first
    print("\n[Main] Loading vocabularies...")
    smiles_vocab_size, nmr_vocab_size, nmr_tokenizer = load_vocabularies(config)

    # Create a collate function with the spectral tokenizer
    collate_with_tokenizer = lambda batch: collate_fn(batch, nmr_tokenizer)

    # Create datasets for each split with separate length limits
    train_dataset = SpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        smiles_tokenizer=tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split='train',
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length']
    )

    val_dataset = SpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        smiles_tokenizer=tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split='val',
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length']
    )

    print(f"[DataLoader] Dataset sizes:")
    print(f"          Train: {len(train_dataset)}")
    print(f"          Val: {len(val_dataset)}")

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

    return train_loader, val_loader

# -------------------------------------------------------------------------
# Config + Argparse
# -------------------------------------------------------------------------
def load_config(config_path=None):
    default_config = {
        'model': {
            'max_seq_length': 128,
            'max_nmr_length': 128,
            'width_factor': 8,   
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 1,
            'learning_rate': [1e-5, 5e-5, 1e-4],
        },
        'scheduler': {
            'warmup_steps': 100,
            'type': 'cosine'
        },
        'data': {
            'tokenized_dir': "tokenized_baseline/data",
            'num_workers': 0
        },
        'wandb': {
            'project': "smiles-generation-sweep",
            'base_run_name': "lr_sweep",
        }
    }
    
    if config_path:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Update the default config with values from YAML
            for key in yaml_config:
                if key in default_config:
                    if isinstance(default_config[key], dict):
                        default_config[key].update(yaml_config[key])
                    else:
                        default_config[key] = yaml_config[key]
                else:
                    default_config[key] = yaml_config[key]
    
    return default_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Learning rate sweep for MultiModalToSMILESModel')
    parser.add_argument('--config', type=str, default='configs/sweep_config.yaml',
                      help='Path to config file (defaults to configs/sweep_config.yaml)')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Main] Using device: {device}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(tokenizer, config)

    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        name=f"{config['wandb']['base_run_name']}",
        config=config
    )

    # Table for examples
    columns = ["step", "learning_rate", "prediction", "target", "exact_match", "tanimoto"]
    examples_table = wandb.Table(columns=columns)

    def quick_train_eval(learning_rate, n_steps=100):
        """
        Does short training and evaluation with a given LR.
        Note the changes for ezmup (muP).
        """
        smiles_vocab_size, nmr_vocab_size, nmr_tokenizer = load_vocabularies(config)

        # -- ezmup changes START --
        # Calculate width directly by multiplying the base width (47) with the scale factor (8)
        embed_dim = 13 * config['model']['width_factor']
        
        model = MultiModalToSMILESModel(
            smiles_vocab_size=smiles_vocab_size,
            nmr_vocab_size=nmr_vocab_size,
            max_seq_length=config['model']['max_seq_length'],
            max_nmr_length=config['model']['max_nmr_length'],
            embed_dim=embed_dim,   #  <--- direct width calculation
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            use_stablemax=config['model'].get('use_stablemax', False)
        ).to(device)

        # 2) Wrap the model in Ezmup
        mup_engine = Ezmup(width_basis=13, model=model)

        # (Optional) If you want to do a width sweep as well, you can do:
        # mup_engine.change_width_as( some_width )
        # For just LR sweeping (with a fixed width), you can keep the width_factor as is.
        # e.g.:
        # mup_engine.change_width_as(8)  # or 16, etc.
        #
        # Here I'll just demonstrate the simplest approach:
        # "width_factor" from config is the initial scale, so just keep it:
        #   no .change_width_as(...) call necessary if you're not changing it on the fly

        # 3) Get your optimizer from ezmup, passing the LR
        optimizer = mup_engine.get_optimizer(
            optimizer_class=torch.optim.AdamW,
            lr=learning_rate
        )

        # If you prefer any special config, do e.g.:
        # optimizer = mup_engine.get_optimizer(
        #     optimizer_class=torch.optim.AdamW,
        #     lr=learning_rate,
        #     betas=(0.9, 0.99),  # or other AdamW parameters
        # )

        # For the scheduler, you can use your normal approach
        scheduler = LinearWarmupCosineDecay(
            optimizer,
            warmup_steps=config['scheduler']['warmup_steps'],
            total_steps=n_steps,
            decay_type=config['scheduler']['type'],
            min_lr=config['training'].get('min_learning_rate', 1e-6)
        )
        # -- ezmup changes END --

        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        model.train()

        step_count = 0
        train_loss = 0.0

        # Short training loop
        for batch in train_loader:
            step_count += 1

            tgt_tokens, ir, h_nmr, _ = batch
            tgt_tokens = tgt_tokens.to(device)
            if ir is not None:
                ir = ir.to(device)
            if h_nmr is not None:
                h_nmr = h_nmr.to(device)

            T = tgt_tokens.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1)
            logits = model(nmr_tokens=h_nmr, ir_data=ir, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            if step_count >= n_steps:
                break

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0

        predictions, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                val_steps += 1
                tgt_tokens, ir, h_nmr, _ = batch
                tgt_tokens = tgt_tokens.to(device)
                if ir is not None:
                    ir = ir.to(device)
                if h_nmr is not None:
                    h_nmr = h_nmr.to(device)

                T = tgt_tokens.shape[1]
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1)
                v_logits = model(nmr_tokens=h_nmr, ir_data=ir, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])

                v_loss = criterion(v_logits.reshape(-1, v_logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))
                val_loss += v_loss.item()

                # collect predictions
                pred_tokens = v_logits.argmax(dim=-1).cpu()
                for i in range(len(pred_tokens)):
                    pred_seq = pred_tokens[i].tolist()
                    tgt_seq  = tgt_tokens[i, 1:].cpu().tolist()  # skip BOS
                    # strip at SEP if found
                    if tokenizer.sep_token_id in pred_seq:
                        sep_idx = pred_seq.index(tokenizer.sep_token_id)
                        pred_seq = pred_seq[:sep_idx]
                    if tokenizer.sep_token_id in tgt_seq:
                        sep_idx = tgt_seq.index(tokenizer.sep_token_id)
                        tgt_seq  = tgt_seq[:sep_idx]

                    predictions.append(tokenizer.decode(pred_seq))
                    targets.append(tokenizer.decode(tgt_seq))

        avg_train_loss = train_loss / max(1, step_count)
        avg_val_loss   = val_loss   / max(1, val_steps)

        # Evaluate SMILES metrics
        # (your evaluate_predictions & aggregate_metrics):
        detailed_results = evaluate_predictions(predictions, targets)
        metrics = aggregate_metrics(detailed_results)

        # Log a few examples in the wandb table
        for (pred, tgt) in zip(predictions[:5], targets[:5]):
            pair_res = evaluate_predictions([pred], [tgt])[0]
            examples_table.add_data(
                step_count,
                learning_rate,
                pred,
                tgt,
                pair_res['exact_match'],
                pair_res['tanimoto']
            )

        return avg_train_loss, avg_val_loss, metrics

    # Sweep over learning rates
    best_val_loss = float('inf')
    best_lr = None

    for lr in tqdm(config['training']['learning_rate'], desc="LR Sweep"):
        train_loss, val_loss, metrics = quick_train_eval(lr, n_steps=100)

        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': lr,
            'exact_match': metrics['exact_match'],
            'avg_tanimoto': metrics['avg_tanimoto']
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lr = lr

    # Save best LR, etc.
    print(f"Best LR found: {best_lr}")
    wandb.log({"examples": examples_table})
    wandb.finish()

if __name__ == '__main__':
    main()