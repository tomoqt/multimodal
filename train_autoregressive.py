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

# Import our custom tokenizer
from models.smiles_tokenizer import SmilesTokenizer
from models.multimodal_to_smiles import MultiModalToSMILESModel


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
    Each shard has multiple samples in format:
    {
        global_row_index: {
            'smiles': str,
            'spectra': {
                'ir': Tensor or None,
                'h_nmr': Tensor or None,
                'c_nmr': Tensor or None,
                'hsqc': Tensor or None
            }
        },
        ...
    }
    """
    def __init__(self, data_dir, tokenizer, max_len=128):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load shard paths - now looking for *_rg*.pt pattern
        self.shard_paths = sorted(self.data_dir.glob("*_rg*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found in {data_dir}")
            
        # Build index mapping: [(shard_path, row_idx), ...]
        self.index_map = []
        for shard_path in self.shard_paths:
            shard_data = torch.load(shard_path)
            for row_idx in shard_data.keys():
                self.index_map.append((shard_path, row_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        shard_path, row_idx = self.index_map[idx]
        shard_data = torch.load(shard_path)
        data = shard_data[row_idx]

        # Tokenize SMILES
        smiles = data['smiles']
        tokens = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )

        # Get spectral data from the new format
        spectra = data['spectra']
        ir = spectra.get('ir', None)
        h_nmr = spectra.get('h_nmr', None)
        c_nmr = spectra.get('c_nmr', None)
        hsqc = spectra.get('hsqc', None)

        return tokens, ir, h_nmr, c_nmr, hsqc


# -------------------------------------------------------------------------
# Dataset / DataLoader for Parquet Files (No Chunking)
# -------------------------------------------------------------------------
class ParquetSpectralDataset(Dataset):
    """
    A PyTorch Dataset that reads Parquet files in one go (no row-group chunking).
    Loads everything into a single pandas DataFrame, then samples row-by-row.
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

        # Load entire Parquet dataset to a DataFrame
        self.dataset = ds.dataset(str(self.data_dir), format="parquet")
        full_table = self.dataset.to_table(
            columns=['smiles', 'ir_spectra', 'h_nmr_spectra', 
                     'c_nmr_spectra', 'hsqc_nmr_spectrum']
        )
        self.df = full_table.to_pandas()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row_data = self.df.iloc[idx]

        smiles = row_data['smiles']
        tokens = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )

        def to_tensor(x):
            if x is None:
                return None
            return torch.tensor(x, dtype=torch.float32)

        ir_spectra = to_tensor(row_data['ir_spectra'])
        h_nmr_spectra = to_tensor(row_data['h_nmr_spectra'])
        c_nmr_spectra = to_tensor(row_data['c_nmr_spectra'])
        hsqc_spectra = to_tensor(row_data['hsqc_nmr_spectrum'])

        return tokens, ir_spectra, h_nmr_spectra, c_nmr_spectra, hsqc_spectra


# -------------------------------------------------------------------------
# Collate Function
# -------------------------------------------------------------------------
def collate_fn(batch):
    """
    Custom collate: pad variable-length SMILES tokens, stack spectral data.
    """
    all_tokens, all_ir, all_h_nmr, all_c_nmr, all_hsqc = zip(*batch)

    # Pad tokens
    max_len = max(len(t) for t in all_tokens)
    padded_tokens = []
    for seq in all_tokens:
        pad_amount = max_len - len(seq)
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        if pad_amount > 0:
            pad_tensor = torch.full(
                (pad_amount,), 
                tokenizer.pad_token_id, 
                dtype=torch.long
            )
            seq_tensor = torch.cat([seq_tensor, pad_tensor], dim=0)
        padded_tokens.append(seq_tensor)

    token_batch = torch.stack(padded_tokens, dim=0)

    # Stack spectral data (if not None)
    def maybe_stack(tensors):
        if tensors[0] is not None:
            return torch.stack(tensors, dim=0)
        return None

    ir_batch = maybe_stack(all_ir)
    h_batch = maybe_stack(all_h_nmr)
    c_batch = maybe_stack(all_c_nmr)
    hsqc_batch = maybe_stack(all_hsqc)

    return token_batch, ir_batch, h_batch, c_batch, hsqc_batch


# -------------------------------------------------------------------------
# Create Data Loaders
# -------------------------------------------------------------------------
def create_data_loaders(
    tokenizer,
    batch_size=32,
    use_parquet=False,
    data_dir="data_extraction/multimodal_spectroscopic_dataset",
    binary_dir="training_binaries",
    max_len=128,
):
    """
    Create train/val/test DataLoaders. Default to using binary format.
    """
    if use_parquet:
        dataset = ParquetSpectralDataset(data_dir, tokenizer, max_len)
    else:
        dataset = SpectralSmilesDataset(binary_dir, tokenizer, max_len)
    
    # Split indices
    all_indices = list(range(len(dataset)))
    train_val_indices, test_indices = train_test_split(all_indices, test_size=20, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, random_state=42)
    
    # DataLoaders
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


# -------------------------------------------------------------------------
# Setup: Tokenizer, Model, Data Loaders
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.realpath(__file__))
vocab_path = os.path.join(current_dir, 'vocab.txt')  # Adjust if needed
tokenizer = SmilesTokenizer(vocab_file=vocab_path)

# Key hyperparams
max_seq_length = 128
batch_size = 32
embed_dim = 768
num_heads = 8
num_layers = 6
dropout = 0.1
resample_size = 1000

PAD_TOKEN_ID = tokenizer.pad_token_id
BOS_TOKEN_ID = tokenizer.cls_token_id  # We'll treat [CLS] as BOS
EOS_TOKEN_ID = tokenizer.sep_token_id  # We'll treat [SEP] as EOS

model = MultiModalToSMILESModel(
    vocab_size=len(tokenizer),
    max_seq_length=max_seq_length,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
    resample_size=resample_size
).cuda()

train_loader, val_loader, test_loader = create_data_loaders(
    tokenizer=tokenizer,
    batch_size=batch_size,
    use_parquet=False,
    data_dir="data_extraction/multimodal_spectroscopic_dataset",
    binary_dir="training_binaries",
    max_len=max_seq_length
)


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


def greedy_decode(model, nmr_data, ir_data, hsqc_data, max_len=128):
    """
    Simple greedy decoding for SMILES generation.
    """
    model.eval()
    with torch.no_grad():
        # Start token
        current_token = torch.tensor([[BOS_TOKEN_ID]], device='cuda')
        
        # Encode
        memory = model.encoder(nmr_data, ir_data, hsqc_data)
        
        generated_tokens = [BOS_TOKEN_ID]
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
        for tgt_tokens, ir, h_nmr, c_nmr, hsqc in val_loader:
            tgt_tokens = tgt_tokens.cuda()
            if ir is not None: ir = ir.cuda()
            if h_nmr is not None: h_nmr = h_nmr.cuda()
            if c_nmr is not None: c_nmr = c_nmr.cuda()
            if hsqc is not None: hsqc = hsqc.cuda()
            
            T = tgt_tokens.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), 1)
            
            logits = model(h_nmr, ir, hsqc, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))
            
            # Greedy decode for comparison
            pred_tokens = greedy_decode(model, h_nmr, ir, hsqc)
            
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


# -------------------------------------------------------------------------
# Setup for training
# -------------------------------------------------------------------------
save_dir = Path('checkpoints') / datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir.mkdir(parents=True, exist_ok=True)

best_val_loss = float('inf')
epoch_loss = 0
num_batches = 0

warmup_steps = 100
run_name = (
    f"smiles_gen_"
    f"enc_d768_"
    f"enc_h4_"
    f"dec_l6_"
    f"bs32_"
    f"lr1e-4_warm{warmup_steps}_"
    f"{datetime.now().strftime('%m%d_%H%M')}"
)

wandb.init(
    project="smiles-generation",
    name=run_name,
    config={
        "architecture": "MultiModalToSMILES",
        "encoder_embed_dim": 768,
        "encoder_num_heads": 4,
        "decoder_num_layers": 6,
        "decoder_nhead": 8,
        "decoder_dim_feedforward": 2048,
        "dropout": 0,
        "learning_rate": 1e-4,
        "min_learning_rate": 1e-6,
        "scheduler": "WarmupCosineRestarts",
        "warmup_steps": warmup_steps,
        "scheduler_T0": 5,
        "scheduler_T_mult": 2,
        "batch_size": 32,
        "max_len": 128,
    }
)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Warmup + Cosine
scheduler = WarmupCosineLR(
    optimizer,
    warmup_steps=warmup_steps,
    T_0=5 * len(train_loader),  # in steps
    T_mult=2,
    eta_min=1e-6
)

NUM_EPOCHS = 1
validation_frequency = 500  # validate every 500 steps
global_step = 0

# -------------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------------
model.train()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, (tgt_tokens, ir, h_nmr, c_nmr, hsqc) in enumerate(pbar):
        tgt_tokens = tgt_tokens.cuda()
        if ir is not None:   ir = ir.cuda()
        if h_nmr is not None: h_nmr = h_nmr.cuda()
        if c_nmr is not None: c_nmr = c_nmr.cuda()
        if hsqc is not None: hsqc = hsqc.cuda()

        T = tgt_tokens.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), 1)

        logits = model(h_nmr, ir, hsqc, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = scheduler.get_lr()[0]
        epoch_loss += loss.item()
        num_batches += 1
        global_step += 1

        wandb.log({
            "batch_loss": loss.item(),
            "learning_rate": current_lr,
            "epoch": epoch + 1,
            "global_step": global_step,
        }, step=global_step)

        if batch_idx % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Safely get first item of each spectral data, handling None cases
                h_nmr_batch = h_nmr[0:1] if h_nmr is not None else None
                ir_batch = ir[0:1] if ir is not None else None
                hsqc_batch = hsqc[0:1] if hsqc is not None else None
                
                pred_tokens = greedy_decode(model,
                                          h_nmr_batch,
                                          ir_batch,
                                          hsqc_batch)
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
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for tgt_tokens, ir, h_nmr, c_nmr, hsqc in test_loader:
        if ir is not None: ir = ir.cuda()
        if h_nmr is not None: h_nmr = h_nmr.cuda()
        if c_nmr is not None: c_nmr = c_nmr.cuda()
        if hsqc is not None: hsqc = hsqc.cuda()
        
        pred_tokens = greedy_decode(model, h_nmr, ir, hsqc)
        pred_smiles = tokenizer.decode(pred_tokens[0].tolist(), skip_special_tokens=True)
        target_smiles = tokenizer.decode(tgt_tokens[0].tolist(), skip_special_tokens=True)
        
        all_predictions.append(pred_smiles)
        all_targets.append(target_smiles)

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
