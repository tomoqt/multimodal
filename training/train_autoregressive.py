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
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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
from utils.logging.logging_utils import evaluate_predictions, aggregate_metrics, log_results
import heavyball
from utils.optimization.ortho_grad import OrthoGrad  # Import our new optimizer wrapper
# Import our custom tokenizer
from models.smiles_tokenizer import SmilesTokenizer
from models.multimodal_to_smiles import MultiModalToSMILESModel
import subprocess
from utils.optimization.muon import Muon  # Import Muon optimizer from the local file
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# Disable RDKit logging
RDLogger.DisableLog("rdApp.*")

try:
    import torch._dynamo
    torch._dynamo.config.backend = "aot_eager"
    print("[TorchDynamo] Set backend to aot_eager to avoid inductor backend issues.")
except Exception as e:
    print(f"[TorchDynamo] Warning: could not set backend: {e}")

current_dir = os.path.dirname(os.path.realpath(__file__))
vocab_path = os.path.join(current_dir, 'vocab.txt')
tokenizer = SmilesTokenizer(vocab_file=vocab_path)


def greedy_decode(model, nmr_tokens, ir_data, tokenizer, max_len=128, device=None, temperature=1.0, sample=False, precision='fp32'):
    """
    Decoding for SMILES generation with optional sampling.
    Args:
        model: The MultiModalToSMILESModel instance
        nmr_tokens: NMR token tensor or None
        ir_data: IR data tensor or None
        tokenizer: SmilesTokenizer instance
        max_len: Maximum sequence length for generation
        device: torch device to use
        temperature: Temperature for sampling (higher = more random, lower = more deterministic)
        sample: If True, sample from the distribution; if False, use greedy decoding (argmax)
        precision: Precision type ('fp32', 'fp16', 'bf16')
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Get tokenizer's special token IDs
    BOS_TOKEN_ID = tokenizer.cls_token_id  # [CLS] token is used as BOS
    EOS_TOKEN_ID = tokenizer.sep_token_id  # [SEP] token is used as EOS
    PAD_TOKEN_ID = tokenizer.pad_token_id
        
    model.eval()
    with torch.no_grad():
        # Get batch size from input data
        batch_size = 1  # Default
        if nmr_tokens is not None:
            batch_size = nmr_tokens.size(0)
        elif ir_data is not None:
            batch_size = ir_data.size(0)
            
        # Start tokens for each sequence in the batch
        current_token = torch.tensor([[BOS_TOKEN_ID]] * batch_size, device=device)
        
        # Encode spectral data with appropriate precision
        use_amp = (precision in ['fp16', 'bf16']) and torch.cuda.is_available()
        
        if use_amp:
            amp_dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                memory = model.encoder(None, ir_data, None)  # NMR tokens not needed here
        else:
            memory = model.encoder(None, ir_data, None)  # NMR tokens not needed here
        
        if memory is None:
            if nmr_tokens is not None:
                batch_size = nmr_tokens.size(0)
            elif ir_data is not None:
                batch_size = ir_data.size(0)
            else:
                batch_size = 1
            memory = torch.zeros(batch_size, model.decoder.max_memory_length, model.decoder.memory_dim, device=device)
        
        # Initialize storage for generated tokens
        generated_sequences = [[] for _ in range(batch_size)]
        for seq in generated_sequences:
            seq.append(BOS_TOKEN_ID)
        
        # Use model's max sequence length as the limit
        max_len = min(max_len, model.decoder.max_seq_length)
        
        finished_sequences = [False] * batch_size
        
        for _ in range(max_len):
            # Get next token predictions with appropriate precision
            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits = model.decoder(
                        tgt=current_token,
                        memory=memory,
                        nmr_tokens=nmr_tokens  # NMR tokens used here
                    )
            else:
                logits = model.decoder(
                    tgt=current_token,
                    memory=memory,
                    nmr_tokens=nmr_tokens  # NMR tokens used here
                )
            
            # Get the next token - either sample or take argmax
            if sample and temperature > 0:
                # Apply temperature and convert to probabilities
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding (argmax)
                next_token = logits[:, -1:].argmax(dim=-1)
            
            # Update each sequence
            for i in range(batch_size):
                if not finished_sequences[i]:
                    token = next_token[i].item()
                    generated_sequences[i].append(token)
                    if token == EOS_TOKEN_ID:
                        finished_sequences[i] = True
            
            # Stop if all sequences are finished
            if all(finished_sequences):
                break
            
            current_token = torch.cat([current_token, next_token], dim=1)
        
        # Decode sequences
        decoded_sequences = []
        for seq in generated_sequences:
            # Find EOS token if present
            try:
                eos_idx = seq.index(EOS_TOKEN_ID)
                seq = seq[:eos_idx]  # Exclude EOS token
            except ValueError:
                pass  # No EOS found, use full sequence
            
            # Remove BOS token and decode
            decoded = tokenizer.decode(seq[1:])  # Skip BOS token
            decoded_sequences.append(decoded)
            
        return decoded_sequences


# Updated helper function for canonicalizing SMILES using RDKit: remove spaces before canonicalization

def canonicalize_smiles(smiles):
    """Convert a SMILES string to its canonical form using RDKit. Removes extra spaces before conversion. Returns the canonical SMILES if possible, otherwise returns the cleaned string."""
    # Remove spaces and strip leading/trailing whitespace
    cleaned = smiles.replace(' ', '').strip()
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return cleaned
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return cleaned


def evaluate_with_greedy_decode(model, test_loader, tokenizer, device, num_examples=None, block_ir=False, block_nmr=False):
    """Evaluate model using greedy decoding with optional IR/NMR blocking"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Get current precision setting from config if it exists
    precision = 'fp32'  # Default
    try:
        # Access the global config if available
        precision = config['training'].get('precision', 'fp32')
    except (NameError, KeyError):
        pass

    with torch.no_grad():
        for target_tokens, ir_data, nmr_tokens, _ in tqdm(test_loader, desc="Greedy decoding"):
            # Block modalities if flags are set
            if block_ir:
                ir_data = None
            if block_nmr:
                nmr_tokens = None

            # Move data to device
            if ir_data is not None:
                ir_data = ir_data.to(device)
            if nmr_tokens is not None:
                nmr_tokens = nmr_tokens.to(device)

            predictions = greedy_decode(
                model=model,
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                tokenizer=tokenizer,
                device=device,
                sample=False,  # Ensure we're using greedy decoding (not sampling) for evaluation
                precision=precision  # Pass precision setting
            )

            targets = []
            for tgt in target_tokens:
                try:
                    eos_idx = tgt.tolist().index(tokenizer.sep_token_id)
                    tgt = tgt[:eos_idx]
                except ValueError:
                    pass
                decoded = tokenizer.decode(tgt[1:]).strip()
                targets.append(decoded)

            all_predictions.extend(predictions)
            all_targets.extend(targets)

            if num_examples and len(all_predictions) >= num_examples:
                break

    # Canonicalize predictions and targets before evaluation
    all_predictions = [canonicalize_smiles(pred) for pred in all_predictions]
    all_targets = [canonicalize_smiles(tgt) for tgt in all_targets]

    detailed_results = evaluate_predictions(all_predictions, all_targets)
    metrics = aggregate_metrics(detailed_results)
    metrics['predictions'] = all_predictions[:10]
    metrics['targets'] = all_targets[:10]
    metrics['num_samples'] = len(all_predictions)
    return metrics


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
        max_nmr_len=128,      # Separate length limit for NMR
        ir_as_prompt=False,   # NEW: flag to indicate IR should be processed as prompt tokens
        ir_tokenizer=None     # NEW: IR vocabulary mapping for tokenization
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.smiles_tokenizer = smiles_tokenizer
        self.spectral_tokenizer = spectral_tokenizer
        self.max_smiles_len = max_smiles_len
        self.max_nmr_len = max_nmr_len
        self.split = split
        self.ir_as_prompt = ir_as_prompt
        self.ir_tokenizer = ir_tokenizer

        # Load source (NMR) and target sequences
        with open(self.data_dir / f"src-{split}.txt") as f:
            self.sources = [line.strip() for line in f]
        with open(self.data_dir / f"tgt-{split}.txt") as f:
            # Remove spaces when loading SMILES sequences
            self.targets = [line.strip().replace(" ", "") for line in f]

        # Load IR data - either from pre-tokenized text file or memory-mapped binary
        if self.ir_as_prompt:
            # Load pre-tokenized IR data from text file
            ir_text_path = self.data_dir.parent / "ir_processed" / f"ir-{split}.txt"
            if not ir_text_path.exists():
                raise FileNotFoundError(
                    f"IR text file not found at {ir_text_path}. "
                    "Please run build_ir_vocab.py first to generate tokenized IR data."
                )
            with open(ir_text_path) as f:
                self.ir_sources = [line.strip() for line in f]
            print(f"[Dataset] Loaded tokenized IR data from {ir_text_path}")
            self.ir_data = None  # Not needed when using pre-tokenized data
        else:
            # Load raw IR data using memory-mapped binary
            self.ir_sources = None
            ir_path = self.data_dir / f"ir-{split}.npy"
            self.ir_data = None
            if ir_path.exists():
                try:
                    self.ir_data = np.memmap(
                        ir_path,
                        dtype='float32',
                        mode='r',
                        shape=None
                    )
                    array_shape = self.ir_data.shape
                    if len(array_shape) == 1:
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

        # Get IR data - either from pre-tokenized text or raw data
        ir_data = None
        if self.ir_as_prompt and self.ir_sources is not None:
            # Use pre-tokenized IR data
            ir_seq = self.ir_sources[idx]
            ir_tokens = ir_seq.split()
            ir_token_ids = [self.ir_tokenizer.get(token, self.ir_tokenizer["<UNK>"]) 
                          for token in ir_tokens]
            ir_data = torch.tensor(ir_token_ids, dtype=torch.long)
        elif self.ir_data is not None:
            # Use raw IR data
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

    # Load IR tokenizer if IR as prompt is enabled
    ir_tokenizer = None
    ir_vocab_size = None
    if config['data'].get('ir_as_prompt', False):
        ir_vocab_path = config['data'].get('ir_tokenizer_path')
        if not ir_vocab_path:
            raise ValueError("IR as prompt is enabled but 'ir_tokenizer_path' is not provided in config.")
        with open(ir_vocab_path, 'r') as f:
            ir_tokenizer = json.load(f)
            ir_vocab_size = len(ir_tokenizer)  # Get vocabulary size for model initialization

    # Create a collate function with the spectral tokenizer
    collate_with_tokenizer = lambda batch: collate_fn(batch, nmr_tokenizer)

    # Create datasets for each split with separate length limits, passing the new IR prompt parameters
    train_dataset = SpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split='train',
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length'],
        ir_as_prompt=config['data'].get('ir_as_prompt', False),
        ir_tokenizer=ir_tokenizer
    )

    val_dataset = SpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split='val',
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length'],
        ir_as_prompt=config['data'].get('ir_as_prompt', False),
        ir_tokenizer=ir_tokenizer
    )

    test_dataset = SpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split='test',
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length'],
        ir_as_prompt=config['data'].get('ir_as_prompt', False),
        ir_tokenizer=ir_tokenizer
    )

    print(f"[DataLoader] Dataset sizes:")
    print(f"          Train: {len(train_dataset)}")
    print(f"          Val: {len(val_dataset)}")
    print(f"          Test: {len(test_dataset)}")

    # Create distributed samplers if running in distributed mode
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # Create data loaders with the wrapped collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),  # Only shuffle if not using distributed sampler
        num_workers=config['data'].get('num_workers', 0),
        collate_fn=collate_with_tokenizer,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        collate_fn=collate_with_tokenizer,
        sampler=val_sampler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training'].get('test_batch_size', 1),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        collate_fn=collate_with_tokenizer,
        sampler=test_sampler
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
            'use_concat': True,
            'use_stablemax': False,
            'ir_encoder_type': 'regular',
            'max_loops': 1,
            'loop_range': [0, 5]  # Range for uniform sampling of loop count during training
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
            'save_local': False,
            'greedy_decode_frequency': 1000,
            'weight_decay': 0.01
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
        },
        'optimizer': {
            'type': 'adamw',  # Options: 'adamw', 'foreachadopt', 'ortho_adamw', 'foreachmuon', 'muon_mix'
            'adamw': {
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.1,
                'caution': False
            },
            'foreachadopt': {
                'caution': True
            },
            'ortho': {
                'eps': 1e-30,
                'rescale': True
            },
            'foreachmuon': {
                'betas': (0.9, 0.99),
                'eps': 1e-8,
                'weight_decay': 0.01,
                'warmup_steps': 0,
                'beta2_scale': 0.8,
                'nesterov': True,
                'mars': False,
                'mars_gamma': 0.0025,
                'caution': False
            },
            'muon': {
                'lr': 0.02,
                'weight_decay': 0.01,
                'momentum': 0.95,
                'nesterov': True,
                'ns_steps': 5
            }
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
    parser.add_argument('--block-ir', action='store_true', help='Block IR signals in the model inputs')
    parser.add_argument('--block-nmr', action='store_true', help='Block NMR signals in the model inputs')
    return parser.parse_args()


def cleanup_wandb_cache():
    """Clean up wandb cache to prevent disk space issues"""
    try:
        # Clean files older than 24 hours and don't include online runs
        subprocess.run(['wandb', 'sync', '--clean-old-hours', '24', '--no-include-online'], 
                      capture_output=True, text=True)
        # Clean artifact cache over 5GB
        subprocess.run(['wandb', 'artifact', 'cache', 'cleanup', '1GB'],
                      capture_output=True, text=True)
        print("[wandb] Cache cleaned successfully")
    except Exception as e:
        print(f"[wandb] Cache cleanup failed: {e}")


# -------------------------------------------------------------------------
# Main Training Script
# -------------------------------------------------------------------------
def main():
    print("\n[Main] Starting training script...")
    args = parse_args()
    block_ir = args.block_ir
    block_nmr = args.block_nmr

    # Initialize distributed training if running with torchrun
    rank = 0
    world_size = 1
    if "LOCAL_RANK" in os.environ:
        print("[Main] Detected distributed environment (torchrun)")
        
        # Initialize process group with gloo backend
        torch.distributed.init_process_group(
            backend="gloo",  # Use gloo instead of NCCL
            init_method="env://",
        )
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        print(f"[Main] Process rank: {rank}, world size: {world_size}")
        
        # Set device based on local rank
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        
        # Print GPU topology information on rank 0
        if rank == 0:
            print("\n[Main] GPU Topology:")
            try:
                topo = subprocess.check_output(['nvidia-smi', 'topo', '-m']).decode()
                print(topo)
            except Exception as e:
                print(f"Could not get GPU topology: {e}")
    else:
        print("[Main] Running in non-distributed mode")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Helper function for wandb logging that only logs on rank 0
    def log_wandb(metrics, step=None):
        if rank == 0 and wandb.run is not None:
            wandb.log(metrics, step=step)

    print("[Main] Loading configuration...")
    config = load_config(args.config)
    print("[Main] Configuration loaded.")

    # Clean wandb cache before starting
    print("\n[Main] Cleaning wandb cache...")
    cleanup_wandb_cache()

    # Variable to store loaded checkpoint for restoring optimizer state later
    loaded_checkpoint = None
    global_step = 0

    # Load vocabularies first
    print("\n[Main] Loading vocabularies...")
    smiles_vocab_size, nmr_vocab_size, nmr_tokenizer = load_vocabularies(config)

    print("\n[Main] Setting up device...")
    if torch.cuda.is_available():
        print(f"[Main] Found {torch.cuda.device_count()} CUDA devices.")
        for i in range(torch.cuda.device_count()):
            print(f"  - {torch.cuda.get_device_name(i)}")
    else:
        print("[Main] No CUDA devices found, using CPU.")
    print(f"[Main] Using device: {device}")

    print("\n[Main] Initializing model...")
    ir_vocab_size = None
    if config['data'].get('ir_as_prompt', False):
        ir_vocab_path = config['data'].get('ir_tokenizer_path')
        if not ir_vocab_path:
            raise ValueError("IR as prompt is enabled but 'ir_tokenizer_path' is not provided in config.")
        with open(ir_vocab_path, 'r') as f:
            ir_vocab = json.load(f)
            ir_vocab_size = len(ir_vocab)
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
        verbose=False,
        use_stablemax=config['model'].get('use_stablemax', False),
        ir_encoder_type=config['model'].get('ir_encoder_type', 'regular'),
        ir_as_prompt=config['data'].get('ir_as_prompt', False),
        ir_vocab_size=ir_vocab_size,
        max_loops=max(config['model'].get('max_loops', 1), max(config['model'].get('loop_range', [0, 1])))
    )
    
    # Set precision for training based on configuration
    precision = config['training'].get('precision', 'fp32')
    print(f"[Main] Using {precision} precision for training")
    
    # Load checkpoint if specified in config
    if 'checkpoint' in config and 'load_path' in config['checkpoint'] and config['checkpoint']['load_path']:
        checkpoint_path = config['checkpoint']['load_path']
        print(f"\n[Main] Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Main] Successfully loaded model from checkpoint at epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
            
            # Record the best validation loss from the checkpoint
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"[Main] Checkpoint validation loss: {best_val_loss:.4f}")
            
            # Save global_step from checkpoint if available
            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
                print(f"[Main] Resuming from global step: {global_step}")
            
            # Store checkpoint for optimizer state restoration later
            loaded_checkpoint = checkpoint
        except Exception as e:
            print(f"[Main] Error loading checkpoint: {e}")
    
    if precision == 'fp16' and torch.cuda.is_available():
        print("[Main] Enabling automatic mixed precision (AMP) for FP16 training")
        # Do NOT convert model to half precision - this causes issues with GradScaler
        # The autocast context manager will handle precision conversion during forward pass
    elif precision == 'bf16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("[Main] Enabling bfloat16 precision for training")
        # For bf16, we can use either approach. We'll use autocast for consistency.
    else:
        # Default to fp32
        if precision != 'fp32':
            print(f"[Main] Warning: Requested precision {precision} not supported. Using fp32 instead.")
        print("[Main] Using default full (FP32) precision")
    
    # Move model to device (always in default precision)
    model = model.to(device)
    
    # Wrap model with DDP if running in distributed mode
    if torch.distributed.is_initialized():
        model = DDP(model, device_ids=[rank], output_device=rank)

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

    # Only initialize wandb on the main process (rank 0) in distributed mode
    if rank == 0:
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

    # Initialize optimizer based on config
    optimizer_type = config.get('optimizer', {}).get('type', 'adamw')
    
    if optimizer_type == 'foreachadopt':
        optimizer = heavyball.ForeachADOPT(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            caution=config['optimizer']['foreachadopt'].get('caution', True)
        )
        optimizers = [optimizer]
    elif optimizer_type == 'foreachmuon':
        optimizer = heavyball.ForeachMuon(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=config['optimizer']['foreachmuon'].get('betas', (0.9, 0.99)),
            eps=config['optimizer']['foreachmuon'].get('eps', 1e-8),
            weight_decay=config['training']['weight_decay'],  # Use weight decay from training config
            warmup_steps=config['optimizer']['foreachmuon'].get('warmup_steps', 0),
            beta2_scale=config['optimizer']['foreachmuon'].get('beta2_scale', 0.8),
            nesterov=config['optimizer']['foreachmuon'].get('nesterov', True),
        )
        optimizers = [optimizer]
    elif optimizer_type == 'muon_mix':
        # Directly use the approach from the Muon repository
        # Filter parameters by dimensionality
        matrix_params = [p for p in model.parameters() if p.ndim >= 2]
        vector_params = [p for p in model.parameters() if p.ndim < 2]
        
        # Print parameter counts for each optimizer
        matrix_param_count = sum(p.numel() for p in matrix_params)
        vector_param_count = sum(p.numel() for p in vector_params)
        total_params = matrix_param_count + vector_param_count
        
        print(f"[Optimizer] Parameter distribution:")
        print(f"  - Muon (≥2D): {matrix_param_count:,} parameters ({matrix_param_count/total_params:.1%})")
        print(f"  - AdamW (<2D): {vector_param_count:,} parameters ({vector_param_count/total_params:.1%})")
        
        # Get distributed training info from torch.distributed if available
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        # Create separate optimizers in a list
        muon_opt = Muon(
            matrix_params,
            lr=config['optimizer'].get('muon', {}).get('lr', 0.02),
            weight_decay=config['training']['weight_decay'],
            momentum=config['optimizer'].get('muon', {}).get('momentum', 0.95),
            nesterov=config['optimizer'].get('muon', {}).get('nesterov', True),
            ns_steps=config['optimizer'].get('muon', {}).get('ns_steps', 5),
            rank=rank,
            world_size=world_size,
            orthogonalize=config['optimizer']['muon'].get('orthogonalize', False),
            ortho_eps=config['optimizer']['muon'].get('ortho_eps', 1e-30),
            ortho_rescale=config['optimizer']['muon'].get('ortho_rescale', True),
            use_distributed=False  # Disable distributed communication in Muon
        ) if matrix_params else None
        
        adamw_opt = optim.AdamW(
            vector_params,
            lr=config['training']['learning_rate'],
            betas=config['optimizer']['adamw'].get('betas', (0.9, 0.999)),
            eps=config['optimizer']['adamw'].get('eps', 1e-8),
            weight_decay=config['training']['weight_decay']
        ) if vector_params else None
        
        # Create list of optimizers (filtering out None)
        optimizers = [opt for opt in [muon_opt, adamw_opt] if opt is not None]
        
        # For scheduler compatibility, we'll use the first optimizer
        # The learning rate scheduler will only apply to this optimizer
        optimizer = optimizers[0] if optimizers else None
        
        # Debug: print learning rates for optimizers
        if muon_opt is not None:
            for i, group in enumerate(muon_opt.param_groups):
                print(f"[Muon Optimizer] Group {i} lr: {group['lr']}")
        if adamw_opt is not None:
            for i, group in enumerate(adamw_opt.param_groups):
                print(f"[AdamW Optimizer] Group {i} lr: {group['lr']}")
    elif optimizer_type == 'ortho_adamw':
        # Use our orthogonal gradient wrapper with AdamW
        # Separate base optimizer args from ortho args
        base_args = {
            'lr': config['training']['learning_rate'],
            'betas': config['optimizer']['adamw'].get('betas', (0.9, 0.999)),
            'weight_decay': config['training']['weight_decay']  # Use weight decay from training config
        }
        # Only pass eps to base optimizer if not using ortho's eps
        if not config['optimizer'].get('ortho', {}).get('eps'):
            base_args['eps'] = config['optimizer']['adamw'].get('eps', 1e-8)
            
        # Create optimizer with proper parameter separation
        optimizer = OrthoGrad(
            model.parameters(),
            base_optimizer_cls=optim.AdamW,
            eps=config['optimizer']['ortho'].get('eps', 1e-30),
            rescale=config['optimizer']['ortho'].get('rescale', True),
            **base_args
        )
        optimizers = [optimizer]
    else:  # AdamW variants
        use_caution = config['optimizer']['adamw'].get('caution', False)
        if use_caution:
            optimizer = heavyball.AdamW(
                model.parameters(),
                lr=config['training']['learning_rate'],
                betas=config['optimizer']['adamw'].get('betas', (0.9, 0.999)),
                eps=config['optimizer']['adamw'].get('eps', 1e-8),
                weight_decay=config['training']['weight_decay'],  # Use weight decay from training config
                caution=True
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['training']['learning_rate'],
                betas=config['optimizer']['adamw'].get('betas', (0.9, 0.999)),
                eps=config['optimizer']['adamw'].get('eps', 1e-8),
                weight_decay=config['training']['weight_decay']  # Use weight decay from training config
            )
        optimizers = [optimizer]

    print(f"[Main] Using optimizer: {optimizer_type}")
    if optimizer_type == 'ortho_adamw':
        print(f"      - Base optimizer: AdamW")
        print(f"      - Orthogonalization eps: {config['optimizer']['ortho'].get('eps', 1e-30)}")
        print(f"      - Rescale gradients: {config['optimizer']['ortho'].get('rescale', True)}")
    elif optimizer_type == 'foreachmuon':
        print(f"      - Betas: {config['optimizer']['foreachmuon'].get('betas', (0.9, 0.99))}")
        print(f"      - Weight decay: {config['training']['weight_decay']}")
        print(f"      - Beta2 scale: {config['optimizer']['foreachmuon'].get('beta2_scale', 0.8)}")
        print(f"      - Nesterov: {config['optimizer']['foreachmuon'].get('nesterov', True)}")
    elif optimizer_type == 'muon_mix':
        print(f"      - Muon config:")
        print(f"        - Learning rate: {config['optimizer'].get('muon', {}).get('lr', 0.02)}")
        print(f"        - Weight decay: {config['training']['weight_decay']}")
        print(f"        - Momentum: {config['optimizer'].get('muon', {}).get('momentum', 0.95)}")
        print(f"        - Nesterov: {config['optimizer'].get('muon', {}).get('nesterov', True)}")
        print(f"        - NS steps: {config['optimizer'].get('muon', {}).get('ns_steps', 5)}")
        print(f"      - AdamW config:")
        print(f"        - Learning rate: {config['training']['learning_rate']}")
        print(f"        - Betas: {config['optimizer']['adamw'].get('betas', (0.9, 0.999))}")
        print(f"        - Weight decay: {config['training']['weight_decay']}")

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

    # Restore optimizer state if we loaded a checkpoint and reset_optimizer is False
    if loaded_checkpoint is not None and 'checkpoint' in config and not config['checkpoint'].get('reset_optimizer', True):
        print("\n[Main] Restoring optimizer state from checkpoint...")
        try:
            if 'optimizer_state_dicts' in loaded_checkpoint and loaded_checkpoint['optimizer_state_dicts']:
                # For multiple optimizers
                if len(loaded_checkpoint['optimizer_state_dicts']) == len(optimizers):
                    for i, opt in enumerate(optimizers):
                        opt.load_state_dict(loaded_checkpoint['optimizer_state_dicts'][i])
                    print(f"[Main] Successfully restored state for {len(optimizers)} optimizers")
                else:
                    print(f"[Main] Warning: Mismatch in optimizer count. Checkpoint has {len(loaded_checkpoint['optimizer_state_dicts'])}, current has {len(optimizers)}.")
                    print("[Main] Will only restore the primary optimizer state.")
                    optimizers[0].load_state_dict(loaded_checkpoint['optimizer_state_dicts'][0])
            else:
                print("[Main] No optimizer state found in checkpoint or state is empty.")
        except Exception as e:
            print(f"[Main] Error restoring optimizer state: {e}")
    else:
        print("\n[Main] Starting with fresh optimizer state.")

    print("\n[Main] Creating checkpoint directory (overwriting previous checkpoints)...")
    save_dir = Path('checkpoints')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Main] Checkpoint directory: {save_dir}")

    # Track best model info
    best_val_loss = float('inf')
    best_model_path = None
    latest_model_path = None

    def save_checkpoint(model, optimizers, epoch, global_step, val_loss, is_best=False):
        """Helper function to save checkpoints and manage storage"""
        nonlocal best_model_path, latest_model_path

        # Create checkpoint - adapted to handle multiple optimizers
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dicts': [opt.state_dict() for opt in optimizers],
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        }

        # Save as latest checkpoint locally
        if latest_model_path and os.path.exists(latest_model_path):
            os.remove(latest_model_path)  # Remove old latest checkpoint

        latest_model_path = save_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_model_path, _use_new_zipfile_serialization=False)

        # Create metadata
        meta = {
            'epoch': epoch,
            'global_step': global_step,
            'val_loss': val_loss,
            'timestamp': checkpoint['timestamp']
        }

        # Only log artifacts on rank 0
        if rank == 0 and wandb.run is not None:
            # Log artifacts based on save_model_frequency from config
            should_log_artifact = (global_step % config['training']['save_model_frequency'] == 0)

            if should_log_artifact:
                # Create and log latest artifact
                latest_artifact = wandb.Artifact(
                    name=f"{wandb.run.name}-latest",
                    type="model",
                    metadata=meta
                )
                latest_artifact.add_file(str(latest_model_path))
                wandb.log_artifact(latest_artifact, aliases=["latest"])

            # If this is the best model, save it separately
            if is_best:
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)  # Remove old best checkpoint

                best_model_path = save_dir / "best_model.pt"
                torch.save(checkpoint, best_model_path, _use_new_zipfile_serialization=False)

                # Always log best model artifact since it's important
                best_artifact = wandb.Artifact(
                    name=f"{wandb.run.name}-best",
                    type="model",
                    metadata=meta
                )
                best_artifact.add_file(str(best_model_path))
                wandb.log_artifact(best_artifact, aliases=["best"])

                # Log best model metrics to wandb
                wandb.run.summary.update({
                    "best_val_loss": val_loss,
                    "best_model_step": global_step,
                    "best_model_epoch": epoch,
                    "best_model_timestamp": checkpoint['timestamp']
                })
        else:
            # For non-rank-0 processes, just save the model files without wandb
            if is_best:
                best_model_path = save_dir / "best_model.pt"
                torch.save(checkpoint, best_model_path, _use_new_zipfile_serialization=False)

    NUM_EPOCHS = config['training']['num_epochs']
    validation_frequency = config['training']['validation_frequency']
    logging_frequency = config['training']['logging_frequency']
    save_frequency = config['training']['save_frequency']
    greedy_decode_frequency = config['training']['greedy_decode_frequency']
    
    # Initialize gradient scaler for mixed precision training
    scaler = None
    precision = config['training'].get('precision', 'fp32')
    use_amp = (precision in ['fp16', 'bf16']) and torch.cuda.is_available()
    
    # Only FP16 needs gradient scaling (BF16 has same dynamic range as FP32)
    if precision == 'fp16' and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("[Main] Initialized GradScaler for FP16 mixed precision training")
    
    start_time = time.time()
    
    # Helper for validation
    def validate(model, loader, criterion, tokenizer, device, block_ir=False, block_nmr=False):
        model.eval()
        total_loss = 0.0
        total_batches = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for target_tokens, ir_data, nmr_tokens, _ in loader:
                if block_ir:
                    ir_data = None
                if block_nmr:
                    nmr_tokens = None

                target_tokens = target_tokens.to(device)
                if ir_data is not None:
                    ir_data = ir_data.to(device)
                if nmr_tokens is not None:
                    nmr_tokens = nmr_tokens.to(device)

                T = target_tokens.size(1)
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=target_tokens.device), 1)
                
                # Uniformly sample the number of loops for validation if loop_range is set
                loop_range = config['model'].get('loop_range', None)
                num_loops = None
                if loop_range and model.training:
                    min_loops, max_loops = loop_range
                    num_loops = torch.randint(min_loops, max_loops + 1, (1,)).item()
                
                # Use mixed precision for validation too if enabled
                precision = config['training'].get('precision', 'fp32')
                use_amp = (precision in ['fp16', 'bf16']) and torch.cuda.is_available()
                if use_amp:
                    amp_dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        logits = model(
                            nmr_tokens=nmr_tokens,
                            ir_data=ir_data,
                            target_seq=target_tokens[:, :-1],
                            target_mask=mask[:-1, :-1],
                            num_loops=num_loops
                        )
                        loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens[:, 1:].reshape(-1))
                else:
                    logits = model(
                        nmr_tokens=nmr_tokens,
                        ir_data=ir_data,
                        target_seq=target_tokens[:, :-1],
                        target_mask=mask[:-1, :-1],
                        num_loops=num_loops
                    )
                    loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens[:, 1:].reshape(-1))
                
                pred_tokens = logits.argmax(dim=-1).cpu().tolist()
                tgt_tokens = target_tokens[:, 1:].cpu().tolist()
                
                for pred_seq in pred_tokens:
                    try:
                        sep_idx = pred_seq.index(tokenizer.sep_token_id)
                        pred_seq = pred_seq[:sep_idx]
                    except ValueError:
                        pass
                    decoded = tokenizer.decode(pred_seq).strip()
                    predictions.append(decoded)

                for tgt_seq in tgt_tokens:
                    try:
                        sep_idx = tgt_seq.index(tokenizer.sep_token_id)
                        tgt_seq = tgt_seq[:sep_idx]
                    except ValueError:
                        pass
                    decoded = tokenizer.decode(tgt_seq).strip()
                    targets.append(decoded)
                
                total_loss += loss.item()
                total_batches += 1
                torch.cuda.empty_cache()
        
        val_loss = total_loss / max(total_batches, 1)
        
        detailed_results = evaluate_predictions(predictions, targets)
        metrics = aggregate_metrics(detailed_results)
        combined_metrics = {
            'val_loss': val_loss,
            'valid_smiles_rate': metrics['valid_smiles'],
            'exact_match_rate': metrics['exact_match'],
            'exact_match_all_rate': metrics['exact_match_all'],
            'tanimoto_similarity': metrics['avg_tanimoto'],
            'mcs_ratio': metrics['avg_#mcs/#target'],
            'ecfp6_iou': metrics['avg_ecfp6_iou'],
            'predictions': predictions[:10],
            'targets': targets[:10],
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
            
            # Uniformly sample the number of loops for training if loop_range is set
            loop_range = config['model'].get('loop_range', None)
            num_loops = None
            if loop_range and model.training:
                min_loops, max_loops = loop_range
                num_loops = torch.randint(min_loops, max_loops + 1, (1,)).item()
            
            # Zero gradients for all optimizers
            for opt in optimizers:
                opt.zero_grad()
            
            # Forward and backward pass with automatic mixed precision for fp16/bf16
            if use_amp:
                amp_dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits = model(
                        nmr_tokens=nmr_tokens,
                        ir_data=ir_data,
                        target_seq=target_tokens[:, :-1],
                        target_mask=mask[:-1, :-1],
                        num_loops=num_loops
                    )
                    loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens[:, 1:].reshape(-1))
                
                # For FP16, we need to use the scaler for numerical stability
                if scaler is not None:
                    # Scale loss and do backward pass
                    scaler.scale(loss).backward()
                    
                    # Step optimizers with scaler
                    for opt in optimizers:
                        scaler.step(opt)
                    
                    # Update scaler
                    scaler.update()
                else:
                    # For BF16, we don't need scaling since it has the same dynamic range as FP32
                    loss.backward()
                    
                    # Step all optimizers
                    for opt in optimizers:
                        opt.step()
            else:
                # Regular forward and backward pass for fp32
                logits = model(
                    nmr_tokens=nmr_tokens,
                    ir_data=ir_data,
                    target_seq=target_tokens[:, :-1],
                    target_mask=mask[:-1, :-1],
                    num_loops=num_loops
                )
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens[:, 1:].reshape(-1))
                
                loss.backward()
                
                # Step all optimizers
                for opt in optimizers:
                    opt.step()
            
            # Step the scheduler regardless of precision mode
            scheduler.step()
            
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
                log_wandb({
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "progress": global_step / total_training_steps  # Add progress tracking
                }, step=global_step)

            # Periodic validation
            if global_step % validation_frequency == 0:
                print(f"\nRunning validation at step {global_step}...")
                val_metrics = validate(model, val_loader, criterion, tokenizer, device, block_ir, block_nmr)
                
                # Clean wandb cache periodically (every 5 validation steps)
                if (global_step // validation_frequency) % 5 == 0:
                    cleanup_wandb_cache()
                
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
                log_wandb({
                    "val_loss": val_metrics['val_loss'],
                    "val_valid_smiles": val_metrics['valid_smiles_rate'],
                    "val_exact_matches": val_metrics['exact_match_rate'],
                    "val_exact_matches_all": val_metrics['exact_match_all_rate'],
                    "val_tanimoto": val_metrics['tanimoto_similarity'],
                    "val_mcs_ratio": val_metrics['mcs_ratio'],
                    "val_ecfp6_iou": val_metrics['ecfp6_iou'],
                    "val_examples": examples_table,  # Log new table each time
                    "global_step": global_step
                }, step=global_step)
                
                print(f"[Val] Loss: {val_metrics['val_loss']:.4f}")

                # Save model periodically (outside validation check)
                if global_step % config['training']['save_model_frequency'] == 0:
                    print(f"\nSaving model checkpoint at step {global_step}...")
                    
                    # Get current validation metrics if available
                    current_val_loss = val_metrics['val_loss'] if 'val_metrics' in locals() else float('inf')
                    
                    # Save checkpoint and manage storage
                    is_best = current_val_loss < best_val_loss
                    if is_best:
                        best_val_loss = current_val_loss
                        print(f"New best validation loss: {best_val_loss:.4f}")
                    
                    save_checkpoint(
                        model=model,
                        optimizers=optimizers,  # Pass all optimizers
                        epoch=epoch,
                        global_step=global_step,
                        val_loss=current_val_loss,
                        is_best=is_best
                    )
                    
                    print(f"[Main] Model checkpoint saved at step {global_step}")

            # Periodic greedy decode evaluation
            if global_step % greedy_decode_frequency == 0:
                print(f"\nRunning greedy decode evaluation at step {global_step}...")
                greedy_metrics = evaluate_with_greedy_decode(
                    model=model,
                    test_loader=test_loader,
                    tokenizer=tokenizer,
                    device=device,
                    num_examples=100,
                    block_ir=block_ir,
                    block_nmr=block_nmr
                )
                
                # Create a new table for greedy decode examples
                greedy_table = wandb.Table(columns=columns)
                
                # Log sample results
                for pred, tgt in zip(greedy_metrics['predictions'], greedy_metrics['targets']):
                    # Calculate metrics for this pair
                    pair_results = evaluate_predictions([pred], [tgt])[0]
                    pair_results = evaluate_predictions([pred], [tgt])[0]
                    greedy_table.add_data(
                        global_step,
                        pred,
                        tgt,
                        pair_results['exact_match'],
                        pair_results['tanimoto'],
                        pair_results['#mcs/#target'],
                        pair_results['ecfp6_iou']
                    )
                
                # Log metrics
                log_wandb({
                    "greedy_valid_smiles": greedy_metrics['valid_smiles'],
                    "greedy_exact_matches": greedy_metrics['exact_match'],
                    "greedy_exact_matches_all": greedy_metrics['exact_match_all'],
                    "greedy_tanimoto": greedy_metrics['avg_tanimoto'],
                    "greedy_mcs_ratio": greedy_metrics['avg_#mcs/#target'],
                    "greedy_ecfp6_iou": greedy_metrics['avg_ecfp6_iou'],
                    "greedy_examples": greedy_table,
                    "global_step": global_step
                }, step=global_step)
                
                print(f"[Greedy] Valid SMILES: {greedy_metrics['valid_smiles']:.2%}")
                print(f"[Greedy] Exact matches: {greedy_metrics['exact_match']:.2%}")
                print(f"[Greedy] Tanimoto similarity: {greedy_metrics['avg_tanimoto']:.4f}")

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} completed | Average Loss: {avg_epoch_loss:.4f}")

    # Final test set evaluation
    print("\n[Main] Evaluating on test set...")
    final_test_loss = validate(model, test_loader, criterion, tokenizer, device, block_ir, block_nmr)
    log_wandb({"test_loss": final_test_loss['val_loss']}, step=global_step)
    print(f"[Test] Loss: {final_test_loss['val_loss']:.4f}")

    # Final greedy decode evaluation
    print("\n[Main] Running final greedy decode evaluation...")
    final_greedy_metrics = evaluate_with_greedy_decode(
        model=model,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        block_ir=block_ir,
        block_nmr=block_nmr
    )
    log_wandb({
        "final_greedy_valid_smiles": final_greedy_metrics['valid_smiles'],
        "final_greedy_exact_matches": final_greedy_metrics['exact_match'],
        "final_greedy_tanimoto": final_greedy_metrics['avg_tanimoto'],
        "final_greedy_mcs_ratio": final_greedy_metrics['avg_#mcs/#target'],
        "final_greedy_ecfp6_iou": final_greedy_metrics['avg_ecfp6_iou']
    }, step=global_step)
    print(f"[Final Greedy] Valid SMILES: {final_greedy_metrics['valid_smiles']:.2%}")
    print(f"[Final Greedy] Exact matches: {final_greedy_metrics['exact_match']:.2%}")
    print(f"[Final Greedy] Tanimoto similarity: {final_greedy_metrics['avg_tanimoto']:.4f}")

    # Save final model if requested
    if config['training'].get('save_local', False):
        save_checkpoint(
            model=model,
            optimizers=optimizers,  # Pass all optimizers
            epoch=NUM_EPOCHS,
            global_step=global_step,
            val_loss=final_test_loss['val_loss'],
            is_best=final_test_loss['val_loss'] < best_val_loss
        )
        print(f"Final checkpoint saved in {save_dir}")

    print("[Main] Training script completed.")
    if rank == 0 and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
