# train_rl_grpo.py
"""
Script for post-training a pretrained MultiModalToSMILESModel using GRPO RL.
Rewards exact matches between generated SMILES and target SMILES.
"""

import os
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import datetime
import time
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import contextlib
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

# Import from the base training script
from train_autoregressive import (
    MultiModalToSMILESModel,
    SmilesTokenizer,
    tokenizer,
    load_config,
    evaluate_with_greedy_decode,
    SpectralSmilesDataset,
    collate_fn,
    evaluate_predictions,
    load_vocabularies,
    create_data_loaders,
    greedy_decode
)

class GRPO:
    """
    Grouped Reward-weighted Policy Optimization for fine-tuning with RL.
    Modified to work with MultiModalToSMILESModel for SMILES generation.
    """
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        group_size=8,
        micro_group_size=2,
        batch_size=1,
        max_iterations=1000,    
        train_loader=None,
        val_loader=None,
        log_wandb=False,
        dtype=None,
        lr=5e-6,
        weight_decay=0.0,
        beta=0.0,
        epsilon=0.1,
        use_exact_match_reward=True,
        use_tanimoto_reward=False,
        use_ecfp6_reward=False,
        use_valid_smiles_reward=False,
        use_mcs_ratio_reward=False,
        use_cot_reward=False,
        temperature=1,
        device=None,
        use_kl=True,
        optimizer_type='adamw',
        log_frequency=1,
        validation_frequency=10
    ):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader_iter = iter(self.train_loader)
        self.group_size = group_size
        self.micro_group_size = micro_group_size   
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.dtype = dtype if dtype is not None else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.beta = beta
        self.epsilon = epsilon
        self.log_frequency = log_frequency
        
        self.use_kl = use_kl
        
        # Initialize optimizer based on optimizer_type parameter
        if optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'muon_mix':
            # Separate parameters: matrix_params for 2D or higher, vector_params for others
            matrix_params = [p for p in self.model.parameters() if p.ndim >= 2]
            vector_params = [p for p in self.model.parameters() if p.ndim < 2]
            
            from muon import Muon
            muon_opt = Muon(matrix_params, lr=lr, weight_decay=weight_decay, momentum=0.95, nesterov=True, ns_steps=5) if matrix_params else None
            adamw_opt = torch.optim.AdamW(vector_params, lr=lr, weight_decay=weight_decay) if vector_params else None
            
            # Define a simple composite optimizer to update both optimizers
            class CompositeOptimizer:
                def __init__(self, optimizers):
                    # Filter out any None optimizers
                    self.optimizers = [opt for opt in optimizers if opt is not None]
                def step(self):
                    for opt in self.optimizers:
                        opt.step()
                def zero_grad(self):
                    for opt in self.optimizers:
                        opt.zero_grad()
            
            self.optimizer = CompositeOptimizer([muon_opt, adamw_opt])
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
        
        # Active reward configuration
        self.use_exact_match_reward = use_exact_match_reward
        self.use_tanimoto_reward = use_tanimoto_reward
        self.use_ecfp6_reward = use_ecfp6_reward
        self.use_valid_smiles_reward = use_valid_smiles_reward
        self.use_mcs_ratio_reward = use_mcs_ratio_reward
        self.use_cot_reward = use_cot_reward
        
        # Ensure at least one reward is active
        assert (
            self.use_exact_match_reward
            or self.use_tanimoto_reward
            or self.use_ecfp6_reward
            or self.use_valid_smiles_reward
            or self.use_mcs_ratio_reward
            or self.use_cot_reward
        ), "At least one reward function must be enabled"
        
        # Print reward configuration
        print(f"[GRPO] 🎯 Active rewards:")
        print(f"  - Exact match reward: {'✓' if self.use_exact_match_reward else '✗'}")
        print(f"  - Tanimoto reward: {'✓' if self.use_tanimoto_reward else '✗'}")
        print(f"  - ECFP6 reward: {'✓' if self.use_ecfp6_reward else '✗'}")
        print(f"  - Valid SMILES reward: {'✓' if self.use_valid_smiles_reward else '✗'}")
        print(f"  - MCS Ratio reward: {'✓' if self.use_mcs_ratio_reward else '✗'}")
        print(f"  - CoT structure reward: {'✓' if self.use_cot_reward else '✗'}")

        # For the MultiModal model, we're not using LoRA adapters
        self.using_lora = False

        self.distributed = False
        self.log_wandb = log_wandb
        if self.log_wandb and wandb.run is None:
            wandb.init(project="SMILES-GRPO")

        self.metrics = defaultdict(list)

        # Move models to device and dtype
        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
        
        # Watch model with wandb to track gradients and parameters
        if self.log_wandb and wandb.run is not None:
            wandb.watch(
                self.model,
                log="gradients",  # Track gradients
                log_freq=self.log_frequency * 10,  # Log less frequently than metrics to avoid overhead
                log_graph=True  # Log model graph
            )
        
        # Print some info about models
        print(f"[GRPO] Model device: {next(self.model.parameters()).device}")
        print(f"[GRPO] Model training: {self.model.training}")
        
        # Set up validation steps
        self.validation_frequency = validation_frequency

    def get_per_token_logps(self, model, target_seq, nmr_tokens, ir_data) -> Tensor:
        """
        Compute log probabilities for each token in the target sequence.
        Adapted to work with MultiModalToSMILESModel.
        """
        # Get logits from the model; rely on the decoder to handle attention mask automatically
        logits = model(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            target_seq=target_seq[:, :-1]
        )
        
        # Get the target tokens (shifted right)
        target_tokens = target_seq[:, 1:]
        
        # Compute log probabilities
        logps = F.log_softmax(logits, dim=-1)
        
        # Gather the log probabilities of the target tokens
        per_token_logps = torch.gather(logps, -1, target_tokens.unsqueeze(-1)).squeeze(-1)
        
        return per_token_logps

    def compute_loss(self, target_seq, nmr_tokens, ir_data, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask) -> Tensor:
        """
        Compute the GRPO loss.
        """
        # Get log probabilities from current policy (decoder will handle attention mask internally)
        policy_log_probs = self.get_per_token_logps(
            self.model,
            target_seq=target_seq,
            nmr_tokens=nmr_tokens,
            ir_data=ir_data
        )
        
        # Get log probabilities from reference policy (using decoder's internal mask)
        if self.ref_model is not None:
            ref_policy_log_probs = self.get_per_token_logps(
                self.ref_model,
                target_seq=target_seq,
                nmr_tokens=nmr_tokens,
                ir_data=ir_data
            )
        else:
            ref_policy_log_probs = old_policy_log_probs

        # Ensure reward is properly shaped for broadcasting
        if reward.dim() == 1:
            reward = reward.reshape(-1, 1)
        
        # Calculate advantage - normalize rewards for stable training
        advantage = (reward - mean_rewards) / (std_rewards + 1e-6)
        advantage = advantage.reshape(-1, 1)

        # KL divergence calculation
        log_ratios = ref_policy_log_probs - policy_log_probs
        kld = torch.exp(log_ratios) - log_ratios - 1

        # Policy ratio for clipped objective
        policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs.detach())

        # Calculate GRPO loss (clipped surrogate objective)
        loss1 = policy_ratio * advantage
        loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(loss1, loss2)  # Negative sign is correct
        
        # Ensure loss and loss_mask have the same shape before multiplication
        if loss.shape[1] != loss_mask.shape[1]:
            # Truncate the loss tensor to match the loss_mask
            loss = loss[:, :loss_mask.shape[1]]
        
        # Apply mask and average
        loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        
        # Ensure KLD has the same shape as loss_mask
        if kld.shape[1] != loss_mask.shape[1]:
            kld = kld[:, :loss_mask.shape[1]]
            
        kld = (kld * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        
        # Optionally add KL divergence penalty if use_kl is True
        if self.use_kl:
            loss += kld * self.beta
        
        # Log KL divergence
        if self.log_wandb:
            for _kd in kld:
                self.metrics["kld"].append(_kd.mean().item())
                
        return loss.mean()

    def sample_batch(self):
        """
        Sample a batch of data and generate outputs for training.
        Adapted for the MultiModalToSMILESModel's data format.
        Uses temperature-based sampling to increase exploration.
        """
        print(f"\n[GRPO] ⏱️ Starting batch sampling at {datetime.datetime.now().strftime('%H:%M:%S')}")
        start_time = time.time()
        
        # Reset the iterator if needed
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
            
        # Unpack the batch
        target_tokens, ir_data, nmr_tokens, _ = batch
        print(f"[GRPO] 📊 Batch size: {target_tokens.size(0)}, Target seq length: {target_tokens.size(1)}")
        
        # Move data to device
        print(f"[GRPO] 🔄 Moving data to device: {self.device}")
        target_tokens = target_tokens.to(self.device)
        if ir_data is not None:
            ir_data = ir_data.to(self.device)
        if nmr_tokens is not None:
            nmr_tokens = nmr_tokens.to(self.device)
            
        # We'll replicate each sample group_size times for exploration
        batch_size = target_tokens.size(0)
        expanded_batch_size = batch_size * self.group_size
        print(f"[GRPO] 📈 Expanding batch: {batch_size} samples x {self.group_size} groups = {expanded_batch_size} total samples")
        
        # Expand the inputs
        expanded_target_tokens = target_tokens.repeat(self.group_size, 1)
        expanded_ir_data = ir_data.repeat(self.group_size, 1) if ir_data is not None else None
        expanded_nmr_tokens = nmr_tokens.repeat(self.group_size, 1) if nmr_tokens is not None else None
        
        # Store original targets for reward calculation
        original_targets = []
        for tgt in target_tokens:
            # Get target without special tokens
            try:
                eos_idx = tgt.tolist().index(self.tokenizer.sep_token_id)
                tgt = tgt[:eos_idx]  # Exclude EOS token
            except ValueError:
                pass
            decoded = self.tokenizer.decode(tgt[1:])  # Skip BOS token
            original_targets.append(decoded)
        
        # Generate responses with the model
        print(f"[GRPO] 🧪 Starting SMILES generation for {expanded_batch_size} samples with temperature {self.temperature}...")
        gen_start = time.time()
        with torch.no_grad():
            # Set the model to eval mode for generation
            self.model.eval()
            
            # Use our modified greedy_decode function with sampling for generation
            generated_smiles = greedy_decode(
                model=self.model,
                nmr_tokens=expanded_nmr_tokens,
                ir_data=expanded_ir_data,
                tokenizer=self.tokenizer,
                max_len=self.model.decoder.max_seq_length,
                device=self.device,
                temperature=self.temperature,
                sample=True  # Enable sampling
            )
            
            # Set model back to train mode
            self.model.train()
        
        gen_end = time.time()
        gen_time = gen_end - gen_start
        avg_gen_time = gen_time / expanded_batch_size if expanded_batch_size > 0 else 0
        print(f"[GRPO] ⌛ Generation completed in {gen_time:.2f}s ({avg_gen_time:.4f}s per sample)")
        
        # Compute rewards based on generated outputs
        print(f"[GRPO] 💯 Calculating rewards...")
        reward_start = time.time()
        rewards = self.compute_rewards(original_targets, generated_smiles)
        reward_end = time.time()
        print(f"[GRPO] ⌛ Reward calculation completed in {reward_end - reward_start:.2f}s")
        
        # Print some examples of generated SMILES and rewards
        print(f"\n[GRPO] 📝 Sample generations (first 3):")
        for i in range(min(3, len(generated_smiles))):
            target_idx = i % batch_size
            print(f"  Target:     {original_targets[target_idx]}")
            print(f"  Generated:  {generated_smiles[i]}")
            print(f"  Reward:     {rewards[i]:.4f}")
            print()
        
        # Re-encode the generated outputs to get targets for training...
        print(f"[GRPO] 🔄 Re-encoding generated SMILES for training...")
        encode_start = time.time()
        encoded_outputs = []
        for smiles in generated_smiles:
            tokens = self.tokenizer.encode(
                smiles,
                add_special_tokens=True,
                max_length=self.model.decoder.max_seq_length,
                truncation=True
            )
            encoded_outputs.append(tokens)

        # Pad the encoded outputs
        max_len = max(len(seq) for seq in encoded_outputs)
        padded_outputs = []
        for seq in encoded_outputs:
            pad_amount = max_len - len(seq)
            if pad_amount > 0:
                padded_seq = seq + [self.tokenizer.pad_token_id] * pad_amount
            else:
                padded_seq = seq
            padded_outputs.append(padded_seq)

        output_tokens = torch.tensor(padded_outputs, dtype=torch.long, device=self.device)
        encode_end = time.time()
        print(f"[GRPO] ⌛ Re-encoding completed in {encode_end - encode_start:.2f}s")

        # Create loss mask (ignore padding tokens)
        print(f"[GRPO] 🎭 Creating loss mask...")
        loss_mask = torch.ones(output_tokens.size(0), output_tokens.size(1) - 1, 
                               dtype=torch.bool, device=self.device)

        for i, tgt in enumerate(output_tokens):
            try:
                eos_idx = tgt.tolist().index(self.tokenizer.sep_token_id)
                if eos_idx < output_tokens.size(1) - 1:
                    loss_mask[i, eos_idx:] = False
            except ValueError:
                pass

            pad_mask = tgt[1:] != self.tokenizer.pad_token_id
            loss_mask[i] = loss_mask[i] & pad_mask
            
        total_time = time.time() - start_time
        print(f"[GRPO] ⏱️ Total batch preparation time: {total_time:.2f}s")
        
        return expanded_target_tokens, expanded_ir_data, expanded_nmr_tokens, output_tokens, torch.tensor(rewards, device=self.device), loss_mask

    def compute_rewards(self, original_targets, generated_smiles) -> list:
        """
        Compute rewards for generated outputs.
        """
        print(f"[GRPO] 🎯 Computing rewards for {len(generated_smiles)} generated SMILES...")
        rewards = []
        batch_size = len(original_targets)
        
        # Initialize counters for reporting
        exact_matches = 0
        valid_count = 0
        invalid_count = 0
        total_tanimoto = 0.0
        total_ecfp6 = 0.0
        total_mcs_ratio = 0.0
        total_cot = 0.0
        
        # Match generated SMILES with their targets based on group position
        for i, generated in enumerate(generated_smiles):
            target_idx = i % batch_size  # Cycle through targets based on group position
            target = original_targets[target_idx]

            # Extract potential answer from CoT formatted string
            parsed_answer = parse_cot_answer(generated)
            gen_for_eval = parsed_answer if parsed_answer else generated

            # Initialize reward for this sample
            reward = 0.0

            # CoT structure reward
            cot_score = 0.0
            if self.use_cot_reward:
                cot_score = cot_structure_reward(generated)
                reward += cot_score
                total_cot += cot_score

            # Apply exact match reward if enabled
            exact_match_score = 0.0
            if self.use_exact_match_reward:
                exact_match_score = exact_match_reward(target, gen_for_eval)
                reward += exact_match_score
                if exact_match_score > 0:
                    exact_matches += 1
            
            # Apply Tanimoto similarity reward if enabled
            tanimoto_score = 0.0
            if self.use_tanimoto_reward:
                tanimoto_score = tanimoto_reward(target, gen_for_eval)
                reward += tanimoto_score
                total_tanimoto += tanimoto_score
            
            # Apply ECFP6 reward if enabled
            ecfp6_score = 0.0
            if self.use_ecfp6_reward:
                ecfp6_score = ecfp6_reward(target, gen_for_eval)
                reward += ecfp6_score
                total_ecfp6 += ecfp6_score
            
            # Apply MCS ratio reward if enabled
            mcs_ratio_score = 0.0
            if self.use_mcs_ratio_reward:
                mcs_ratio_score = mcs_ratio_reward(target, gen_for_eval)
                reward += mcs_ratio_score
                total_mcs_ratio += mcs_ratio_score
            
            # Check validity and apply valid SMILES reward if enabled
            mol = None
            valid_smiles_score = 0.0
            try:
                mol = Chem.MolFromSmiles(gen_for_eval)
                if mol is not None:
                    valid_count += 1
                    valid_smiles_score = 1.0
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1
            
            if self.use_valid_smiles_reward:
                reward += valid_smiles_score
            
            rewards.append(reward)
            
            # Log rewards for tracking (log for all samples, not just a subset)
            if self.log_wandb:
                if self.use_exact_match_reward:
                    self.metrics["exact_match_rewards"].append(exact_match_score)
                if self.use_tanimoto_reward:
                    self.metrics["tanimoto_rewards"].append(tanimoto_score)
                if self.use_ecfp6_reward:
                    self.metrics["ecfp6_rewards"].append(ecfp6_score)
                if self.use_valid_smiles_reward:
                    self.metrics["valid_smiles_rewards"].append(valid_smiles_score)
                if self.use_mcs_ratio_reward:
                    self.metrics["mcs_ratio_rewards"].append(mcs_ratio_score)
                if self.use_cot_reward:
                    self.metrics["cot_rewards"].append(cot_score)
                
                if mol is not None:
                    self.metrics["valid_molecule"].append(1.0)
                else:
                    self.metrics["valid_molecule"].append(0.0)
        
        # Print reward statistics
        total = len(generated_smiles)
        valid_pct = valid_count / total * 100
        exact_pct = exact_matches / total * 100
        avg_tanimoto = total_tanimoto / total if self.use_tanimoto_reward else 0.0
        avg_ecfp6 = total_ecfp6 / total if self.use_ecfp6_reward else 0.0
        avg_mcs_ratio = total_mcs_ratio / total if self.use_mcs_ratio_reward else 0.0
        avg_cot = total_cot / total if self.use_cot_reward else 0.0
        
        print(f"[GRPO] 📊 Reward stats:")
        print(f"  - Valid SMILES: {valid_count}/{total} ({valid_pct:.2f}%)")
        if self.use_exact_match_reward:
            print(f"  - Exact matches: {exact_matches}/{total} ({exact_pct:.2f}%)")
        if self.use_tanimoto_reward:
            print(f"  - Avg Tanimoto similarity: {avg_tanimoto:.4f}")
        if self.use_ecfp6_reward:
            print(f"  - Avg ECFP6 IoU: {avg_ecfp6:.4f}")
        if self.use_mcs_ratio_reward:
            print(f"  - Avg MCS ratio: {avg_mcs_ratio:.4f}")
        if self.use_cot_reward:
            print(f"  - CoT format rate: {avg_cot:.4f}")
        
        print(f"[DEBUG] Exact match rate: {exact_matches}/{total} = {exact_matches/total:.4f}")
        
        # Add normalization for stability
        rewards = torch.tensor(rewards, device=self.device)
        
        # Log overall statistics
        if self.log_wandb:
            self.metrics["avg_reward"].append(rewards.mean().item())
            if self.use_exact_match_reward:
                self.metrics["exact_match_rate"].append(exact_pct / 100.0)
            self.metrics["valid_smiles_rate"].append(valid_pct / 100.0)
            if self.use_tanimoto_reward:
                self.metrics["avg_tanimoto"].append(avg_tanimoto)
            if self.use_ecfp6_reward:
                self.metrics["avg_ecfp6"].append(avg_ecfp6)
            if self.use_mcs_ratio_reward:
                self.metrics["avg_mcs_ratio"].append(avg_mcs_ratio)
            if self.use_cot_reward:
                self.metrics["avg_cot"].append(avg_cot)
        
        return rewards.tolist()

    def log_metrics(self, step):
        """Log metrics to wandb"""
        if self.log_wandb:
            metrics = {}
            # Prepare metrics to log - use ALL collected metrics instead of just the last batch
            for k, v in self.metrics.items():
                if v:  # If not empty
                    metrics[f"train/{k}"] = np.mean(v)
            
            # Add current step
            metrics["step"] = step
            
            # Count how many samples were used for these metrics
            metrics["train/samples_in_metrics"] = sum(len(v) for v in self.metrics.values()) / max(1, len(self.metrics))
            
            # Log to wandb
            wandb.log(metrics)
            
            # Print metrics summary
            print(f"[GRPO] 📊 Logging metrics for {int(metrics['train/samples_in_metrics'])} samples:")
            for k, v in metrics.items():
                if k != "step" and k != "train/samples_in_metrics":
                    print(f"  - {k}: {v:.4f}")
            
            # Clear metrics
            for k in self.metrics:
                self.metrics[k] = []

    def train(self, num_iterations=1000):
        """
        Train the model with GRPO.
        """
        print(f"\n[GRPO] 🚀 Starting training for {num_iterations} iterations")
        print(f"[GRPO] 📊 Logging metrics every {self.log_frequency} iterations")
        print(f"[GRPO] 🔍 Running validation every {self.validation_frequency} iterations")
        
        start_time = time.perf_counter()
        
        for iteration in range(num_iterations):
            iter_start = time.perf_counter()
            print(f"\n{'='*80}")
            print(f"[GRPO] 🔄 Iteration {iteration+1}/{num_iterations} - Started at {datetime.datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*80}")
            
            # Sample batch and generate outputs
            sample_start = time.perf_counter()
            target_tokens, ir_data, nmr_tokens, generated_tokens, rewards, loss_mask = self.sample_batch()
            sample_time = time.perf_counter() - sample_start
            
            # Calculate mean and std of rewards for advantage normalization
            mean_rewards = rewards.mean().item()
            std_rewards = rewards.std().item()
            print(f"[GRPO] 📊 Rewards: mean={mean_rewards:.4f}, std={std_rewards:.4f}, max={rewards.max().item():.4f}")
            
            # Skip update if rewards are all the same (no learning signal)
            if std_rewards < 1e-6:
                print(f"[GRPO] ⚠️ Skipping update - rewards have no variance")
                continue

            # Get log probabilities from current policy (decoder will handle attention mask internally)
            print(f"[GRPO] 🎭 Computing old policy log probabilities...")
            logprob_start = time.perf_counter()
            with torch.no_grad():
                old_policy_log_probs = self.get_per_token_logps(
                    self.model,
                    target_seq=generated_tokens,
                    nmr_tokens=nmr_tokens,
                    ir_data=ir_data
                )
            logprob_time = time.perf_counter() - logprob_start
            print(f"[GRPO] ⌛ Log probabilities computed in {logprob_time:.2f}s")
            
            # Reshape into groups
            batch_size = target_tokens.size(0) // self.group_size
            print(f"[GRPO] 📊 Processing {batch_size} groups of {self.group_size} samples")
            
            def reshape_into_groups(tensor):
                """Helper to reshape tensors into groups"""
                if tensor is None:
                    return None
                return tensor.view(batch_size, self.group_size, *tensor.size()[1:])
            
            # Group the inputs
            print(f"[GRPO] 🔄 Reshaping tensors into groups...")
            group_start = time.perf_counter()
            grouped_target_tokens = reshape_into_groups(target_tokens)
            grouped_ir_data = reshape_into_groups(ir_data)
            grouped_nmr_tokens = reshape_into_groups(nmr_tokens)
            grouped_generated_tokens = reshape_into_groups(generated_tokens)
            grouped_old_policy_log_probs = reshape_into_groups(old_policy_log_probs)
            grouped_rewards = reshape_into_groups(rewards)
            grouped_loss_mask = reshape_into_groups(loss_mask)
            group_time = time.perf_counter() - group_start
            print(f"[GRPO] ⌛ Group reshaping completed in {group_time:.2f}s")
            
            # Process each group
            print(f"[GRPO] 🧮 Processing groups with micro-batches...")
            update_start = time.perf_counter()
            total_loss = 0.0
            group_count = 0
            
            for group_idx in range(batch_size):
                if group_idx % max(1, batch_size // 5) == 0:  # Print progress every ~20% of groups
                    print(f"[GRPO] ⏳ Processing group {group_idx+1}/{batch_size}")
                
                # Get the group
                target_tokens_g = grouped_target_tokens[group_idx]
                ir_data_g = grouped_ir_data[group_idx] if grouped_ir_data is not None else None
                nmr_tokens_g = grouped_nmr_tokens[group_idx] if grouped_nmr_tokens is not None else None
                generated_tokens_g = grouped_generated_tokens[group_idx]
                old_policy_log_probs_g = grouped_old_policy_log_probs[group_idx]
                rewards_g = grouped_rewards[group_idx]
                loss_mask_g = grouped_loss_mask[group_idx]
                
                # Process micro-batches within the group
                micro_batch_size = self.micro_group_size
                group_loss = 0.0
                micro_batch_count = 0
                num_micro_batches = (self.group_size + micro_batch_size - 1) // micro_batch_size
                
                for mb_idx in range(0, self.group_size, micro_batch_size):
                    mb_start = time.perf_counter()
                    mb_end = min(mb_idx + micro_batch_size, self.group_size)
                    mb_size = mb_end - mb_idx
                    
                    # Skip if this would produce an empty batch
                    if mb_size <= 0:
                        continue
                    
                    # Get the micro-batch
                    target_tokens_mb = target_tokens_g[mb_idx:mb_end]
                    ir_data_mb = ir_data_g[mb_idx:mb_end] if ir_data_g is not None else None
                    nmr_tokens_mb = nmr_tokens_g[mb_idx:mb_end] if nmr_tokens_g is not None else None
                    generated_tokens_mb = generated_tokens_g[mb_idx:mb_end]
                    old_policy_log_probs_mb = old_policy_log_probs_g[mb_idx:mb_end]
                    rewards_mb = rewards_g[mb_idx:mb_end]
                    loss_mask_mb = loss_mask_g[mb_idx:mb_end]
                    
                    # Compute loss for this micro-batch
                    loss = self.compute_loss(
                        target_seq=generated_tokens_mb,
                        nmr_tokens=nmr_tokens_mb,
                        ir_data=ir_data_mb,
                        old_policy_log_probs=old_policy_log_probs_mb,
                        reward=rewards_mb,
                        mean_rewards=mean_rewards,
                        std_rewards=std_rewards,
                        loss_mask=loss_mask_mb
                    )
                    
                    # Accumulate loss and update
                    loss = loss / num_micro_batches  # Scale loss by number of micro-batches
                    loss.backward()
                    
                    # Record loss
                    group_loss += loss.item() * num_micro_batches  # Rescale for reporting
                    micro_batch_count += 1
                    
                    # Log micro-batch timing
                    mb_time = time.perf_counter() - mb_start
                    if group_idx == 0 or group_idx == batch_size - 1:  # Only log for first and last group
                        print(f"[GRPO] ⏱️ Micro-batch {mb_idx//micro_batch_size + 1}/{num_micro_batches} (size {mb_size}) processed in {mb_time:.2f}s")
                
                # Update parameters for this group if we processed any micro-batches
                if micro_batch_count > 0:
                    # Track total loss for reporting
                    total_loss += group_loss
                    group_count += 1
                    
                    # Log group loss
                    if self.log_wandb:
                        self.metrics["policy_loss"].append(group_loss)
                    
                    # Update model parameters after each group (like in original implementation)
                    # Add gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if group_idx % max(1, batch_size // 5) == 0:  # Print progress every ~20% of groups
                        print(f"[GRPO] 🔄 Updated model after group {group_idx+1}/{batch_size}")
            
            # Calculate average loss
            avg_loss = total_loss / max(1, group_count)
            update_time = time.perf_counter() - update_start
            print(f"[GRPO] 📊 Average loss: {avg_loss:.4f} (calculated over {group_count} groups)")
            print(f"[GRPO] ⏱️ Total update time: {update_time:.2f}s")
            
            # Log metrics
            if self.log_wandb and (iteration + 1) % self.log_frequency == 0:
                self.log_metrics(iteration)
                
            # Run validation periodically
            if self.val_loader is not None and (iteration + 1) % self.validation_frequency == 0:
                print(f"\n[GRPO] 🔍 Running validation at iteration {iteration + 1}")
                metrics = self.validate()
            
            # Calculate total iteration time
            iter_time = time.perf_counter() - iter_start
            print(f"[GRPO] ⏱️ Iteration {iteration+1} completed in {iter_time:.2f}s")
            print(f"[GRPO] ⏱️ Total training time so far: {time.perf_counter() - start_time:.2f}s")
            
            # Print estimated time remaining
            avg_iter_time = (time.perf_counter() - start_time) / (iteration + 1)
            remaining_iters = num_iterations - (iteration + 1)
            est_remaining_time = avg_iter_time * remaining_iters
            print(f"[GRPO] ⏱️ Estimated time remaining: {est_remaining_time:.2f}s ({est_remaining_time/60:.2f}m)")

    def validate(self):
        """Run validation and log metrics"""
        print(f"\n[GRPO] 🔍 Starting validation at {datetime.datetime.now().strftime('%H:%M:%S')}")
        val_start = time.perf_counter()
        self.model.eval()
        
        # Define minimum number of samples to validate
        min_samples = 50
        
        with torch.no_grad():
            print(f"[GRPO] 📂 Loading validation data...")
            batch_start = time.perf_counter()
            
            # Initialize lists to collect data across batches
            all_targets = []
            all_predictions = []
            all_detailed_results = []
            
            # Create iterator for val_loader
            val_iter = iter(self.val_loader)
            samples_processed = 0
            
            # Process batches until we reach min_samples
            while samples_processed < min_samples:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    # If we've gone through the whole dataset, restart the iterator
                    val_iter = iter(self.val_loader)
                    val_batch = next(val_iter)
                
                target_tokens, ir_data, nmr_tokens, _ = val_batch
                
                batch_size = target_tokens.size(0)
                samples_processed += batch_size
                print(f"[GRPO] 📊 Processing validation batch: {batch_size} samples (Total: {samples_processed})")
                
                # Move to device
                target_tokens = target_tokens.to(self.device)
                if ir_data is not None:
                    ir_data = ir_data.to(self.device)
                if nmr_tokens is not None:
                    nmr_tokens = nmr_tokens.to(self.device)
                
                # Generate with greedy decoding
                predictions = greedy_decode(
                    model=self.model,
                    nmr_tokens=nmr_tokens,
                    ir_data=ir_data,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    temperature=0.0,
                    sample=False  # Use greedy decoding (not sampling) for evaluation
                )
                
                # Get targets
                targets = []
                for tgt in target_tokens:
                    try:
                        eos_idx = tgt.tolist().index(self.tokenizer.sep_token_id)
                        tgt = tgt[:eos_idx]
                    except ValueError:
                        pass
                    decoded = self.tokenizer.decode(tgt[1:])
                    targets.append(decoded)
                
                # Evaluate this batch of predictions
                batch_results = evaluate_predictions(predictions, targets)
                
                # Collect results
                all_targets.extend(targets)
                all_predictions.extend(predictions)
                all_detailed_results.extend(batch_results)
            
            batch_time = time.perf_counter() - batch_start
            print(f"[GRPO] ⏱️ Processed {samples_processed} validation samples in {batch_time:.2f}s")
            
            # Calculate metrics across all processed samples
            print(f"[GRPO] 📈 Calculating metrics for {len(all_detailed_results)} samples...")
            metrics_start = time.perf_counter()
            metrics = {
                "exact_match": np.mean([r["exact_match"] for r in all_detailed_results]),
                "valid_smiles": np.mean([r["valid"] for r in all_detailed_results]),
                "tanimoto": np.mean([r["tanimoto"] for r in all_detailed_results]),
                "ecfp6_iou": np.mean([r["ecfp6_iou"] for r in all_detailed_results])
            }
            
            if self.log_wandb:
                wandb.log(metrics)
            metrics_time = time.perf_counter() - metrics_start
            print(f"[GRPO] ⏱️ Metrics calculated in {metrics_time:.2f}s")
            
            # Print results
            print(f"\n[GRPO] 📊 Validation results:")
            for k, v in metrics.items():
                print(f"  - {k}: {v:.4f}")
            
            # Print some examples
            print("\n[GRPO] 📝 Examples:")
            for i in range(min(3, len(all_predictions))):
                print(f"  Target:     {all_targets[i]}")
                print(f"  Prediction: {all_predictions[i]}")
                print(f"  Match:      {'✓' if all_detailed_results[i]['exact_match'] else '✗'}")
                print(f"  Similarity: {all_detailed_results[i]['tanimoto']:.4f}")
                print()
        
        # Set back to train mode
        self.model.train()
        
        total_val_time = time.perf_counter() - val_start
        print(f"[GRPO] ✅ Validation completed in {total_val_time:.2f}s (processed {len(all_detailed_results)} samples)")
        
        return metrics


# Define reward functions
def exact_match_reward(target, generated):
    """Reward function that gives 1.0 for exact matches, 0.0 otherwise using canonical SMILES representation"""
    try:
        # Remove spaces and strip both target and generated SMILES
        target_clean = target.replace(' ', '').strip()
        generated_clean = generated.replace(' ', '').strip()

        # Convert the cleaned SMILES into molecular objects
        mol_target = Chem.MolFromSmiles(target_clean)
        mol_generated = Chem.MolFromSmiles(generated_clean)

        # If either conversion fails, return 0.0 reward
        if mol_target is None or mol_generated is None:
            return 0.0

        # Generate canonical SMILES for both molecules
        can_target = Chem.MolToSmiles(mol_target, canonical=True)
        can_generated = Chem.MolToSmiles(mol_generated, canonical=True)

        return 1.0 if can_target == can_generated else 0.0
    except Exception:
        return 0.0

def tanimoto_reward(target, generated):
    """Reward based on Tanimoto similarity between molecular fingerprints"""
    try:
        # Remove spaces and strip both target and generated SMILES
        target_clean = target.replace(' ', '').strip()
        generated_clean = generated.replace(' ', '').strip()
        
        # Convert SMILES to molecules
        mol1 = Chem.MolFromSmiles(target_clean)
        mol2 = Chem.MolFromSmiles(generated_clean)
        
        # If either molecule is invalid, return 0
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Generate canonical SMILES for both molecules
        can_target = Chem.MolToSmiles(mol1, canonical=True)
        can_generated = Chem.MolToSmiles(mol2, canonical=True)
        
        # Recreate molecules from canonical SMILES to ensure consistency
        mol1 = Chem.MolFromSmiles(can_target)
        mol2 = Chem.MolFromSmiles(can_generated)
        
        # Generate fingerprints
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        # Convert to numpy arrays
        fp1_array = np.array(fp1)
        fp2_array = np.array(fp2)
        
        # Calculate Tanimoto similarity
        intersection = np.sum(fp1_array & fp2_array)
        union = np.sum(fp1_array | fp2_array)
        
        # Avoid division by zero
        if union == 0:
            return 0.0
            
        tanimoto = intersection / union
        return float(tanimoto)
    except:
        return 0.0

def ecfp6_reward(target, generated):
    """Reward based on ECFP6 (Morgan radius 3) IoU similarity between molecules"""
    try:
        # Remove spaces and strip both target and generated SMILES
        target_clean = target.replace(' ', '').strip()
        generated_clean = generated.replace(' ', '').strip()
        
        # Convert SMILES to molecules
        mol1 = Chem.MolFromSmiles(target_clean)
        mol2 = Chem.MolFromSmiles(generated_clean)
        
        # If either molecule is invalid, return 0
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Generate ECFP6 fingerprints (Morgan radius 3)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
        
        # Convert to numpy arrays
        fp1_array = np.array(fp1)
        fp2_array = np.array(fp2)
        
        # Calculate Intersection over Union
        intersection = np.sum(fp1_array & fp2_array)
        union = np.sum(fp1_array | fp2_array)
        
        # Avoid division by zero
        if union == 0:
            return 0.0
            
        iou = intersection / union
        return float(iou)
    except:
        return 0.0

def valid_smiles_reward(target, generated):
    """Reward based on whether the generated SMILES is valid"""
    try:
        # Remove spaces and strip the generated SMILES
        generated_clean = generated.replace(' ', '').strip()
        
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(generated_clean)
        
        # Return 1 if valid, 0 if invalid
        return 1.0 if mol is not None else 0.0
    except:
        return 0.0

def mcs_ratio_reward(target, generated):
    """Reward based on Maximum Common Substructure (MCS) ratio"""
    try:
        # Remove spaces and strip both target and generated SMILES
        target_clean = target.replace(' ', '').strip()
        generated_clean = generated.replace(' ', '').strip()
        
        # Convert SMILES to molecules
        mol1 = Chem.MolFromSmiles(target_clean)
        mol2 = Chem.MolFromSmiles(generated_clean)
        
        # If either molecule is invalid, return 0
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Find MCS with timeout to prevent hanging on complex structures
        mcs = rdFMCS.FindMCS([mol1, mol2], timeout=1)
        
        # If MCS is empty, return 0
        if mcs.numAtoms == 0:
            return 0.0
        
        # Return ratio of MCS atoms to target molecule atoms
        target_atoms = mol1.GetNumAtoms()
        mcs_ratio = mcs.numAtoms / target_atoms if target_atoms > 0 else 0.0
        
        return float(mcs_ratio)
    except:
        return 0.0

def parse_cot_answer(text: str) -> str:
    """Extract the answer portion from a CoT-formatted string."""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[1].split("</answer>")[0].strip()
    return ""


def cot_structure_reward(text: str) -> float:
    """Reward if the text follows the <thinking>...</thinking><answer>...</answer> template."""
    required_tags = ["<thinking>", "</thinking>", "<answer>", "</answer>"]
    if all(tag in text for tag in required_tags):
        if text.count("<thinking>") == 1 and text.count("</thinking>") == 1 and text.count("<answer>") == 1 and text.count("</answer>") == 1:
            if text.index("</thinking>") < text.index("<answer>"):
                if text.split("</answer>")[1].strip() == "":
                    return 1.0
    return 0.0

def load_pretrained_model(checkpoint_path, config, device):
    """Load a pretrained model from a checkpoint"""
    print(f"Loading pretrained model from {checkpoint_path}")
    
    # Load vocabularies
    smiles_vocab_size, nmr_vocab_size, nmr_tokenizer = load_vocabularies(config)
    
    # Initialize model
    ir_vocab_size = None
    if config['data'].get('ir_as_prompt', False):
        ir_vocab_path = config['data'].get('ir_tokenizer_path')
        if ir_vocab_path:
            with open(ir_vocab_path, 'r') as f:
                import json
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
        ir_vocab_size=ir_vocab_size
    ).to(device)
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from checkpoint at epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    
    return model, nmr_tokenizer

def main():
    parser = argparse.ArgumentParser(description='Post-train SMILES generation model with GRPO RL')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to pretrained model checkpoint')
    parser.add_argument('--iterations', type=int, default=None, help='Number of training iterations')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for sampling (higher = more random)')
    parser.add_argument('--group-size', type=int, default=None, help='Group size for GRPO')
    parser.add_argument('--micro-group-size', type=int, default=None, help='Micro-batch size within each group')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate for GRPO')
    parser.add_argument('--epsilon', type=float, default=None, help='GRPO epsilon for clipping')
    parser.add_argument('--beta', type=float, default=None, help='KL penalty coefficient')
    parser.add_argument('--wandb', type=lambda s: s.lower() in ["true", "1", "yes"], default=None, help='Log to wandb (true/false)')
    parser.add_argument('--block-ir', type=lambda s: s.lower() in ["true", "1", "yes"], default=None, help='Block IR signals in the model inputs (true/false)')
    parser.add_argument('--block-nmr', type=lambda s: s.lower() in ["true", "1", "yes"], default=None, help='Block NMR signals in the model inputs (true/false)')
    parser.add_argument('--exact-match-reward', type=lambda s: s.lower() in ["true", "1", "yes"], default=None, help='Use exact match reward (true/false)')
    parser.add_argument('--tanimoto-reward', type=lambda s: s.lower() in ["true", "1", "yes"], default=None, help='Use Tanimoto similarity reward (true/false)')
    parser.add_argument('--ecfp6-reward', type=lambda s: s.lower() in ["true", "1", "yes"], default=None, help='Use ECFP6 IoU reward (true/false)')
    parser.add_argument('--valid-smiles-reward', type=lambda s: s.lower() in ["true", "1", "yes"], default=None, help='Use valid SMILES reward (true/false)')
    parser.add_argument('--mcs-ratio-reward', type=lambda s: s.lower() in ["true", "1", "yes"], default=None, help='Use MCS ratio reward (true/false)')
    parser.add_argument('--cot-reward', type=lambda s: s.lower() in ["true", "1", "yes"], default=None, help='Reward correct CoT format (true/false)')
    parser.add_argument('--log-frequency', type=int, default=None, help='Number of iterations between metric logging')
    parser.add_argument('--validation-frequency', type=int, default=None, help='Number of iterations between validations')

    args = parser.parse_args()

    # Load configuration and ensure necessary sections exist
    config = load_config(args.config)
    config.setdefault('rl', {}).setdefault('grpo', {})
    config.setdefault('data', {})
    config.setdefault('training', {})

    # Override config with CLI args if provided, otherwise use config defaults or hard-coded defaults
    config['rl']['grpo']['max_iterations'] = (args.iterations if args.iterations is not None 
                                                else config['rl']['grpo'].get('max_iterations', 1000))
    config['rl']['grpo']['temperature'] = (args.temperature if args.temperature is not None 
                                           else config['rl']['grpo'].get('temperature', 1))
    config['rl']['grpo']['group_size'] = (args.group_size if args.group_size is not None 
                                          else config['rl']['grpo'].get('group_size', 8))
    config['rl']['grpo']['micro_group_size'] = (args.micro_group_size if args.micro_group_size is not None 
                                                else config['rl']['grpo'].get('micro_group_size', 1))
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
        config['training']['batch_size'] = args.batch_size
        config['rl']['grpo']['batch_size'] = args.batch_size
    else:
        batch_size_for_loader = config['training'].get('batch_size', 32)
        config['data']['batch_size'] = batch_size_for_loader
        config['training']['batch_size'] = batch_size_for_loader
        config['rl']['grpo']['batch_size'] = config['rl']['grpo'].get('batch_size', 1)

    config['rl']['grpo']['learning_rate'] = (args.lr if args.lr is not None 
                                             else config['rl']['grpo'].get('learning_rate', 1e-6))
    config['rl']['grpo']['epsilon'] = (args.epsilon if args.epsilon is not None 
                                       else config['rl']['grpo'].get('epsilon', 1))
    config['rl']['grpo']['beta'] = (args.beta if args.beta is not None 
                                    else config['rl']['grpo'].get('beta', 0.01))

    config['rl']['grpo']['log_wandb'] = (args.wandb if args.wandb is not None 
                                         else config['rl']['grpo'].get('log_wandb', False))
    config['rl']['grpo']['exact_match_reward'] = (args.exact_match_reward if args.exact_match_reward is not None 
                                                  else config['rl']['grpo'].get('exact_match_reward', True))
    config['rl']['grpo']['tanimoto_reward'] = (args.tanimoto_reward if args.tanimoto_reward is not None 
                                               else config['rl']['grpo'].get('tanimoto_reward', True))
    config['rl']['grpo']['ecfp6_reward'] = (args.ecfp6_reward if args.ecfp6_reward is not None 
                                            else config['rl']['grpo'].get('ecfp6_reward', True))
    config['rl']['grpo']['valid_smiles_reward'] = (args.valid_smiles_reward if args.valid_smiles_reward is not None 
                                                    else config['rl']['grpo'].get('valid_smiles_reward', True))
    config['rl']['grpo']['mcs_ratio_reward'] = (args.mcs_ratio_reward if args.mcs_ratio_reward is not None
                                                 else config['rl']['grpo'].get('mcs_ratio_reward', True))
    config['rl']['grpo']['cot_reward'] = (args.cot_reward if args.cot_reward is not None
                                          else config['rl']['grpo'].get('cot_reward', False))

    log_frequency = (args.log_frequency if args.log_frequency is not None 
                     else config['training'].get('logging_frequency', 10))
    validation_frequency = (args.validation_frequency if args.validation_frequency is not None 
                            else config['training'].get('validation_frequency', 10))

    block_ir = args.block_ir if args.block_ir is not None else False
    block_nmr = args.block_nmr if args.block_nmr is not None else False

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained model
    model, nmr_tokenizer = load_pretrained_model(args.checkpoint, config, device)
    
    # Create a reference model (deep copy of the current model)
    ref_model = deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    # Explicitly print what batch size will be used for data loaders
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    # Check which config section contains the batch size
    data_batch_size = data_config.get('batch_size')
    training_batch_size = training_config.get('batch_size')
    
    print(f"\n[CONFIG] Config sections with batch size:")
    print(f"  - data.batch_size: {data_batch_size}")
    print(f"  - training.batch_size: {training_batch_size}")
    
    # Use training batch size as that's what's in the config file
    batch_size_for_loader = training_batch_size if training_batch_size is not None else 32
    
    # Ensure batch size is set in both locations for consistency
    if 'data' not in config:
        config['data'] = {}
    if 'training' not in config:
        config['training'] = {}
    
    config['data']['batch_size'] = batch_size_for_loader
    config['training']['batch_size'] = batch_size_for_loader
    
    print(f"[CONFIG] Final batch size for data loaders: {batch_size_for_loader}")
    
    # Create data loaders
    print(f"[CONFIG] Creating data loaders with batch size: {batch_size_for_loader}")
    
    # Debug: Print a snippet of the create_data_loaders function to see where it's getting batch_size from
    try:
        import inspect
        create_data_loaders_src = inspect.getsource(create_data_loaders)
        for line in create_data_loaders_src.split('\n'):
            if 'batch_size' in line or 'batch' in line:
                print(f"[DEBUG] In create_data_loaders: {line.strip()}")
    except Exception as e:
        print(f"[DEBUG] Could not print create_data_loaders source: {e}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        smiles_tokenizer=tokenizer,
        nmr_tokenizer=nmr_tokenizer,
        config=config
    )
    
    # Initialize wandb if needed
    if config['rl']['grpo']['log_wandb']:
        run_name = f"grpo_rl_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        wandb.init(
            project=config['wandb']['project'],
            name=run_name,
            config=config
        )
    
    # Print chosen reward configuration
    print(f"\n====== REWARD CONFIGURATION ======")
    print(f"- Exact match reward: {'ENABLED' if config['rl']['grpo']['exact_match_reward'] else 'DISABLED'}")
    print(f"- Tanimoto reward: {'ENABLED' if config['rl']['grpo']['tanimoto_reward'] else 'DISABLED'}")
    print(f"- ECFP6 reward: {'ENABLED' if config['rl']['grpo']['ecfp6_reward'] else 'DISABLED'}")
    print(f"- Valid SMILES reward: {'ENABLED' if config['rl']['grpo']['valid_smiles_reward'] else 'DISABLED'}")
    print(f"- MCS Ratio reward: {'ENABLED' if config['rl']['grpo']['mcs_ratio_reward'] else 'DISABLED'}")
    print(f"- CoT format reward: {'ENABLED' if config['rl']['grpo']['cot_reward'] else 'DISABLED'}")
    print(f"==================================\n")
    
    # Ensure at least one reward is active
    if not (config['rl']['grpo']['exact_match_reward'] or 
            config['rl']['grpo']['tanimoto_reward'] or 
            config['rl']['grpo']['ecfp6_reward'] or 
            config['rl']['grpo']['valid_smiles_reward'] or
            config['rl']['grpo']['mcs_ratio_reward'] or
            config['rl']['grpo']['cot_reward']):
        print("ERROR: At least one reward function must be enabled!")
        return

    # Get batch size for GRPO (from rl.grpo config)
    grpo_batch_size = config['rl']['grpo'].get('batch_size', 1)
    grpo_group_size = config['rl']['grpo'].get('group_size', 8)
    grpo_micro_group_size = config['rl']['grpo'].get('micro_group_size', 1)
    
    # Print GRPO batch configuration
    print(f"\n====== GRPO BATCH CONFIGURATION ======")
    print(f"- GRPO batch size: {grpo_batch_size}")
    print(f"- GRPO group size: {grpo_group_size}")
    print(f"- GRPO micro-group size: {grpo_micro_group_size}")
    print(f"====================================\n")
    
    # Initialize GRPO trainer
    grpo = GRPO(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        group_size=grpo_group_size,
        micro_group_size=grpo_micro_group_size,
        batch_size=grpo_batch_size,
        max_iterations=config['rl']['grpo']['max_iterations'],
        train_loader=train_loader,
        val_loader=val_loader,
        use_exact_match_reward=config['rl']['grpo']['exact_match_reward'],
        use_tanimoto_reward=config['rl']['grpo']['tanimoto_reward'],
        use_ecfp6_reward=config['rl']['grpo']['ecfp6_reward'],
        use_valid_smiles_reward=config['rl']['grpo']['valid_smiles_reward'],
        use_mcs_ratio_reward=config['rl']['grpo']['mcs_ratio_reward'],
        use_cot_reward=config['rl']['grpo']['cot_reward'],
        log_wandb=config['rl']['grpo']['log_wandb'],
        lr=config['rl']['grpo']['learning_rate'],
        beta=config['rl']['grpo']['beta'],
        epsilon=config['rl']['grpo']['epsilon'],
        temperature=config['rl']['grpo']['temperature'],
        device=device,
        use_kl=True,
        log_frequency=log_frequency,
        validation_frequency=validation_frequency
    )
    
    # Run training
    grpo.train(num_iterations=config['rl']['grpo']['max_iterations'])
    
    # Save the fine-tuned model
    output_dir = Path('checkpoints_rl')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"grpo_finetuned_{datetime.datetime.now().strftime('%m%d_%H%M')}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'timestamp': datetime.datetime.now().isoformat()
    }, model_path)
    
    print(f"Fine-tuned model saved to {model_path}")
    
    # Evaluate on test set
    model.eval()
    test_metrics = {}
    
    with torch.no_grad():
        correct = 0
        total = 0
        
        for batch in tqdm(test_loader, desc="Testing"):
            target_tokens, ir_data, nmr_tokens, _ = batch
            
            # Apply blocking if requested
            if block_ir:
                ir_data = None
            if block_nmr:
                nmr_tokens = None
                
            # Move to device
            target_tokens = target_tokens.to(device)
            if ir_data is not None:
                ir_data = ir_data.to(device)
            if nmr_tokens is not None:
                nmr_tokens = nmr_tokens.to(device)
            
            # Generate with greedy decoding
            predictions = greedy_decode(
                model=model,
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                tokenizer=tokenizer,
                device=device
            )
            
            # Get targets
            targets = []
            for tgt in target_tokens:
                try:
                    eos_idx = tgt.tolist().index(tokenizer.sep_token_id)
                    tgt = tgt[:eos_idx]
                except ValueError:
                    pass
                decoded = tokenizer.decode(tgt[1:])
                targets.append(decoded)
            
            # Count correct predictions
            for pred, tgt in zip(predictions, targets):
                total += 1
                if pred.strip().replace(" ", "") == tgt.strip().replace(" ", ""):
                    correct += 1
    
    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0
    test_metrics["test_accuracy"] = accuracy
    print(f"Final test accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Log final metrics
    if config['rl']['grpo']['log_wandb']:
        wandb.log(test_metrics)
        wandb.finish()


if __name__ == '__main__':
    main()
