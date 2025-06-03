import math
import time
import torch
import wandb
import numpy
import random
import argparse
import contextlib
import torch.optim as optim
import torch.autograd
from statistics import mean
from dataclasses import asdict
from pathlib import Path
import json
import sys
import os
import yaml
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from utils.optimization.muon import Muon # Import Muon optimizer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.spectral_datasets import SpectralVLMDataset, SpectralCollator
from data.processors import get_image_processor, get_tokenizer, extract_smiles_from_response
from nanoVLM.models.spectral_vision_language_model import SpectralVisionLanguageModel
from nanoVLM.models.config import VLMConfig
import nanoVLM.models.utils as utils

# Prevent tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEBUG = True # Or False, to toggle debugging


def init_dist():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())


def destroy_dist():
    dist.destroy_process_group()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_master():
    return dist.get_rank() == 0 if is_dist() else True


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_rank():
    return dist.get_rank() if is_dist() else 0


def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all


def wrap_model(model):
    return DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)


def load_spectral_config(config_path):
    """Load spectral VLM training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to object with dot notation access
    class ConfigObj:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigObj(value))
                else:
                    setattr(self, key, value)
        
        def __getitem__(self, key):
            return getattr(self, key)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
            
        def to_dict(self):
            """Convert back to dictionary for logging"""
            result = {}
            for key, value in self.__dict__.items():
                if isinstance(value, ConfigObj):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
            return result
    
    return ConfigObj(config_dict)


def get_run_name(train_cfg):
    dataset_size = "full_ds" if train_cfg.training.data_cutoff_idx is None else f"{train_cfg.training.data_cutoff_idx}samples"
    batch_size = f"bs{int(train_cfg.training.batch_size*get_world_size()*train_cfg.training.gradient_accumulation_steps)}"
    epochs = f"ep{train_cfg.training.epochs}"
    # Updated learning rate logging for muon_mix
    lr_adamw = train_cfg.training.learning_rate
    lr_muon = train_cfg.optimizer.muon.lr
    learning_rate_str = f"lrAdam{lr_adamw}-lrMuon{lr_muon}"
    num_gpus = f"{get_world_size()}xGPU"
    encoder_type = f"{train_cfg.model.spectral_encoder_type}enc"
    date = time.strftime("%m%d")

    return f"SpectralVLM_{encoder_type}_{num_gpus}_{dataset_size}_{batch_size}_{epochs}_{learning_rate_str}_{date}"


def get_dataloaders(train_cfg, vlm_cfg):
    """Create data loaders for spectral VLM training"""
    
    # Get processors
    if train_cfg.model.include_images:
        image_processor = get_image_processor(vlm_cfg.vit_img_size)
    else:
        image_processor = None
    
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)
    
    # Create datasets
    train_dataset = SpectralVLMDataset(
        data_dir=train_cfg.data.data_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        split='train',
        max_length=train_cfg.model.max_length,
        include_images=train_cfg.model.include_images,
        #nmr_vocab_path=train_cfg.data.nmr_vocab_path
    )
    
    val_dataset = SpectralVLMDataset(
        data_dir=train_cfg.data.data_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        split='val',
        max_length=train_cfg.model.max_length,
        include_images=train_cfg.model.include_images,
        #nmr_vocab_path=train_cfg.data.nmr_vocab_path
    )
    
    test_dataset = SpectralVLMDataset(
        data_dir=train_cfg.data.data_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        split='test',
        max_length=train_cfg.model.max_length,
        include_images=train_cfg.model.include_images,
        #nmr_vocab_path=train_cfg.data.nmr_vocab_path
    )
    
    # Apply data cutoff if specified
    if train_cfg.training.data_cutoff_idx is not None:
        train_dataset.smiles_data = train_dataset.smiles_data[:train_cfg.training.data_cutoff_idx]
        train_dataset.nmr_data = train_dataset.nmr_data[:train_cfg.training.data_cutoff_idx]
        if train_dataset.ir_data is not None:
            train_dataset.ir_data = train_dataset.ir_data[:train_cfg.training.data_cutoff_idx]
    
    # Create collator
    collator = SpectralCollator(tokenizer, max_length=train_cfg.model.max_length)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset, 
        rank=get_rank(),
        num_replicas=get_world_size(),
    ) if is_dist() else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False
    ) if is_dist() else None
    
    test_sampler = DistributedSampler(
        test_dataset,
        rank=get_rank(), 
        num_replicas=get_world_size(),
        shuffle=False
    ) if is_dist() else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.training.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.training.batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    return train_loader, val_loader, test_loader


def evaluate_smiles_generation(model, tokenizer, test_loader, device, num_samples=50):
    """Evaluate SMILES generation quality"""
    model.eval()
    total_examples = 0
    valid_smiles = 0
    exact_matches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if total_examples >= num_samples:
                break
                
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get optional modalities
            ir_data = batch.get('ir_data', None)
            if ir_data is not None:
                ir_data = ir_data.to(device)
            
            image = batch.get('image', None)
            if image is not None:
                image = image.to(device)
            
            # Generate SMILES
            generated = model.generate(
                input_ids=input_ids,
                image=image,
                ir_data=ir_data,
                attention_mask=attention_mask,
                max_new_tokens=100,
                temperature=0.7,
                greedy=True
            )
            
            # Decode generated and target sequences
            generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            # Replace -100 with pad_token_id before decoding labels
            # Create a copy to avoid modifying the original labels tensor
            labels_for_decoding = labels.clone()
            labels_for_decoding[labels_for_decoding == -100] = tokenizer.pad_token_id
            target_text = tokenizer.batch_decode(labels_for_decoding, skip_special_tokens=True)
            
            # Extract SMILES from XML tags
            for gen, tgt in zip(generated_text, target_text):
                total_examples += 1
                
                # Extract SMILES from generated text
                try:
                    if '<smiles>' in gen and '</smiles>' in gen:
                        gen_smiles = gen.split('<smiles>')[1].split('</smiles>')[0].strip()
                    else:
                        gen_smiles = gen.strip()
                    
                    # Extract target SMILES
                    if '<smiles>' in tgt and '</smiles>' in tgt:
                        tgt_smiles = tgt.split('<smiles>')[1].split('</smiles>')[0].strip()
                    else:
                        tgt_smiles = tgt.strip()
                    
                    # Check if generated SMILES is valid (basic check)
                    if len(gen_smiles) > 0 and not gen_smiles.isspace():
                        valid_smiles += 1
                        
                        # Check exact match
                        if gen_smiles == tgt_smiles:
                            exact_matches += 1
                            
                except Exception as e:
                    print(f"Error processing SMILES: {e}")
                    continue
    
    # Calculate metrics
    valid_rate = valid_smiles / max(total_examples, 1)
    exact_match_rate = exact_matches / max(total_examples, 1)
    
    return {
        'valid_smiles_rate': valid_rate,
        'exact_match_rate': exact_match_rate,
        'total_samples': total_examples
    }


def log_qualitative_generations(model_eval, tokenizer, val_loader, device, train_cfg, global_step):
    """Logs a few qualitative generation examples to console and WandB."""
    if not is_master():
        return

    print(f"\nLogging qualitative generation samples at global step {global_step}...")
    model_eval.eval()
    
    num_samples_to_log = train_cfg.evaluation.get('num_qualitative_generation_samples', 3)
    logged_samples = 0
    wandb_table_data = []

    # Special tokens used by the tokenizer/model that we might need to handle or be aware of.
    # These are often added by the tokenizer and might appear in decoded strings if not skipped.
    # For simple SMILES extraction, skip_special_tokens=True is usually sufficient.
    # smiles_tag_str = "<smiles>"
    # smiles_tag_tokens = tokenizer.encode(smiles_tag_str, add_special_tokens=False)


    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if logged_samples >= num_samples_to_log:
                break

            input_ids_full_prompt_batch = batch['input_ids'].to(device)
            attention_mask_full_prompt_batch = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            
            ir_data_batch = batch.get('ir_data', None)
            if ir_data_batch is not None:
                ir_data_batch = ir_data_batch.to(device)
            
            image_batch = batch.get('image', None)
            if image_batch is not None:
                image_batch = image_batch.to(device)

            current_batch_size = input_ids_full_prompt_batch.size(0)
            for k in range(current_batch_size):
                if logged_samples >= num_samples_to_log:
                    break

                input_ids_full_sample = input_ids_full_prompt_batch[k]
                attention_mask_full_sample = attention_mask_full_prompt_batch[k]
                labels_sample = labels_batch[k]

                # 1. Decode Target SMILES for logging
                valid_labels_sample = labels_sample[labels_sample != -100]
                # Ensure skip_special_tokens=True to remove things like <EOS> if present after SMILES.
                # The `extract_smiles_from_response` should handle the <smiles>content</smiles> part.
                target_smiles_full_text = tokenizer.decode(valid_labels_sample, skip_special_tokens=True)
                target_smiles_for_log = extract_smiles_from_response(target_smiles_full_text)


                # 2. Prepare Generation Prompt (truncate before <smiles> tag)
                # We need to find the tokenized version of "<smiles>"
                # It's safer to tokenize "<smiles>" once and use that.
                # However, simple string split on decoded text before re-tokenizing for generation prompt might be error prone.
                # Best to find in token space.
                
                # Let's find the sequence of tokens for "<smiles>"
                # Important: tokenizer might add prefix space or behave differently, so test this.
                # For many tokenizers `tokenizer.encode("<smiles>", add_special_tokens=False)` is robust.
                # Using a fixed string and finding it in the decoded version of input_ids_full_sample, then truncating
                # the original input_ids_full_sample based on character length is NOT robust due to token-char misalignment.

                # Robust way: Find token sequence for "<smiles>"
                # We assume the prompt is "Text... <nmr_data>NMR_TEXT</nmr_data> ... <smiles>TARGET_SMILES</smiles>"
                # The collator creates `input_ids_full_sample` from this.
                # We want the part "Text... <nmr_data>NMR_TEXT</nmr_data> ..." as input for generation.

                decoded_full_input_prompt_text = tokenizer.decode(input_ids_full_sample, skip_special_tokens=False) # Keep special tokens to find structure
                smiles_tag_str_in_prompt = "<smiles>" # The string we are looking for
                
                idx_smiles_tag = decoded_full_input_prompt_text.find(smiles_tag_str_in_prompt)

                generation_prompt_text = ""
                if idx_smiles_tag != -1:
                    generation_prompt_text = decoded_full_input_prompt_text[:idx_smiles_tag]
                else:
                    # Fallback: if <smiles> tag is not found, this sample might be problematic for this logging.
                    # Or, the prompt structure is different. For now, try to use what we have.
                    # This could be an issue if EOS tokens are part of the slice.
                    print(f"Warning: '{smiles_tag_str_in_prompt}' not found in decoded input for sample {logged_samples + 1}. Using full decoded input as generation prompt (may be incorrect).")
                    generation_prompt_text = decoded_full_input_prompt_text # This will likely just regenerate the target if it's already there.

                # Re-tokenize the generation prompt.
                # Ensure consistent padding/truncation strategy with how model expects inputs for generation.
                # `model.generate` usually handles this internally if just `input_ids` are passed without explicit mask.
                # But it's better to provide the attention mask.
                
                gen_inputs = tokenizer(generation_prompt_text, return_tensors="pt", padding=False, truncation=True, max_length=train_cfg.model.max_length)
                generation_input_ids = gen_inputs["input_ids"].to(device)
                generation_attention_mask = gen_inputs["attention_mask"].to(device)


                # 3. Extract NMR text for logging from the generation_prompt_text
                nmr_for_log = "Could not extract NMR from prompt"
                nmr_tag_open = "<nmr_data>"
                nmr_tag_close = "</nmr_data>"
                if nmr_tag_open in generation_prompt_text and nmr_tag_close in generation_prompt_text:
                    try:
                        nmr_for_log = generation_prompt_text.split(nmr_tag_open)[1].split(nmr_tag_close)[0].strip()
                    except IndexError: # Should not happen if both tags are present
                        pass 
                
                current_ir_data_sample = ir_data_batch[k].unsqueeze(0) if ir_data_batch is not None else None
                current_image_sample = image_batch[k].unsqueeze(0) if image_batch is not None else None
                
                # 4. Generate SMILES
                # Use a consistent setting for generation logging
                generated_tokens_indices = model_eval.generate(
                    input_ids=generation_input_ids,
                    image=current_image_sample,
                    ir_data=current_ir_data_sample,
                    attention_mask=generation_attention_mask,
                    max_new_tokens=train_cfg.model.get('generation_max_new_tokens', 100), # Configurable max new tokens for logging
                    temperature=train_cfg.evaluation.get('generation_temperature', 0.7),
                    greedy=train_cfg.evaluation.get('generation_greedy', True)
                )
                
                # 5. Decode generated tokens and extract SMILES
                full_generated_text = tokenizer.decode(generated_tokens_indices[0], skip_special_tokens=True)
                predicted_smiles_for_log = extract_smiles_from_response(full_generated_text)
                generated_tokens_indices_list = generated_tokens_indices[0].cpu().tolist()
                
                # Console Log
                print(f"  --- Qualitative Sample {logged_samples + 1} ---")
                print(f"  Input NMR (extracted): {nmr_for_log}")
                print(f"  IR Data Present: {'Yes' if current_ir_data_sample is not None else 'No'}")
                print(f"  Image Present: {'Yes' if current_image_sample is not None else 'No'}")
                print(f"  Target SMILES: {target_smiles_for_log}")
                print(f"  Predicted SMILES: {predicted_smiles_for_log}")
                print(f"  Generated Token Indices: {generated_tokens_indices_list}") # Log indices
                # print(f"  Full Generation Prompt (for debug): {generation_prompt_text}")
                # print(f"  Full Generated Text (raw): {full_generated_text}")
                print(f"  -----------------------------")

                if train_cfg.logging.log_wandb and wandb.run:
                    wandb_table_data.append([
                        global_step,
                        nmr_for_log,
                        'Yes' if current_ir_data_sample is not None else 'No',
                        'Yes' if current_image_sample is not None else 'No',
                        target_smiles_for_log,
                        predicted_smiles_for_log,
                        str(generated_tokens_indices_list), # Log indices as string
                        generation_prompt_text, # Log the actual prompt used for generation
                        full_generated_text
                    ])
                
                logged_samples += 1
            
            if logged_samples >= num_samples_to_log:
                break
    
    if train_cfg.logging.log_wandb and wandb.run and wandb_table_data:
        columns = ["Global Step", "Input NMR", "IR Present", "Image Present", 
                   "Target SMILES", "Predicted SMILES", "Generated Token Indices", 
                   "Generation Prompt", "Full Generated Text"]
        table = wandb.Table(columns=columns, data=wandb_table_data)
        wandb.log({"qualitative_generation_samples": table}, step=global_step)
        print("Qualitative generation samples logged to WandB.")

    model_eval.train() # Set back to train mode
    print("Finished logging qualitative generation samples.\n")


# Cosine learning rate schedule with warmup
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
        
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(train_cfg, vlm_cfg):
    """Main training function"""
    torch.autograd.set_detect_anomaly(True)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(train_cfg, vlm_cfg)
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)
    
    total_dataset_size = len(train_loader.dataset)
    
    # Initialize wandb
    if train_cfg.logging.log_wandb and is_master():
        run_name = get_run_name(train_cfg)
        # Corrected run_name update based on total_dataset_size
        if train_cfg.training.data_cutoff_idx is None and 'total_dataset_size' in locals():
            run_name = run_name.replace(f"{total_dataset_size}samples", f"{total_dataset_size}samples") # Placeholder, ensure total_dataset_size is defined earlier or handle
        elif train_cfg.training.data_cutoff_idx is None:
             # Fallback if total_dataset_size is not available when expected
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")


        run = wandb.init(
            entity=train_cfg.logging.wandb_entity,
            project=train_cfg.logging.wandb_project,
            config={
                "VLMConfig": asdict(vlm_cfg),
                "SpectralTrainConfig": train_cfg.to_dict()
            },
            name=run_name,
        )
    
    # Initialize model
    spectral_cfg = {
        'embed_dim': vlm_cfg.vit_hidden_dim,
        'encoder_type': train_cfg.model.spectral_encoder_type,
        'ir_as_prompt': train_cfg.model.ir_as_prompt
    }
    
    if train_cfg.vlm_checkpoint.resume_from_checkpoint:
        print(f"Loading pretrained VLM from {train_cfg.vlm_checkpoint.path}")
        # Update VLM config with checkpoint path
        vlm_cfg.vlm_checkpoint_path = train_cfg.vlm_checkpoint.path
        # Load base VLM and initialize spectral components
        model = SpectralVisionLanguageModel(vlm_cfg, load_backbone=train_cfg.vlm_checkpoint.load_backbone, spectral_cfg=spectral_cfg)
    else:
        model = SpectralVisionLanguageModel(vlm_cfg, load_backbone=False, spectral_cfg=spectral_cfg)
    
    if is_master():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"SpectralVLM initialized with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Training summary{{\' (global)\' if is_dist() else \'\'}}: {len(train_loader.dataset)} samples, {int(len(train_loader)*get_world_size())} batches/epoch")
    
    # Optimizer setup for muon_mix
    optimizers = []
    all_params_list = [] # To collect all parameters for gradient clipping

    matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    vector_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    if is_master():
        matrix_param_count = sum(p.numel() for p in matrix_params)
        vector_param_count = sum(p.numel() for p in vector_params)
        total_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Optimizer] Parameter distribution for 'muon_mix':")
        if total_model_params > 0 :
            print(f"  - Muon (matrix_params >=2D): {matrix_param_count:,} parameters ({matrix_param_count/total_model_params:.1%})")
            print(f"  - AdamW (vector_params <2D): {vector_param_count:,} parameters ({vector_param_count/total_model_params:.1%})")
        else:
            print(f"  - Muon (matrix_params >=2D): {matrix_param_count:,} parameters")
            print(f"  - AdamW (vector_params <2D): {vector_param_count:,} parameters")


    if matrix_params:
        muon_optimizer = Muon(
            matrix_params,
            lr=train_cfg.optimizer.muon.lr,
            momentum=train_cfg.optimizer.muon.get('momentum', 0.95),
            nesterov=train_cfg.optimizer.muon.get('nesterov', True),
            weight_decay=train_cfg.optimizer.muon.get('weight_decay', 0.01), # Added weight decay for Muon
            # ns_steps and other Muon specific params can be added from train_cfg if available
            # rank=get_rank(), # Important for some Muon versions, ensure it's used if needed
            # world_size=get_world_size() # Important for some Muon versions
        )
        optimizers.append(muon_optimizer)
        all_params_list.extend(matrix_params)
        if is_master():
            print(f"  Muon optimizer initialized with LR: {train_cfg.optimizer.muon.lr}, WD: {train_cfg.optimizer.muon.get('weight_decay', 0.01)}")


    if vector_params:
        adamw_optimizer = optim.AdamW(
            vector_params,
            lr=train_cfg.training.learning_rate, # General LR for AdamW components
            betas=train_cfg.optimizer.adamw.get('betas', (0.9, 0.999)),
            eps=train_cfg.optimizer.adamw.get('eps', 1.0e-8),
            weight_decay=train_cfg.optimizer.adamw.weight_decay
        )
        optimizers.append(adamw_optimizer)
        all_params_list.extend(vector_params)
        if is_master():
            print(f"  AdamW optimizer initialized with LR: {train_cfg.training.learning_rate}, WD: {train_cfg.optimizer.adamw.weight_decay}")

    if not optimizers:
        raise ValueError("No parameters found for optimization. Check model parameter dimensions and requires_grad settings.")

    # The scheduler will primarily control the AdamW part. Muon might have its own internal schedule or fixed LR.
    # For simplicity, we'll use the first optimizer (likely AdamW if vector_params exist, or Muon if only matrix_params) for the scheduler.
    # Or, more robustly, create a scheduler for AdamW if it exists, and handle Muon LR separately if needed.
    # Let's assume the primary scheduler targets AdamW learning rate.
    primary_optimizer_for_scheduler = next((opt for opt in optimizers if isinstance(opt, optim.AdamW)), optimizers[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if train_cfg.training.compile:
        model = torch.compile(model)
        
    if is_dist():
        model = wrap_model(model)
    
    # Training loop
    epoch_times = []
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(train_cfg.training.epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        # optimizer.zero_grad() # Zero grad for each optimizer in the loop
        for opt in optimizers:
            opt.zero_grad()

        for i, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get optional modalities
            ir_data = batch.get('ir_data', None)
            if ir_data is not None:
                ir_data = ir_data.to(device)
            
            image = batch.get('image', None)
            if image is not None:
                image = image.to(device)
            
            # --- BEGIN DEBUG: Check for NaNs in input data ---
            if DEBUG:
                if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                    print("DEBUG: NaN/Inf found in input_ids!")
                if torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any():
                    print("DEBUG: NaN/Inf found in attention_mask!")
                if torch.isnan(labels).any() or torch.isinf(labels).any():
                    print("DEBUG: NaN/Inf found in labels!")
                if ir_data is not None and (torch.isnan(ir_data).any() or torch.isinf(ir_data).any()):
                    print("DEBUG: NaN/Inf found in ir_data!")
                if image is not None and (torch.isnan(image).any() or torch.isinf(image).any()):
                    print("DEBUG: NaN/Inf found in image!")
            # --- END DEBUG ---
            
            # Skip gradient sync for intermediate steps in DDP
            if (is_dist() and train_cfg.training.gradient_accumulation_steps > 1 and 
                not ((i + 1) % train_cfg.training.gradient_accumulation_steps == 0 or i + 1 == len(train_loader))):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with context:
                    logits, loss = model(
                        input_ids=input_ids,
                        image=image,
                        ir_data=ir_data,
                        attention_mask=attention_mask,
                        targets=labels
                    )
            
            # --- BEGIN DEBUG: Check for NaNs in logits and loss ---
            if DEBUG:
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"DEBUG: NaN/Inf found in logits at step {global_step}, batch {i}!")
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"DEBUG: NaN/Inf found in loss at step {global_step}, batch {i} before grad accumulation adjustment!")
            # --- END DEBUG ---
            
            if train_cfg.training.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.training.gradient_accumulation_steps
            
            loss.backward()
            
            # Optimizer step
            if (i + 1) % train_cfg.training.gradient_accumulation_steps == 0 or i + 1 == len(train_loader):
                if train_cfg.training.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params_list, max_norm=train_cfg.training.max_grad_norm) # Use all_params_list

                # Update learning rates
                # The main scheduler affects the AdamW part primarily.
                # Muon's LR is typically managed differently or kept constant/handled internally after init.
                # Here, we adjust the LR for the primary_optimizer_for_scheduler (AdamW).
                # If Muon needs scheduled LR, it has to be handled explicitly.
                total_steps = len(train_loader) * train_cfg.training.epochs // train_cfg.training.gradient_accumulation_steps # Correct total_steps for scheduler

                current_lr_adamw = get_lr(global_step, train_cfg.training.learning_rate, total_steps)
                # Assuming Muon LR is also scheduled, or keeping it fixed as per its config
                current_lr_muon = get_lr(global_step, train_cfg.optimizer.muon.lr, total_steps) # Or keep fixed: train_cfg.optimizer.muon.lr

                for opt in optimizers:
                    if isinstance(opt, optim.AdamW):
                        for param_group in opt.param_groups:
                            param_group['lr'] = current_lr_adamw
                    elif isinstance(opt, Muon): # Check if it's Muon optimizer
                         for param_group in opt.param_groups: # Muon also has param_groups
                            param_group['lr'] = current_lr_muon


                # optimizer.step()
                # optimizer.zero_grad()
                for opt in optimizers:
                    opt.step()
                for opt in optimizers:
                    opt.zero_grad()

                global_step += 1
            
            batch_loss = loss.item()
            if train_cfg.training.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.training.gradient_accumulation_steps
            total_train_loss += batch_loss
            
            # Calculate tokens per second
            num_tokens = torch.sum(attention_mask).item()
            if ir_data is not None:
                num_tokens += ir_data.shape[0] * ir_data.shape[1]  # Add IR tokens
            total_tokens_processed += num_tokens
            
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = num_tokens / batch_duration
            
            # Gather metrics for distributed training
            batch_loss = mean(dist_gather(batch_loss)) if is_dist() else batch_loss
            tokens_per_second = sum(dist_gather(tokens_per_second)) if is_dist() else tokens_per_second
            
            # Validation and logging
            if global_step % train_cfg.evaluation.eval_interval == 0 and is_master():
                model.eval()
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    total_val_loss = 0
                    val_batches = 0
                    for val_batch in val_loader:
                        if val_batches >= 10:  # Limit validation batches
                            break
                            
                        # Move to device
                        val_input_ids = val_batch['input_ids'].to(device)
                        val_attention_mask = val_batch['attention_mask'].to(device)
                        val_labels = val_batch['labels'].to(device)
                        
                        val_ir_data = val_batch.get('ir_data', None)
                        if val_ir_data is not None:
                            val_ir_data = val_ir_data.to(device)
                        
                        val_image = val_batch.get('image', None)
                        if val_image is not None:
                            val_image = val_image.to(device)
                        
                        # --- BEGIN DEBUG: Check for NaNs in validation input data ---
                        if DEBUG:
                            if torch.isnan(val_input_ids).any() or torch.isinf(val_input_ids).any():
                                print("DEBUG: NaN/Inf found in val_input_ids!")
                            if torch.isnan(val_attention_mask).any() or torch.isinf(val_attention_mask).any():
                                print("DEBUG: NaN/Inf found in val_attention_mask!")
                            if torch.isnan(val_labels).any() or torch.isinf(val_labels).any():
                                print("DEBUG: NaN/Inf found in val_labels!")
                            if val_ir_data is not None and (torch.isnan(val_ir_data).any() or torch.isinf(val_ir_data).any()):
                                print("DEBUG: NaN/Inf found in val_ir_data!")
                            if val_image is not None and (torch.isnan(val_image).any() or torch.isinf(val_image).any()):
                                print("DEBUG: NaN/Inf found in val_image!")
                        # --- END DEBUG ---

                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            val_logits, val_loss = model(
                                input_ids=val_input_ids,
                                image=val_image,
                                ir_data=val_ir_data,
                                attention_mask=val_attention_mask,
                                targets=val_labels
                            )

                        # --- BEGIN DEBUG: Check for NaNs in validation logits and loss ---
                        if DEBUG:
                            if torch.isnan(val_logits).any() or torch.isinf(val_logits).any():
                                print(f"DEBUG: NaN/Inf found in val_logits at step {global_step}!")
                            if torch.isnan(val_loss).any() or torch.isinf(val_loss).any():
                                print(f"DEBUG: NaN/Inf found in val_loss at step {global_step}!")
                        # --- END DEBUG ---
                        
                        total_val_loss += val_loss.item()
                        val_batches += 1
                    
                    avg_val_loss = total_val_loss / max(val_batches, 1)
                    avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss
                    
                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if train_cfg.checkpointing.save_local and train_cfg.checkpointing.save_best:
                            save_path = Path("checkpoints") / "best_spectral_vlm"
                            save_path.mkdir(parents=True, exist_ok=True)
                            eval_model = model.module if is_dist() else model
                            eval_model.save_pretrained(str(save_path))
                    
                    if train_cfg.logging.log_wandb:
                        run.log({"val_loss": avg_val_loss, "best_val_loss": best_val_loss}, step=global_step)
                    
                    print(f"Step: {global_step}, Train Loss: {batch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}")
                
                model.train()
            
            # Generation evaluation (quantitative)
            if (train_cfg.evaluation.generate_during_training and 
                global_step % train_cfg.evaluation.generation_interval == 0 and 
                global_step > 0 and is_master()):
                
                eval_model = model.module if is_dist() else model
                generation_metrics = evaluate_smiles_generation(
                    eval_model, tokenizer, test_loader, device, num_samples=train_cfg.evaluation.generation_samples
                )
                
                if train_cfg.logging.log_wandb and train_cfg.logging.log_generation_metrics:
                    run.log({
                        "valid_smiles_rate": generation_metrics['valid_smiles_rate'],
                        "exact_match_rate": generation_metrics['exact_match_rate']
                    }, step=global_step)
                
                print(f"Generation - Valid SMILES: {generation_metrics['valid_smiles_rate']:.2%}, "
                      f"Exact Match: {generation_metrics['exact_match_rate']:.2%}")
                model.train() # Ensure model is back in train mode after eval

            # Qualitative Generation Logging
            if (train_cfg.evaluation.get('log_qualitative_generations', False) and
                global_step > 0 and # Avoid logging at step 0 if not desired
                global_step % train_cfg.evaluation.get('qualitative_generation_interval', 1000) == 0 and
                is_master()):
                
                eval_model_for_qual_log = model.module if is_dist() else model
                log_qualitative_generations(
                    eval_model_for_qual_log,
                    tokenizer,
                    val_loader, # Using val_loader for samples
                    device,
                    train_cfg,
                    global_step
                )
                # log_qualitative_generations handles setting model back to train mode.
            
            # Logging
            if train_cfg.logging.log_wandb and is_master():
                log_payload = {
                    "train_loss": batch_loss,
                    "tokens_per_second": tokens_per_second,
                    # "lr_spectral": adj_lr_spectral if global_step > 0 else train_cfg.learning_rates.spectral,
                    # "lr_backbones": adj_lr_backbones if global_step > 0 else train_cfg.learning_rates.backbones,
                }
                if 'current_lr_adamw' in locals() and global_step > 0 :
                    log_payload["lr_adamw"] = current_lr_adamw
                else:
                    log_payload["lr_adamw"] = train_cfg.training.learning_rate # Initial LR for AdamW
                
                if 'current_lr_muon' in locals() and global_step > 0 :
                     log_payload["lr_muon"] = current_lr_muon
                else:
                    log_payload["lr_muon"] = train_cfg.optimizer.muon.lr # Initial LR for Muon

                run.log(log_payload, step=global_step)
        
        # End of epoch
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed
        epoch_tokens_per_second = total_tokens_processed / epoch_duration
        
        if is_master():
            if train_cfg.logging.log_wandb:
                run.log({
                    "epoch_loss": avg_train_loss,
                    "epoch_duration": epoch_duration,
                    "epoch_tokens_per_second": epoch_tokens_per_second
                })
            
            print(f"Epoch {epoch+1}/{train_cfg.training.epochs}, Loss: {avg_train_loss:.4f}, "
                  f"Time: {epoch_duration:.2f}s, T/s: {epoch_tokens_per_second:.2f}")
    
    # Final evaluation
    if is_master():
        print("Final evaluation...")
        eval_model = model.module if is_dist() else model
        final_metrics = evaluate_smiles_generation(
            eval_model, tokenizer, test_loader, device, num_samples=train_cfg.evaluation.final_generation_samples
        )
        
        print(f"Final Results - Valid SMILES: {final_metrics['valid_smiles_rate']:.2%}, "
              f"Exact Match: {final_metrics['exact_match_rate']:.2%}")
        
        if train_cfg.logging.log_wandb:
            run.log({
                "final_valid_smiles_rate": final_metrics['valid_smiles_rate'],
                "final_exact_match_rate": final_metrics['exact_match_rate']
            })
            run.finish()
        
        # Save final model
        if train_cfg.checkpointing.save_local:
            save_path = Path("checkpoints") / "final_spectral_vlm"
            save_path.mkdir(parents=True, exist_ok=True)
            eval_model.save_pretrained(str(save_path))
            print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Path to tokenized data directory')
    parser.add_argument('--lr_adamw', type=float, help='Learning rate for AdamW components')
    parser.add_argument('--lr_muon', type=float, help='Learning rate for Muon components')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--include_images', action='store_true', help='Include molecular images')
    
    args = parser.parse_args()
    
    # Load configs
    vlm_cfg = VLMConfig()
    train_cfg = load_spectral_config(args.config)
    
    # Override from command line
    if args.data_dir:
        train_cfg.data.data_dir = args.data_dir
    if args.lr_adamw:
        train_cfg.training.learning_rate = args.lr_adamw
    if args.lr_muon:
        train_cfg.optimizer.muon.lr = args.lr_muon
    if args.batch_size:
        train_cfg.training.batch_size = args.batch_size
    if args.epochs:
        train_cfg.training.epochs = args.epochs
    if args.compile:
        train_cfg.training.compile = args.compile
    if args.include_images:
        train_cfg.model.include_images = args.include_images
    
    # Initialize distributed training
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
    
    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Spectral Training Config ---")
        config_dict = train_cfg.to_dict()
        for section, values in config_dict.items():
            print(f"\n{section}:")
            if isinstance(values, dict):
                for k, v in values.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {values}")
    
    train(train_cfg, vlm_cfg)
    
    if is_dist():
        destroy_dist()


if __name__ == "__main__":
    main() 