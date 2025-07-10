import argparse
import os
import sys
from pprint import pprint

import torch
from datasets import Dataset
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Add project root to path so we can import utils if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param if all_param > 0 else 0}"
    )

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)

PROMPT_TEMPLATE = (
    SYSTEM_PROMPT
    + "\n\nUser: Given the following NMR data, predict the corresponding SMILES string.\n\n"
    + "NMR data: {nmr_data}\n\nAssistant:"
)

PROMPT_END = "Assistant:"

# -----------------------------------------------------------------------------
# Helper chemistry utilities (minimal subset to keep the script self-contained)
# -----------------------------------------------------------------------------

def canonicalize_smiles(smiles: str) -> str:
    """Return canonical RDKit SMILES or cleaned string if invalid."""
    s = smiles.replace(" ", "").strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    return Chem.MolToSmiles(mol, canonical=True)

def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    """Compute Tanimoto similarity between two SMILES. Returns 0 if invalid."""
    mol_a, mol_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if not mol_a or not mol_b:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)

# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------

def load_nmr_dataset(data_dir: str, split: str) -> Dataset:
    """Load src/tgt pairs and return Dataset with 'prompt' and 'target' fields."""
    src_path = os.path.join(data_dir, f"src-{split}.txt")
    tgt_path = os.path.join(data_dir, f"tgt-{split}.txt")
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        raise FileNotFoundError(
            f"Expected files src-{split}.txt and tgt-{split}.txt inside {data_dir}"
        )

    with open(src_path) as f:
        src_lines = [l.strip() for l in f]
    with open(tgt_path) as f:
        tgt_lines = [l.strip().replace(" ", "") for l in f]

    prompts, targets = [], []
    for nmr, tgt in zip(src_lines, tgt_lines):
        prompts.append(PROMPT_TEMPLATE.format(nmr_data=nmr))
        targets.append(tgt)

    return Dataset.from_dict({"prompt": prompts, "target": targets})

# -----------------------------------------------------------------------------
# Reward functions
# -----------------------------------------------------------------------------

def reward_format(completions, **kwargs):
    """Reward completions that have both <think>...</think> and <answer>...</answer> blocks."""
    rewards = []
    for c in completions:
        has_think = "<think>" in c and "</think>" in c
        has_answer = "<answer>" in c and "</answer>" in c
        rewards.append(1.0 if has_think and has_answer else -1.0)
    return rewards


def reward_tanimoto(completions, **kwargs):
    """Reward completions based on Tanimoto similarity to the reference target SMILES.

    The reference targets are expected to be passed by the trainer via **kwargs.
    We try a few common key names ("target", "targets", "reference", "samples").
    """
    # Extract potential target(s) from kwargs – they should align with completions length
    targets = None
    # Direct names
    for key in ["target", "targets", "reference", "references"]:
        if key in kwargs:
            targets = kwargs[key]
            break

    # Fallback to samples dict if provided
    if targets is None and "samples" in kwargs and isinstance(kwargs["samples"], dict):
        targets = kwargs["samples"].get("target")

    # If we still couldn't find targets, assign neutral reward
    if targets is None:
        return [0.0 for _ in completions]

    # Ensure targets is list-like and matches completions length
    if not isinstance(targets, (list, tuple)):
        targets = [targets] * len(completions)

    rewards = []
    for comp, tgt in zip(completions, targets):
        tgt_can = canonicalize_smiles(tgt)

        # Remove <think>...</think> and <answer>...</answer> wrappers
        comp_stripped = comp
        # Keep only text inside <answer> if present; else full string
        if "<answer>" in comp_stripped and "</answer>" in comp_stripped:
            comp_stripped = comp_stripped.split("<answer>")[-1].split("</answer>")[0]
        # Fallback to remove think tags
        comp_stripped = comp_stripped.replace("<think>", "").replace("</think>", "")
        comp_smiles = comp_stripped.strip()

        comp_can = canonicalize_smiles(comp_smiles)
        rewards.append(tanimoto_similarity(comp_can, tgt_can))

    return rewards

# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on NMR→SMILES using GRPOTrainer"
    )
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/reshaped_tokenized_data/data",
        help="Directory containing src-*.txt and tgt-*.txt pairs",
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints_grpo_nmr")
    parser.add_argument("--max_length", type=int, default=3000)
    parser.add_argument(
        "--batch_size",
        "--per_device_batch_size",
        dest="batch_size",
        type=int,
        default=2,
        help="Per-device batch size for training and evaluation.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL coefficient beta for GRPO (0 disables reference model)",
    )
    parser.add_argument("--number_of_generations", type = int , default = 2)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    # Hugging Face hub args
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_repo_id", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    # Prompt/completion length controls
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Maximum token length for the prompt passed to the model.")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="Maximum number of tokens the model can generate per completion.")
    # PEFT arguments (PEFT is on by default for GRPO)
    parser.add_argument("--full_finetune", action="store_true", help="Disable PEFT and run full fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout parameter.")
    # Verbose/debug flag
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging and print example samples.")

    # Dataset slicing arguments
    parser.add_argument("--max_train_samples", type=int, default=None, help="Number of training samples to use. If None, use all available after skipping.")
    parser.add_argument("--skip_train_samples", type=int, default=0, help="Number of samples to skip from the beginning of the training set.")

    # vLLM arguments
    parser.add_argument("--use_vllm", action="store_true", help="Enable vLLM for faster generation.")
    parser.add_argument("--vllm_server_host", type=str, default="localhost", help="Hostname for the vLLM server.")
    parser.add_argument("--vllm_server_port", type=int, default=8000, help="Port for the vLLM server.")
    #parser.add_argument("--vllm_server_endpoint", type=str, default="/v1/completions", help="Endpoint for the vLLM server.")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Evaluation helper (greedy generation)
# -----------------------------------------------------------------------------

def evaluate_model(model, tokenizer, dataset, batch_size=4, max_new_tokens=128):
    model.eval()
    preds, tgts = [], []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        prompts = batch["prompt"]
        tgts.extend(batch["target"])

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # Extract completion part
        comps = []
        for d in decoded:
            if PROMPT_END in d:
                comps.append(d.split(PROMPT_END)[-1].strip())
            else:
                comps.append(d)
        preds.extend(comps)

    # Compute average Tanimoto
    sims = [tanimoto_similarity(canonicalize_smiles(p), canonicalize_smiles(t)) for p, t in zip(preds, tgts)]
    return {"avg_tanimoto": sum(sims) / len(sims)}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    print("--- Configuration ---")
    pprint(vars(args))
    print("---------------------")

    # 1. Load dataset (prompts include system prompt automatically)
    print("Loading dataset ...")
    train_ds = load_nmr_dataset(args.data_dir, "train")
    val_ds = load_nmr_dataset(args.data_dir, "val")

    # Slice the dataset as requested
    if args.skip_train_samples > 0 or args.max_train_samples is not None:
        original_size = len(train_ds)
        
        start_index = args.skip_train_samples
        if start_index >= original_size:
            print(f"Warning: --skip_train_samples ({start_index}) is >= dataset size ({original_size}). Training set will be empty.")
            start_index = original_size
        
        end_index = original_size
        if args.max_train_samples is not None and args.max_train_samples > 0:
            end_index = start_index + args.max_train_samples
        
        selected_indices = range(start_index, min(end_index, original_size))
        
        print(f"Selecting training samples from index {selected_indices.start} to {selected_indices.stop -1} (original size: {original_size}).")
        train_ds = train_ds.select(selected_indices)

    # ------------------------------------------------------------------
    # Verbose mode: print a handful of formatted samples for inspection
    # ------------------------------------------------------------------
    if args.verbose:
        print("\n--- Verbose sample inspection (first 3 training samples) ---")
        for i in range(min(3, len(train_ds))):
            print(f"Prompt {i}: {train_ds[i]['prompt']}")
            print(f"Target {i}: {train_ds[i]['target']}\n{'-'*80}")

    # 2. Load model & tokenizer
    print(f"Loading model and tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to 'left' for decoder-only models to ensure correct generation
    tokenizer.padding_side = 'left'

    # Load the model first. If it's a PEFT checkpoint, from_pretrained will load the base
    # and apply the adapters. We also specify torch_dtype for FSDP compatibility.
    print(f"Loading model from {args.model_name}...")
    
    # Check if this is a PEFT checkpoint directory
    is_peft_checkpoint = False
    if os.path.isdir(args.model_name):
        adapter_config_path = os.path.join(args.model_name, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            is_peft_checkpoint = True
            print(f"Detected PEFT checkpoint: {adapter_config_path} exists")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    
    # Debug: Check initial model state
    print(f"Model type after loading: {type(model)}")
    print(f"Has peft_config: {hasattr(model, 'peft_config')}")
    print(f"Has merge_and_unload: {hasattr(model, 'merge_and_unload')}")
    
    # Determine if this is a genuine PEFT model
    is_peft_model = hasattr(model, 'peft_config') and hasattr(model, 'merge_and_unload')
    
    if hasattr(model, 'peft_config'):
        print(f"Active adapters: {list(model.peft_config.keys()) if model.peft_config else 'None'}")
        print(f"PEFT type: {type(model.peft_config)}")
    
    print(f"Is genuine PEFT model: {is_peft_model}")

    peft_config = None
    if not args.full_finetune:
        # If the loaded model doesn't have adapters, and we want PEFT, add them.
        if not is_peft_model:
            print("PEFT enabled (default). Applying new LoRA configuration.")
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            )
            from peft import get_peft_model
            model = get_peft_model(model, peft_config)
            print_trainable_parameters(model)
        else:
            print("Model is already a PEFT model (likely from SFT stage).")
            print("Merging existing adapters and applying new LoRA configuration for GRPO.")
            
            # Critical fix: Merge existing adapters first to avoid stacking
            if hasattr(model, 'merge_and_unload'):
                model = model.merge_and_unload()
                print("Existing adapters merged into base model.")
            else:
                print("Warning: Model has peft_config but no merge_and_unload method. This may be a regular model.")
                print("Proceeding to apply new LoRA configuration anyway.")
            
            # Now add fresh adapters for GRPO training
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            )
            from peft import get_peft_model
            model = get_peft_model(model, peft_config)
            print("New LoRA adapters applied for GRPO training.")
            print_trainable_parameters(model)
    else:
        print("Full fine-tuning is enabled.")
        # If the model has adapters, they must be merged before full fine-tuning.
        if is_peft_model:
            print("Merging PEFT adapters for full fine-tuning...")
            model = model.merge_and_unload()
            print("Adapters merged.")
        elif hasattr(model, "peft_config"):
            print("Warning: Model has peft_config but no merge_and_unload method. Proceeding with full fine-tuning.")

    # Final verification: Ensure all model parameters are ready for gradient computation
    print("\n=== Final Model Verification ===")
    print(f"Model type: {type(model)}")
    print(f"Has peft_config: {hasattr(model, 'peft_config')}")
    print(f"Model training mode: {model.training}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Count parameters by gradient requirement
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            trainable_params += 1
        else:
            frozen_params += 1
    
    print(f"Parameter summary: {total_params} total, {trainable_params} trainable, {frozen_params} frozen")
    
    if frozen_params > 0:
        print("⚠️  Warning: Some parameters are frozen. This might cause gradient computation issues.")
        # Enable gradients for all parameters as a safety measure
        for param in model.parameters():
            param.requires_grad = True
        print("✓ Enabled gradients for all parameters.")
    else:
        print("✓ All parameters are trainable.")

    # Ensure model is in training mode
    model.train()
    print("✓ Model set to training mode.")
    print("=== Verification Complete ===\n")

    # 3. Configure GRPO
    # ------------------------------------------------------------------
    # Respect model's absolute maximum sequence length to avoid CUDA/shape
    # errors during generation & log-prob computations.
    # ------------------------------------------------------------------
    model_max_len = getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 2048))

    desired_total_len = args.max_prompt_length + args.max_completion_length
    if desired_total_len > model_max_len:
        print(
            f"[WARNING] Requested max_prompt_length+max_completion_length ({desired_total_len}) exceeds model capability ({model_max_len}). "
            f"Reducing max_completion_length to keep total ≤ {model_max_len}."
        )
        args.max_completion_length = max(1, model_max_len - args.max_prompt_length)

    tokenizer.model_max_length = args.max_prompt_length + args.max_completion_length

    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hf_repo_id,
        hub_token=args.hf_token,
        beta=args.beta,
        remove_unused_columns=False,
        report_to=["wandb"],
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.number_of_generations,
        use_vllm=args.use_vllm,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        #vllm_server_endpoint=args.vllm_server_endpoint,
    )

    # 4. Instantiate trainer. We pass both reward functions.
    reward_fns = [reward_format, reward_tanimoto]

    # Ensure model is in training mode
    model.train()
    print(f"Model training mode: {model.training}")
    
    # Check model device
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    
    # Ensure model is on CUDA if available
    if torch.cuda.is_available() and device.type == 'cpu':
        print("Moving model to CUDA...")
        model = model.cuda()
        print(f"Model moved to: {next(model.parameters()).device}")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=reward_fns,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class = tokenizer #that's how it should be aaprently
    )

    # Handle PEFT+FSDP case
    if getattr(trainer.accelerator.state, "fsdp_plugin", None) and peft_config:
        from peft.utils.other import fsdp_auto_wrap_policy
        print("Applying FSDP auto wrap policy for PEFT")
        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    # 5. Train
    print("Starting GRPO training ...")
    trainer.train()

    # 6. Save model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(args.output_dir)

    # 7. Evaluate via generation
    eval_model = trainer.model
    # When doing full finetune, we don't need to merge since we already did it or never had adapters.
    if not args.full_finetune:
        try:
            # For evaluation, we merge the adapters into the base model
            if hasattr(trainer.model, 'merge_and_unload'):
                eval_model = trainer.model.merge_and_unload()
                print("Successfully merged PEFT adapters for evaluation.")
            else:
                print("Model does not have merge_and_unload method. Evaluating with current model state.")
        except Exception as e:
            print(f"Could not merge PEFT adapters: {e}. Evaluating with adapters loaded.")

    metrics = evaluate_model(
        eval_model,
        tokenizer,
        val_ds,
        batch_size=args.batch_size,
        max_new_tokens=args.max_completion_length,
    )
    print("Evaluation metrics:")
    pprint(metrics)
    try:
        import wandb

        wandb.log(metrics)
    except Exception as e:
        print(f"wandb logging failed: {e}")


if __name__ == "__main__":
    main() 