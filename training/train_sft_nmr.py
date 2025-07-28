import argparse
import os
import sys
from pprint import pprint
import wandb  # Added for explicit logging

import torch
from datasets import Dataset, load_dataset, load_from_disk
from rdkit import Chem, RDLogger
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

# Disable RDKit logging
RDLogger.DisableLog("rdApp.*")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a model on NMR data using SFTTrainer.")
    parser.add_argument("--model_name", type=str, default="futurehouse/ether0", help="The pre-trained model to fine-tune.")
    parser.add_argument("--tokenized_data_dir", type=str, required=True, help="Directory containing the pre-tokenized data files (output of pretokenize_sft_data.py).")
    parser.add_argument("--output_dir", type=str, default="checkpoints_sft_nmr", help="Directory to save checkpoints and final model.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    
    # New arguments for Hugging Face Hub
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the Hugging Face Hub after training.")
    parser.add_argument("--hf_repo_id", type=str, default=None, help="The repository ID on the Hugging Face Hub (e.g., 'your-username/your-model').")
    parser.add_argument("--hf_token", type=str, default=None, help="The Hugging Face hub token. If not set, will use HUGGING_FACE_HUB_TOKEN env var or cached token.")
    # Verbose/debug flag
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging and print example samples.")
    
    # Dataset slicing arguments
    parser.add_argument("--max_train_samples", type=int, default=None, help="Number of training samples to use (from the beginning). If None, use all.")
    parser.add_argument("--max_eval_samples", type=int, default=1000, help="Maximum number of samples for final evaluation. If None or 0, use all.")

    # PEFT arguments
    parser.add_argument("--use_peft", action="store_true", help="Enable PEFT for fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout parameter.")

    args = parser.parse_args()
    # For SFTTrainer, we don't need to pass this if the data is already tokenized and truncated
    # but our evaluate_model function might need it. Let's keep it.
    # args.max_seq_length = tokenizer.model_max_length 
    return args

def canonicalize_smiles(smiles):
    """Convert a SMILES string to its canonical form using RDKit. Removes extra spaces before conversion. Returns the canonical SMILES if possible, otherwise returns the cleaned string."""
    # Remove spaces and strip leading/trailing whitespace
    cleaned = smiles.replace(' ', '').strip()
    try:
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return cleaned
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return cleaned

def evaluate_predictions(predictions, targets):
    """
    Computes detailed metrics for predictions against targets.
    - Valid SMILES rate
    - Exact match rate
    - Tanimoto similarity
    - MCS (Maximum Common Substructure) based metrics
    - ECFP6 fingerprint based IoU (Jaccard)
    """
    from rdkit.Chem import AllChem, DataStructs, rdFMCS

    results = []
    for pred, target in zip(predictions, targets):
        pred_mol = Chem.MolFromSmiles(pred)
        target_mol = Chem.MolFromSmiles(target)
        
        metrics = {
            "prediction": pred,
            "target": target,
            "valid_smiles": 1 if pred_mol else 0,
            "exact_match": 0,
            "tanimoto": 0.0,
            "#mcs/#target": 0.0,
            "ecfp6_iou": 0.0,
        }

        if pred_mol and target_mol:
            # Exact Match
            if pred == target:
                metrics["exact_match"] = 1

            # Tanimoto Similarity
            fp_pred = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048)
            fp_target = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
            metrics["tanimoto"] = DataStructs.TanimotoSimilarity(fp_pred, fp_target)

            # MCS Ratio
            mcs_result = rdFMCS.FindMCS([pred_mol, target_mol], timeout=1)
            if mcs_result.numAtoms > 0:
                mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                if mcs_mol:
                    metrics["#mcs/#target"] = mcs_mol.GetNumAtoms() / target_mol.GetNumAtoms()
            
            # ECFP6 IoU
            ecfp6_pred = AllChem.GetMorganFingerprint(pred_mol, 3)
            ecfp6_target = AllChem.GetMorganFingerprint(target_mol, 3)
            intersection = len(set(ecfp6_pred.GetNonzeroElements()) & set(ecfp6_target.GetNonzeroElements()))
            union = len(set(ecfp6_pred.GetNonzeroElements()) | set(ecfp6_target.GetNonzeroElements()))
            if union > 0:
                metrics["ecfp6_iou"] = intersection / union

        results.append(metrics)
    return results

def aggregate_metrics(detailed_results):
    """Aggregates detailed evaluation results into summary metrics."""
    if not detailed_results:
        return {}

    total = len(detailed_results)
    valid_smiles = sum(r["valid_smiles"] for r in detailed_results) / total
    exact_match = sum(r["exact_match"] for r in detailed_results) / total
    
    # Only calculate averages for valid SMILES pairs
    valid_pairs = [r for r in detailed_results if r["valid_smiles"] and Chem.MolFromSmiles(r["target"])]
    if valid_pairs:
        avg_tanimoto = sum(r["tanimoto"] for r in valid_pairs) / len(valid_pairs)
        avg_mcs_ratio = sum(r["#mcs/#target"] for r in valid_pairs) / len(valid_pairs)
        avg_ecfp6_iou = sum(r["ecfp6_iou"] for r in valid_pairs) / len(valid_pairs)
    else:
        avg_tanimoto = 0.0
        avg_mcs_ratio = 0.0
        avg_ecfp6_iou = 0.0

    return {
        "valid_smiles": valid_smiles,
        "exact_match": exact_match,
        "avg_tanimoto": avg_tanimoto,
        "avg_#mcs/#target": avg_mcs_ratio,
        "avg_ecfp6_iou": avg_ecfp6_iou
    }

def compute_metrics(eval_pred, tokenizer, prompt_template):
    """Computes metrics for SFTTrainer."""
    predictions, labels = eval_pred
    
    # Decode predictions. The model outputs the full sequence including prompt.
    # We set skip_special_tokens=True to remove padding and EOS tokens.
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Decode labels, replacing -100 with pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # The prompt part to be removed. We only need the part that is constant.
    # The NMR data is variable. The template is "Given...SMILES:"
    prompt_end = prompt_template.split("{nmr_data}")[1].strip() # This gives "\n\nSMILES:"

    # Extract only the completion part
    preds_completion = [pred.split(prompt_end)[-1].strip() for pred in decoded_preds]
    labels_completion = [label.split(prompt_end)[-1].strip() for label in decoded_labels]
    
    # Canonicalize SMILES strings
    canon_preds = [canonicalize_smiles(s) for s in preds_completion]
    canon_labels = [canonicalize_smiles(s) for s in labels_completion]

    # Compute metrics
    detailed_results = evaluate_predictions(canon_preds, canon_labels)
    metrics = aggregate_metrics(detailed_results)

    return metrics

def evaluate_model(model, tokenizer, dataset, batch_size: int = 4, max_new_tokens: int = 128):
    """Run model.generate on prompts in `dataset` and compute evaluation metrics.

    This function is intended to be called outside the training loop so that
    evaluation results reflect inference-time behaviour. Metrics are computed
    with the same helpers already defined in the script.
    
    NOTE: This function expects a dataset with a 'text' field, which is not
    available if you use pre-tokenized data. This will need adjustment if
    evaluation during training is needed with this exact function. For now,
    the primary goal is to fix the training timeout.
    """

    model.eval()
    prompt_end = "\n\nSMILES:"

    preds, targets = [], []

    # This part will fail if the dataset does not have a "text" column.
    # The pre-tokenized dataset will not have it.
    # We will adjust this to work with the tokenized data by decoding it first.
    # This is inefficient but makes the function work without major refactoring.

    texts_for_eval = []
    if "text" in dataset.column_names:
         texts_for_eval = dataset["text"]
    else:
        # Reconstruct the text from tokens for evaluation
        print("Reconstructing text from tokens for evaluation. This might be slow.")
        decoded_samples = tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        texts_for_eval = decoded_samples


    for start_idx in range(0, len(texts_for_eval), batch_size):
        batch_texts = texts_for_eval[start_idx : start_idx + batch_size]
        
        prompts, batch_targets = [], []
        for txt in batch_texts:
            if prompt_end not in txt:
                # Skip malformed sample
                continue
            prompt_part, target_part = txt.split(prompt_end, 1)
            prompts.append(prompt_part + prompt_end)  # keep delimiter
            batch_targets.append(target_part.strip())

        if not prompts:
            continue

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        batch_preds = [d.split(prompt_end)[-1].strip() for d in decoded]

        preds.extend(batch_preds)
        targets.extend(batch_targets)

    # Canonicalize SMILES prior to metric computation
    canon_preds = [canonicalize_smiles(s) for s in preds]
    canon_targets = [canonicalize_smiles(s) for s in targets]

    detailed = evaluate_predictions(canon_preds, canon_targets)
    return aggregate_metrics(detailed)

def main():
    """Main training function."""
    args = parse_args()

    print("--- Configuration ---")
    pprint(vars(args))
    print("---------------------")
    
    if args.push_to_hub and not args.hf_repo_id:
        print("Error: --hf_repo_id is required when --push_to_hub is set.", file=sys.stderr)
        sys.exit(1)

    # 1. Load pre-tokenized dataset
    print(f"Loading pre-tokenized dataset from {args.tokenized_data_dir}")
    train_dataset = load_from_disk(os.path.join(args.tokenized_data_dir, "train"))
    val_dataset = load_from_disk(os.path.join(args.tokenized_data_dir, "val"))
    
    if args.max_train_samples is not None and args.max_train_samples > 0:
        print(f"Using the first {args.max_train_samples} samples for training.")
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))

    print("Dataset loaded and formatted:")
    print(train_dataset)
    print(val_dataset)
    print("Example data point:")
    print(train_dataset[0])
    
    # ------------------------------------------------------------------
    # Verbose mode: print a handful of formatted samples for inspection
    # ------------------------------------------------------------------
    if args.verbose:
        print("\n--- Verbose sample inspection (first 3 training samples) ---")
        for i in range(min(3, len(train_dataset))):
            sample_text = train_dataset[i]["text"]
            print(f"Sample {i}:\n{sample_text}\n{'-'*80}")

    # 2. Load model and tokenizer
    print(f"Loading model and tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        
    # Set padding side to 'left' for decoder-only models to ensure correct generation
    tokenizer.padding_side = 'left'
        
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # 3. Configure PEFT if requested
    peft_config = None
    if args.use_peft:
        print("PEFT enabled. Using LoRA configuration.")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    # 4. Configure SFT training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hf_repo_id,
        hub_token=args.hf_token,
        remove_unused_columns=False,
        report_to=["wandb"],  # Explicitly report to Weights & Biases
    )

    # 5. Initialize SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        # SFTTrainer will automatically use the tokenized columns.
        # We don't specify `dataset_text_field`.
        # `max_seq_length` is also not needed here as data is pre-truncated,
        # but it's passed to the trainer which might use it for other purposes.
        #max_seq_length=args.max_seq_length,
    )

    # Handle PEFT+FSDP case, inspired by train_grpo_nmr.py
    if getattr(trainer.accelerator.state, "fsdp_plugin", None) and peft_config:
        from peft.utils.other import fsdp_auto_wrap_policy
        print("Applying FSDP auto wrap policy for PEFT")
        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    # 6. Start training
    print("Starting training...")
    trainer.train()

    # 7. Save final model locally
    print("Training finished. Saving final model.")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    # 8. Separate evaluation using `model.generate` and explicit W&B logging
    print("Running evaluation with model.generate ...")

    # To ensure a clean state for evaluation, especially with FSDP,
    # we load the model we just saved to disk. This gives us a regular,
    # non-sharded model that is easy to work with. We only do this on the
    # main process to avoid redundant work and logging issues.
    
    is_main_process = trainer.is_world_process_zero()

    # Free up memory before loading new model
    del model
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    if is_main_process:
        # Re-import peft for loading, if needed
        if args.use_peft:
            from peft import PeftModel
            print(f"Loading base model ({args.model_name}) for PEFT evaluation...")
            # Use a memory-efficient dtype for loading
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
            print(f"Loading PEFT adapters from {args.output_dir}...")
            eval_model = PeftModel.from_pretrained(base_model, args.output_dir)
            print("Merging PEFT adapters...")
            eval_model = eval_model.merge_and_unload()
            print("PEFT adapters merged.")
        else:
            print(f"Loading fully fine-tuned model from {args.output_dir}...")
            eval_model = AutoModelForCausalLM.from_pretrained(args.output_dir)

        # Move model to the correct device for this process
        device = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
        eval_model.to(device)

        # Select a subset of the validation set if requested
        eval_dataset_subset = val_dataset
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            print(f"Slicing validation set to a maximum of {args.max_eval_samples} samples for final evaluation.")
            eval_dataset_subset = val_dataset.select(range(min(args.max_eval_samples, len(val_dataset))))

        eval_metrics = evaluate_model(eval_model, tokenizer, eval_dataset_subset, batch_size=args.batch_size)
        print("Evaluation metrics:")
        pprint(eval_metrics)

        # Log metrics to wandb
        try:
            wandb.log(eval_metrics)
        except Exception as e:
            print(f"wandb logging failed: {e}")

if __name__ == "__main__":
    main()