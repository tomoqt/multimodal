#!/usr/bin/env python3
"""
Generation script for SpectralVLM
Tests the tri-modal model with sample spectroscopic data
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import wandb

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from nanoVLM.models.spectral_vision_language_model import SpectralVisionLanguageModel
from training.data.processors import get_tokenizer, get_image_processor, extract_smiles_from_response
from PIL import Image


def create_sample_data():
    """Create sample spectral data for testing"""
    
    # Sample NMR data (chemical shift values)
    sample_nmr = "7.25 7.30 d 2H 7.15 7.20 d 2H 4.35 4.40 q 1H 3.85 s 3H 1.35 1.40 d 3H"
    
    # Sample IR data (simulated 1000-point spectrum)
    sample_ir = np.random.rand(1000).astype(np.float32)
    
    # Expected SMILES for the sample
    expected_smiles = "COc1ccc(C(C)O)cc1"
    
    return sample_nmr, sample_ir, expected_smiles


def create_prompt(nmr_data, include_target=False, target_smiles=None):
    """Create XML-formatted prompt for the model"""
    
    prompt = f"""Analyze the following spectroscopic data and predict the molecular structure.

<nmr_data>{nmr_data}</nmr_data>

Please provide the SMILES representation of the molecule contained in this format : "<smiles>[PREDICTED SMILES]</smiles>" :"""
    
    if include_target and target_smiles:
        prompt += f"\n<smiles>{target_smiles}</smiles>" #must be on for pretraining, appended to the last so it's causal. 
    
    return prompt


def main():
    parser = argparse.ArgumentParser(description='Generate SMILES from spectral data using SpectralVLM')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained SpectralVLM checkpoint')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to run inference on')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                       help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--greedy', action='store_true',
                       help='Use greedy decoding instead of sampling')
    parser.add_argument('--nmr_data', type=str, default=None,
                       help='Custom NMR data to analyze')
    parser.add_argument('--ir_data_path', type=str, default=None,
                       help='Path to IR data file (.npy)')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to molecular image')
    parser.add_argument('--log_wandb', action='store_true', help='Log results to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default="spectral_vlm_generation", help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team)')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    
    args = parser.parse_args()

    if args.log_wandb:
        run_name = args.run_name if args.run_name else f"spectral_gen_{Path(args.model_path).stem}_{int(torch.cuda.initial_seed()) if torch.cuda.is_available() else np.random.randint(1000)}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=args)
        print(f"Logging to WandB: project='{args.wandb_project}', entity='{args.wandb_entity}', run_name='{run_name}'")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.model_path}...")
    model = SpectralVisionLanguageModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    tokenizer = get_tokenizer()
    print("Model and tokenizer loaded successfully!")
    
    sample_nmr_data = None
    sample_ir_data_used = False
    expected_smiles_str = "N/A"

    if args.nmr_data:
        nmr_data = args.nmr_data
        print(f"Using custom NMR data: {nmr_data}")
    else:
        sample_nmr_data, sample_ir, expected_smiles_str = create_sample_data()
        nmr_data = sample_nmr_data
        print(f"Using sample NMR data: {nmr_data}")
        print(f"Expected SMILES (for sample): {expected_smiles_str}")
    
    ir_data_tensor = None
    ir_data_shape_log = "N/A"
    ir_data_used = False
    if args.ir_data_path:
        try:
            loaded_ir_data = np.load(args.ir_data_path).astype(np.float32)
            ir_data_tensor = torch.tensor(loaded_ir_data).unsqueeze(0).to(device)  # Add batch dimension
            ir_data_shape_log = str(ir_data_tensor.shape)
            print(f"Loaded IR data from {args.ir_data_path} with shape: {ir_data_shape_log}")
            ir_data_used = True
        except Exception as e:
            print(f"Warning: Could not load IR data from {args.ir_data_path}: {e}")
    elif not args.nmr_data:  # Use sample IR data if using sample NMR and no custom IR path
        ir_data_tensor = torch.tensor(sample_ir).unsqueeze(0).to(device)
        ir_data_shape_log = str(ir_data_tensor.shape)
        print(f"Using sample IR data with shape: {ir_data_shape_log}")
        sample_ir_data_used = True # Specifically for sample case
        ir_data_used = True

    image_data_tensor = None
    image_shape_log = "N/A"
    image_used = False
    if args.image_path:
        try:
            image_processor = get_image_processor()
            img = Image.open(args.image_path).convert('RGB')
            image_data_tensor = image_processor(img).unsqueeze(0).to(device)  # Add batch dimension
            image_shape_log = str(image_data_tensor.shape)
            print(f"Loaded image from {args.image_path} with shape: {image_shape_log}")
            image_used = True
        except Exception as e:
            print(f"Warning: Could not load image from {args.image_path}: {e}")
    
    prompt_text = create_prompt(nmr_data)
    print(f"\n--- Input Prompt ---")
    print(prompt_text)
    print(f"--------------------\n")
    
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print("Generating SMILES...")
    print(f"Generation parameters: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, greedy={args.greedy}")
    
    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=input_ids,
            image=image_data_tensor,
            ir_data=ir_data_tensor,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            greedy=args.greedy
        )
    
    generated_text_full = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"\n--- Generated Response (Full) ---")
    print(generated_text_full)
    print(f"---------------------------------\n")
    
    predicted_smiles_str = extract_smiles_from_response(generated_text_full)
    print(f"Predicted SMILES: {predicted_smiles_str}")
    
    log_data = {
        "model_path": args.model_path,
        "nmr_data_input": nmr_data,
        "ir_data_path": args.ir_data_path if args.ir_data_path else ("Sample IR" if sample_ir_data_used else "None"),
        "ir_data_shape": ir_data_shape_log,
        "ir_data_used": ir_data_used,
        "image_path": args.image_path if args.image_path else "None",
        "image_shape": image_shape_log,
        "image_used": image_used,
        "prompt": prompt_text,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "greedy_decoding": args.greedy,
        "generated_full_text": generated_text_full,
        "predicted_smiles": predicted_smiles_str,
        "expected_smiles": expected_smiles_str if not args.nmr_data else "N/A (custom input)",
    }

    if not args.nmr_data: # This means we are using the default sample data
        match_status = predicted_smiles_str.strip() == expected_smiles_str.strip()
        print(f"Expected SMILES (Sample): {expected_smiles_str}")
        print(f"Match with Expected (Sample): {match_status}")
        log_data["match_with_expected_sample"] = match_status
    
    print(f"\n--- SMILES Validation ---")
    print(f"Length: {len(predicted_smiles_str)}")
    contains_typical_chars = any(c in predicted_smiles_str for c in 'CNOSPcno()[]=#@+-%')
    print(f"Contains typical SMILES characters: {contains_typical_chars}")
    log_data["smiles_length"] = len(predicted_smiles_str)
    log_data["smiles_typical_chars"] = contains_typical_chars
    
    rdkit_validation_status = "N/A (RDKit not installed)"
    canonical_smiles_rdkit = "N/A"
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(predicted_smiles_str.strip()) # Strip whitespace for RDKit
        if mol is not None:
            canonical_smiles_rdkit = Chem.MolToSmiles(mol, canonical=True)
            rdkit_validation_status = "VALID"
            print(f"RDKit validation: VALID")
            print(f"Canonical SMILES (RDKit): {canonical_smiles_rdkit}")
        else:
            rdkit_validation_status = "INVALID"
            print(f"RDKit validation: INVALID")
    except ImportError:
        print("RDKit not available for validation.")
    except Exception as e:
        rdkit_validation_status = f"FAILED ({e})"
        print(f"RDKit validation failed: {e}")
    
    log_data["rdkit_validation"] = rdkit_validation_status
    log_data["rdkit_canonical_smiles"] = canonical_smiles_rdkit

    if args.log_wandb and wandb.run:
        wandb.log(log_data)
        print("\nResults logged to WandB.")

    if args.log_wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main() 