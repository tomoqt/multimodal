# ======================
# File: logging_utils.py
# ======================
"""
Utility functions for molecular SMILES logging and evaluation.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS
import wandb
from copy import deepcopy
import selfies as sf
import re
from models.smiles_utils import process_smiles_to_selfies

def is_selfies(s: str) -> bool:
    """
    More robust check for SELFIES format.
    SELFIES tokens are always in the format [xxx] with no unclosed brackets.
    """
    if not (s.startswith('[') and s.endswith(']')):
        return False
    # Check if it follows SELFIES token pattern
    tokens = re.findall(r'\[[^\]]*\]', s)
    # Reconstruct the string from tokens and compare
    reconstructed = ''.join(tokens)
    return reconstructed == s

def evaluate_predictions(predictions, targets, verbose=False):
    """
    Evaluate model predictions vs. targets.
    Assumes inputs are already in SELFIES format.
    Performs both direct SELFIES comparison and SMILES-based metrics.
    """
    detailed_results = []
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        result = {
            'prediction_original': pred,
            'target_original': target,
            'valid': False,
            'valid_target': False,
            'exact_match': False,
            'selfies_exact_match': False,  # New metric for direct SELFIES comparison
            'tanimoto': 0.0,
            '#mcs/#target': 0.0,
            'ecfp6_iou': 0.0
        }
        
        # Direct SELFIES comparison
        result['selfies_exact_match'] = (pred == target)
            
        # Convert to SMILES for chemical comparison
        try:
            pred_smiles = sf.decoder(pred)
            target_smiles = sf.decoder(target)
            
            result['prediction'] = pred_smiles
            result['target'] = target_smiles
            
        except Exception as e:
            if verbose:
                print(f"Error converting to SMILES: {e}")
            continue

        # Remove spaces before creating molecules
        pred_no_spaces = pred_smiles.replace(" ", "")
        target_no_spaces = target_smiles.replace(" ", "")
        
        # Convert to RDKit molecules
        mol_pred = Chem.MolFromSmiles(pred_no_spaces)
        mol_target = Chem.MolFromSmiles(target_no_spaces)

        result['valid'] = mol_pred is not None
        result['valid_target'] = mol_target is not None

        if result['valid'] and result['valid_target']:
            # Get canonical SMILES
            canon_pred = Chem.MolToSmiles(mol_pred, canonical=True)
            canon_target = Chem.MolToSmiles(mol_target, canonical=True)
            result['exact_match'] = canon_pred == canon_target
            result['canonical_pred'] = canon_pred
            result['canonical_target'] = canon_target

            # Standard Tanimoto similarity
            fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2)
            fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, 2)
            tanimoto = DataStructs.TanimotoSimilarity(fp_pred, fp_target)
            result['tanimoto'] = tanimoto

            # ECFP6 IoU
            fp3_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, radius=3, nBits=1024)
            fp3_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, radius=3, nBits=1024)
            intersection = sum((fp3_pred & fp3_target))
            union = sum((fp3_pred | fp3_target))
            result['ecfp6_iou'] = intersection / union if union > 0 else 0.0

            # MCS calculation
            mcs_result = rdFMCS.FindMCS([mol_pred, mol_target])
            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            if mcs_mol:
                mcs_num_atoms = mcs_mol.GetNumAtoms()
                target_num_atoms = mol_target.GetNumAtoms()
                result['#mcs/#target'] = mcs_num_atoms / target_num_atoms if target_num_atoms > 0 else 0.0

        if verbose and i < 5:
            print(f"\nExample {i+1}:")
            print(f"Original SELFIES/SMILES: {pred}")
            print(f"Valid molecule: {result['valid']}")
            if result['valid'] and result['valid_target']:
                print(f"Exact SMILES match: {result['exact_match']}")
                print(f"Canonical SMILES: {canon_pred}")
            
        detailed_results.append(result)
        
    return detailed_results

def aggregate_metrics(detailed_results):
    """
    Aggregates molecular metrics from detailed results.
    Includes both SMILES-based and direct SELFIES comparison metrics.
    """
    metrics = {
        'valid_smiles': np.mean([r['valid'] for r in detailed_results]),
        'selfies_exact_match': np.mean([r['selfies_exact_match'] for r in detailed_results]),  # Direct SELFIES comparison
        'exact_match': 0.0,
        'avg_tanimoto': 0.0,
        'avg_ecfp6_iou': 0.0,
        'avg_#mcs/#target': 0.0
    }
    
    # Only consider pairs where both molecules were valid
    valid_pairs = [r for r in detailed_results if r['valid'] and r['valid_target']]
    if valid_pairs:
        metrics['exact_match'] = np.mean([r['exact_match'] for r in valid_pairs])
        metrics['avg_tanimoto'] = np.mean([r['tanimoto'] for r in valid_pairs])
        metrics['avg_ecfp6_iou'] = np.mean([r['ecfp6_iou'] for r in valid_pairs])
        metrics['avg_#mcs/#target'] = np.mean([r['#mcs/#target'] for r in valid_pairs])

    return metrics

def log_results(val_metrics, step, table=None, prefix=None):
    """
    Log validation metrics and matching pairs to wandb.
    Now handles both SELFIES and SMILES formats.
    """
    if table is not None and "predictions" in val_metrics:
        # Add matching pairs to the table
        for pred, tgt in zip(val_metrics['predictions'][:10], val_metrics['targets'][:10]):
            # Convert to SMILES for logging if needed
            try:
                if '[' in pred and ']' in pred and any(x in pred for x in ['Branch', 'Ring', 'expl']):
                    pred_smiles = sf.decoder(pred)
                else:
                    pred_smiles = pred

                if '[' in tgt and ']' in tgt and any(x in tgt for x in ['Branch', 'Ring', 'expl']):
                    tgt_smiles = sf.decoder(tgt)
                else:
                    tgt_smiles = tgt

                table.add_data(
                    step,
                    pred_smiles,  # Log SMILES version for readability
                    tgt_smiles,   # Log SMILES version for readability
                    pred,         # Also log original SELFIES
                    tgt          # Also log original SELFIES
                )
            except Exception as e:
                print(f"Error converting for logging: {e}")
                continue

    # Create a copy to avoid modifying the original dict
    log_dict = deepcopy(val_metrics)
    
    # Add prefix to metrics if specified
    if prefix:
        log_dict = {f'{prefix}_{k}': v for k, v in log_dict.items()}

    # Log everything to W&B
    wandb.log(log_dict, step=step) 