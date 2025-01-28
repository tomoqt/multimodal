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

def evaluate_predictions(predictions, targets, verbose=False):
    """
    Evaluate model predictions vs. targets using canonical SMILES.
    Returns a list of dictionaries, each containing metrics and info
    about a single (prediction, target) pair.
    """
    detailed_results = []
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        result = {
            'prediction': pred,
            'target': target,
            'valid': False,
            'valid_target': False,
            'exact_match': False,
            'tanimoto': 0.0,
            '#mcs/#target': 0.0,
            'ecfp6_iou': 0.0
        }
        
        # Remove spaces before creating molecules
        pred_no_spaces = pred.replace(" ", "")
        target_no_spaces = target.replace(" ", "")
        
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

            if verbose and i < 5:  # Print first 5 examples
                print(f"\nCanonical SMILES comparison:")
                print(f"Target (canonical):     {canon_target}")
                print(f"Prediction (canonical): {canon_pred}")
        
        if verbose and i < 5:  # Print first 5 examples
            print(f"\nExample {i+1}:")
            print(result)
            
        detailed_results.append(result)
        
    return detailed_results

def aggregate_metrics(detailed_results):
    """
    Aggregates molecular metrics from detailed results.
    """
    metrics = {
        'valid_smiles': np.mean([r['valid'] for r in detailed_results]),
        'exact_match': 0.0,
        'exact_match_all': 0.0,
        'avg_tanimoto': 0.0,
        'avg_ecfp6_iou': 0.0,
        'avg_#mcs/#target': 0.0
    }
    
    # Calculate exact match over all examples
    metrics['exact_match_all'] = np.mean([
        (r['valid'] and r['valid_target'] and r['exact_match']) 
        for r in detailed_results
    ])
    
    # Only consider pairs where both SMILES were valid
    valid_pairs = [r for r in detailed_results if r['valid'] and r['valid_target']]
    if valid_pairs:
        metrics['exact_match'] = np.mean([r['exact_match'] for r in valid_pairs])
        metrics['avg_tanimoto'] = np.mean([r['tanimoto'] for r in valid_pairs])
        metrics['avg_ecfp6_iou'] = np.mean([r['ecfp6_iou'] for r in valid_pairs])
        metrics['avg_#mcs/#target'] = np.mean([r['#mcs/#target'] for r in valid_pairs])

    return metrics

def log_results(val_metrics, step, table=None, prefix=None):
    """
    Log validation metrics and matching SMILES pairs to wandb.
    """
    if table is not None and "valid_set" in val_metrics:
        # Add matching pairs to the table
        for pair in val_metrics['valid_set']:
            table.add_data(step, *list(pair.values()))
        val_metrics["matching_pairs_table"] = table

    # Create a copy to avoid modifying the original dict
    log_dict = deepcopy(val_metrics)
    
    # Remove the original matching_pairs to avoid redundancy
    if "valid_set" in log_dict:
        del log_dict["valid_set"]

    # Add prefix to metrics if specified
    if prefix:
        log_dict = {f'{prefix}_{k}': v for k, v in log_dict.items()}

    # Log everything to W&B
    wandb.log(log_dict, step=step) 