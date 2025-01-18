import time
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sys
import os

def load_vocabulary(vocab_file):
    """
    Load vocabulary from a text file, one token per line.
    """
    if not os.path.exists(vocab_file):
        print(f"Vocabulary file '{vocab_file}' not found.")
        sys.exit(1)
    with open(vocab_file, 'r') as file:
        vocab = [line.strip() for line in file if line.strip()]
    return vocab

def load_initial_smiles(initial_file=False):
    """
    Load initial SMILES from a text file or define them within the script.
    """
    if initial_file:
        if not os.path.exists(initial_file):
            print(f"Initial SMILES file '{initial_file}' not found.")
            sys.exit(1)
        with open(initial_file, 'r') as file:
            initial_smiles = [line.strip() for line in file if line.strip()]
    else:
        # Define initial SMILES within the script
        initial_smiles = [
            'C',          # Methane
            'CC',         # Ethane
            'CCC',        # Propane
            'CCCC',       # Butane
            'CCO',        # Ethanol
            'c1ccccc1',   # Benzene
            'C1CCCCC1',   # Cyclohexane
            'CC(=O)O',    # Acetic acid
            'C1=CC=CN=C1',# Pyridine
            'CCN(CC)CC',  # Triethylamine
            'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC',  # Very long carbon chain
            'C1CC1C2CC2C3CC3C4CC4C5CC5C6CC6C7CC7C8CC8C9CC9C%10CC%10',  # Multiple rings
            'C1=CC=C2C(=C1)C=CC=C2C3=CC=CC=C3C4=CC=CC=C4',  # Polycyclic aromatic hydrocarbons
            'CC(C)(C)C(=O)OC1=CC=CC=C1C2=CC=CC=C2',  # Tert-butyl acetate with benzene rings
            'C[C@H](N)C(=O)OCC(C)C1=CC=C(C=C1)O',  # Amino acid derivative with a benzene ring
            'C1CCCCC1C2=CC=CC=C2C3=CC=CC=C3C4=CC=CC=C4',  # Cyclohexane with multiple benzene rings
            'CN(C)C(=O)C1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3',  # Triethylamine with multiple aromatic rings
            'O=C(OCC(C)C)C1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3',  # Ester with multiple benzene rings
            'CC(C)(C)C(C)(C)C1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=C(C=C3)O',  # Highly substituted aromatic compound
            'C1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=C(C=C3)C4=CC=C(C=C4)O',  # Multi-substituted phenol
            # Add more longer SMILES as needed
        ]
    return initial_smiles

def is_valid_smiles(smiles):
    """
    Check if a SMILES string is valid using RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main():
    # Configuration
    vocab_file = 'vocab.txt'          # Path to your vocabulary file
    initial_file = False # Path to initial SMILES file (optional)
    # If initial_file is None or not provided, initial SMILES are defined within the script

    # Load vocabulary
    vocab = load_vocabulary(vocab_file)
    print(f"Loaded {len(vocab)} vocabulary items from '{vocab_file}'.\n")

    # Load initial SMILES
    initial_smiles = load_initial_smiles(initial_file)
    print(f"Loaded {len(initial_smiles)} initial SMILES.\n")

    # Initialize summary statistics
    summary = []

    # Iterate over each initial SMILES
    for i, initial in enumerate(initial_smiles, 1):
        print(f"Processing Initial SMILES {i}/{len(initial_smiles)}: {initial}")
        if not is_valid_smiles(initial):
            print(f"Warning: Initial SMILES '{initial}' is invalid. Skipping.\n")
            continue

        start_time = time.time()

        valid_smiles = []
        invalid_smiles_count = 0
        total_combinations = len(vocab)

        for j, continuation in enumerate(vocab, 1):
            new_smiles = initial + continuation
            if is_valid_smiles(new_smiles):
                valid_smiles.append(new_smiles)
                # Uncomment the following line to print each valid SMILES
                # print(f"  Valid: {new_smiles}")
            else:
                invalid_smiles_count += 1
                # Uncomment the following line to print each invalid SMILES
                # print(f"  Invalid: {new_smiles}")

            # Optional: Progress indicator for large vocabularies
            if j % 1000 == 0 or j == total_combinations:
                print(f"  Processed {j}/{total_combinations} combinations.")

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Collect summary data
        summary.append({
            'initial_smiles': initial,
            'valid_count': len(valid_smiles),
            'invalid_count': invalid_smiles_count,
            'total': total_combinations,
            'time_sec': elapsed_time
        })

        # Print results for this initial SMILES
        print(f"  Valid SMILES: {len(valid_smiles)}")
        print(f"  Invalid SMILES: {invalid_smiles_count}")
        print(f"  Time taken: {elapsed_time:.4f} seconds\n")

    # Summary of all initial SMILES
    print("\n=== Overall Summary ===")
    total_checked = 0
    total_valid = 0
    total_invalid = 0
    total_time = 0.0

    for entry in summary:
        print(f"Initial SMILES: {entry['initial_smiles']}")
        print(f"  Valid: {entry['valid_count']}")
        print(f"  Invalid: {entry['invalid_count']}")
        print(f"  Time: {entry['time_sec']:.4f} seconds\n")
        total_checked += entry['total']
        total_valid += entry['valid_count']
        total_invalid += entry['invalid_count']
        total_time += entry['time_sec']

    print("--- Aggregate Statistics ---")
    print(f"Total initial SMILES processed: {len(summary)}")
    print(f"Total combinations checked: {total_checked}")
    print(f"Total valid SMILES: {total_valid}")
    print(f"Total invalid SMILES: {total_invalid}")
    print(f"Total time taken: {total_time:.4f} seconds")
    print(f"Average time per initial SMILES: {total_time / len(summary) if summary else 0:.4f} seconds")

if __name__ == "__main__":
    main()
