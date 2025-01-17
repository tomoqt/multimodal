import os
from pathlib import Path
import selfies as sf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings

def process_smiles_to_selfies(smiles, idx=None):
    """Helper function to convert SMILES to SELFIES with proper cleanup"""
    try:
        # 1) Remove spaces
        smiles = smiles.replace(" ", "")

        # 2) Parse as SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: invalid SMILES{' at index '+str(idx) if idx is not None else ''}: {smiles}")
            return None

        # 3) Remove stereochemistry
        Chem.rdmolops.RemoveStereochemistry(mol)
        
        # 4) Create clean SMILES (canonical, no stereo)
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        
        # 5) Convert to SELFIES
        return sf.encoder(canonical_smiles)
    except Exception as e:
        print(f"Error processing sequence{' at index '+str(idx) if idx is not None else ''}: {smiles}")
        print(f"Error: {e}")
        return None

def analyze_selfies_lengths(data_dir):
    """Analyze the length distribution of SELFIES strings in the dataset."""
    splits = ['train', 'val', 'test']
    length_stats = defaultdict(list)
    
    for split in splits:
        file_path = Path(data_dir) / f"tgt-{split}.txt"
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue
            
        print(f"\nAnalyzing {split} split...")
        
        # Read sequences
        with open(file_path) as f:
            sequences = [line.strip() for line in f]
        
        # Convert SMILES to SELFIES and analyze lengths
        selfies_lengths = []
        smiles_lengths = []
        token_counts = []
        
        for i, seq in enumerate(sequences):
            # Store original length
            smiles_lengths.append(len(seq))
            
            # Convert everything through RDKit pipeline
            selfies = process_smiles_to_selfies(seq, i)
            if selfies is None:
                continue
            
            # Store SELFIES length
            selfies_lengths.append(len(selfies))
            
            # Count tokens properly
            tokens = []
            current_token = ""
            bracket_depth = 0
            
            for char in selfies:
                current_token += char
                if char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        tokens.append(current_token)
                        current_token = ""
            
            token_counts.append(len(tokens))
        
        # Calculate statistics
        if not token_counts:  # Check if we have any valid entries
            print(f"No valid SELFIES found in {split} split")
            continue
            
        length_stats[split] = {
            'smiles_lengths': smiles_lengths,
            'selfies_lengths': selfies_lengths,
            'token_counts': token_counts,
            'stats': {
                'min_smiles': min(smiles_lengths),
                'max_smiles': max(smiles_lengths),
                'avg_smiles': np.mean(smiles_lengths),
                'min_selfies': min(selfies_lengths),
                'max_selfies': max(selfies_lengths),
                'avg_selfies': np.mean(selfies_lengths),
                'min_tokens': min(token_counts),
                'max_tokens': max(token_counts),
                'avg_tokens': np.mean(token_counts),
                'p95_tokens': np.percentile(token_counts, 95),
                'p99_tokens': np.percentile(token_counts, 99),
            }
        }
        
        # Print statistics
        stats = length_stats[split]['stats']
        print(f"\n{split} Statistics:")
        print(f"SMILES lengths: min={stats['min_smiles']}, max={stats['max_smiles']:.1f}, avg={stats['avg_smiles']:.1f}")
        print(f"SELFIES lengths: min={stats['min_selfies']}, max={stats['max_selfies']:.1f}, avg={stats['avg_selfies']:.1f}")
        print(f"Token counts: min={stats['min_tokens']}, max={stats['max_tokens']}, avg={stats['avg_tokens']:.1f}")
        print(f"95th percentile tokens: {stats['p95_tokens']:.1f}")
        print(f"99th percentile tokens: {stats['p99_tokens']:.1f}")
    
    return length_stats

def plot_distributions(length_stats, output_dir="plots"):
    """Create visualizations of the length distributions"""
    os.makedirs(output_dir, exist_ok=True)
    
    for split, data in length_stats.items():
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot SMILES length distribution
        sns.histplot(data=data['smiles_lengths'], ax=ax1, bins=50)
        ax1.set_title(f'{split}: SMILES Length Distribution')
        ax1.set_xlabel('Length')
        ax1.set_ylabel('Count')
        
        # Plot SELFIES length distribution
        sns.histplot(data=data['selfies_lengths'], ax=ax2, bins=50)
        ax2.set_title(f'{split}: SELFIES Length Distribution')
        ax2.set_xlabel('Length')
        ax2.set_ylabel('Count')
        
        # Plot token count distribution
        sns.histplot(data=data['token_counts'], ax=ax3, bins=50)
        ax3.set_title(f'{split}: SELFIES Token Count Distribution')
        ax3.set_xlabel('Number of Tokens')
        ax3.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{split}_length_distributions.png'))
        plt.close()

def analyze_short_selfies(data_dir, threshold=5):
    """Analyze SELFIES strings with very few tokens."""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        file_path = Path(data_dir) / f"tgt-{split}.txt"
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue
            
        print(f"\nAnalyzing short SELFIES in {split} split...")
        
        # Read sequences
        with open(file_path) as f:
            sequences = [line.strip() for line in f]
        
        short_selfies = []
        
        for i, seq in enumerate(sequences):
            # Convert through RDKit pipeline
            selfies = process_smiles_to_selfies(seq, i)
            if selfies is None:
                continue
            
            # Count tokens
            tokens = []
            current_token = ""
            bracket_depth = 0
            
            for char in selfies:
                current_token += char
                if char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        tokens.append(current_token)
                        current_token = ""
            
            if len(tokens) < threshold:
                # Verify by decoding back to SMILES
                try:
                    decoded_smiles = sf.decoder(selfies)
                    mol = Chem.MolFromSmiles(decoded_smiles)
                    validity = "Valid Molecule" if mol is not None else "Invalid Molecule"
                except Exception as e:
                    decoded_smiles = "DECODE_ERROR"
                    validity = str(e)
                
                short_selfies.append({
                    'index': i,
                    'original': seq,
                    'selfies': selfies,
                    'tokens': tokens,
                    'token_count': len(tokens),
                    'decoded_smiles': decoded_smiles,
                    'validity': validity
                })
        
        if short_selfies:
            print(f"\nFound {len(short_selfies)} SELFIES with < {threshold} tokens:")
            print("\nIndex | Count | Original | Clean SELFIES | Tokens | Decoded SMILES | Validity")
            print("-" * 100)
            for entry in short_selfies:
                tokens_str = ' '.join(entry['tokens'])
                print(f"{entry['index']:5d} | {entry['token_count']:5d} | "
                      f"{entry['original'][:20]:20s} | "
                      f"{entry['selfies'][:20]:20s} | "
                      f"{tokens_str[:20]:20s} | "
                      f"{entry['decoded_smiles'][:20]:20s} | "
                      f"{entry['validity']}")
        else:
            print(f"No SELFIES found with < {threshold} tokens")

def main():
    data_dir = "tokenized_baseline/data"
    
    print(f"Analyzing SELFIES lengths in {data_dir}")
    
    # First run the original analysis
    length_stats = analyze_selfies_lengths(data_dir)
    
    print("\nGenerating plots...")
    plot_distributions(length_stats)
    print("Plots saved in plots/ directory")
    
    # Now analyze short SELFIES
    print("\nAnalyzing suspiciously short SELFIES...")
    analyze_short_selfies(data_dir, threshold=5)

if __name__ == "__main__":
    main() 