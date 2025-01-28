import os
import json
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import click
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import Draw
from rdkit import RDLogger

# Import our custom tokenizer
from models.smiles_tokenizer import SmilesTokenizer

# Disable RDKit logging
RDLogger.DisableLog("rdApp.*")

# Default paths from config
current_dir = os.path.dirname(os.path.realpath(__file__))
default_config_path = os.path.join(current_dir, 'configs', 'test_config.yaml')
vocab_path = os.path.join(current_dir, 'vocab.txt')
tokenizer = SmilesTokenizer(vocab_file=vocab_path)

def load_config(config_path=None):
    """Load config from yaml file, falling back to defaults if not specified"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

# Load config to get data paths
config = load_config(default_config_path)
if config:
    default_data_dir = Path(config['data']['data_dir'])
    default_tokenized_dir = Path(config['data'].get('tokenized_dir', 'tokenized_baseline/data'))
else:
    default_data_dir = Path('data_extraction/multimodal_spectroscopic_dataset')
    default_tokenized_dir = Path('tokenized_baseline/data')

def analyze_smiles_basic_stats(smiles_list: list) -> dict:
    """Analyze basic SMILES statistics."""
    stats = {
        'total_molecules': len(smiles_list),
        'sequence_lengths': [],  # Now this will be token lengths
        'token_counts': Counter(),  # Now counting actual tokens
        'ring_counts': [],
        'branch_counts': [],
        'atom_counts': [],
        'bond_counts': [],
        'valid_smiles': 0,
        'unique_smiles': len(set(smiles_list)),
        'molecular_weights': [],
        'logp_values': [],
        'rotatable_bonds': [],
        'aromatic_rings': [],
        'stereocenters': [],
        'charge_states': [],
        'token_sequence_lengths': [],  # Length after tokenization
        'unique_tokens': set()
    }
    
    for smiles in tqdm(smiles_list, desc="Analyzing SMILES"):
        # Tokenize using the same tokenizer as training
        tokens = tokenizer.encode(
            smiles,
            add_special_tokens=True,
            max_length=512,
            truncation=True
        )
        decoded_tokens = [tokenizer._convert_id_to_token(t) for t in tokens]
        
        # Track token statistics
        stats['token_sequence_lengths'].append(len(tokens))
        stats['token_counts'].update(decoded_tokens)
        stats['unique_tokens'].update(decoded_tokens)
        
        # Basic string statistics (on original SMILES)
        stats['sequence_lengths'].append(len(smiles))
        
        # Count structural features in SMILES string
        stats['ring_counts'].append(smiles.count('1') + smiles.count('2') + 
                                  smiles.count('3') + smiles.count('4'))
        stats['branch_counts'].append(smiles.count('('))
        
        # RDKit-based analysis
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            stats['valid_smiles'] += 1
            
            # Molecular properties
            stats['atom_counts'].append(mol.GetNumAtoms())
            stats['bond_counts'].append(mol.GetNumBonds())
            stats['molecular_weights'].append(Descriptors.ExactMolWt(mol))
            stats['logp_values'].append(Descriptors.MolLogP(mol))
            stats['rotatable_bonds'].append(Descriptors.NumRotatableBonds(mol))
            stats['aromatic_rings'].append(len(Chem.GetSymmSSSR(mol)))
            stats['stereocenters'].append(len(Chem.FindMolChiralCenters(mol)))
            
            # Calculate formal charge
            total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            stats['charge_states'].append(total_charge)
    
    return stats

def analyze_atom_distributions(smiles_list: list) -> dict:
    """Analyze atom type distributions and common substructures."""
    stats = {
        'atom_types': Counter(),
        'common_fragments': Counter(),
        'ring_systems': Counter(),
        'functional_groups': defaultdict(int)
    }
    
    # SMARTS patterns for common functional groups
    functional_groups = {
        'alcohol': '[OH]',
        'amine': '[NH2]',
        'carboxyl': '[CX3](=O)[OX2H1]',
        'carbonyl': '[CX3]=O',
        'alkene': '[CX3]=[CX3]',
        'alkyne': '[CX2]#[CX2]',
        'ether': '[OX2]([CX4])[CX4]',
        'ester': '[CX3](=O)[OX2][CX4]',
        'amide': '[CX3](=O)[NX3]',
        'nitro': '[NX3](=O)=O',
        'sulfonic_acid': '[SX4](=O)(=O)[OX2H]',
        'phosphate': '[PX4](=O)([OX2H])([OX2H])[OX2H]'
    }
    
    for smiles in tqdm(smiles_list, desc="Analyzing atom distributions"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
            
        # Count atom types
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            stats['atom_types'][symbol] += 1
        
        # Count functional groups
        for name, smarts in functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                stats['functional_groups'][name] += 1
        
        # Analyze ring systems
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            ring_size = len(ring)
            stats['ring_systems'][f"{ring_size}-membered ring"] += 1
    
    return stats

def plot_statistics(basic_stats: dict, atom_stats: dict, output_dir: Path):
    """Create plots of SMILES statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="darkgrid")
    
    # 1. Original sequence length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=basic_stats['sequence_lengths'], bins=50)
    plt.title('SMILES Raw Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'raw_sequence_lengths.png')
    plt.close()
    
    # 2. Token sequence length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=basic_stats['token_sequence_lengths'], bins=50)
    plt.title('SMILES Token Sequence Length Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'token_sequence_lengths.png')
    plt.close()
    
    # 3. Token frequency distribution
    plt.figure(figsize=(15, 8))
    token_df = pd.DataFrame.from_dict(
        dict(basic_stats['token_counts'].most_common(30)), 
        orient='index', 
        columns=['count']
    )
    sns.barplot(data=token_df.reset_index(), x='count', y='index')
    plt.title('Token Frequency Distribution')
    plt.xlabel('Count')
    plt.ylabel('Token')
    plt.tight_layout()
    plt.savefig(output_dir / 'token_frequencies.png')
    plt.close()
    
    # 4. Molecular property distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    sns.histplot(data=basic_stats['molecular_weights'], bins=50, ax=axes[0,0])
    axes[0,0].set_title('Molecular Weight Distribution')
    axes[0,0].set_xlabel('Molecular Weight')
    
    sns.histplot(data=basic_stats['logp_values'], bins=50, ax=axes[0,1])
    axes[0,1].set_title('LogP Distribution')
    axes[0,1].set_xlabel('LogP')
    
    sns.histplot(data=basic_stats['rotatable_bonds'], bins=50, ax=axes[1,0])
    axes[1,0].set_title('Rotatable Bonds Distribution')
    axes[1,0].set_xlabel('Number of Rotatable Bonds')
    
    sns.histplot(data=basic_stats['aromatic_rings'], bins=50, ax=axes[1,1])
    axes[1,1].set_title('Aromatic Rings Distribution')
    axes[1,1].set_xlabel('Number of Aromatic Rings')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'molecular_properties.png')
    plt.close()
    
    # 5. Atom type distribution
    plt.figure(figsize=(12, 6))
    atom_df = pd.DataFrame.from_dict(
        dict(atom_stats['atom_types'].most_common()), 
        orient='index', 
        columns=['count']
    )
    sns.barplot(data=atom_df.reset_index(), x='count', y='index')
    plt.title('Atom Type Distribution')
    plt.xlabel('Count')
    plt.ylabel('Atom Type')
    plt.tight_layout()
    plt.savefig(output_dir / 'atom_types.png')
    plt.close()
    
    # 6. Functional group distribution
    plt.figure(figsize=(12, 6))
    func_df = pd.DataFrame.from_dict(
        dict(atom_stats['functional_groups']), 
        orient='index', 
        columns=['count']
    ).sort_values('count', ascending=False)
    sns.barplot(data=func_df.reset_index(), x='count', y='index')
    plt.title('Functional Group Distribution')
    plt.xlabel('Count')
    plt.ylabel('Functional Group')
    plt.tight_layout()
    plt.savefig(output_dir / 'functional_groups.png')
    plt.close()

def save_statistics(basic_stats: dict, atom_stats: dict, output_dir: Path):
    """Save statistics to JSON file."""
    # Convert Counter objects to dictionaries
    stats_to_save = {
        'basic_stats': {
            'total_molecules': basic_stats['total_molecules'],
            'valid_smiles': basic_stats['valid_smiles'],
            'unique_smiles': basic_stats['unique_smiles'],
            'raw_sequence_length_stats': {
                'mean': np.mean(basic_stats['sequence_lengths']),
                'std': np.std(basic_stats['sequence_lengths']),
                'min': min(basic_stats['sequence_lengths']),
                'max': max(basic_stats['sequence_lengths'])
            },
            'token_sequence_length_stats': {
                'mean': np.mean(basic_stats['token_sequence_lengths']),
                'std': np.std(basic_stats['token_sequence_lengths']),
                'min': min(basic_stats['token_sequence_lengths']),
                'max': max(basic_stats['token_sequence_lengths'])
            },
            'molecular_property_stats': {
                'molecular_weight': {
                    'mean': np.mean(basic_stats['molecular_weights']),
                    'std': np.std(basic_stats['molecular_weights']),
                    'min': min(basic_stats['molecular_weights']),
                    'max': max(basic_stats['molecular_weights'])
                },
                'logp': {
                    'mean': np.mean(basic_stats['logp_values']),
                    'std': np.std(basic_stats['logp_values']),
                    'min': min(basic_stats['logp_values']),
                    'max': max(basic_stats['logp_values'])
                },
                'rotatable_bonds': {
                    'mean': np.mean(basic_stats['rotatable_bonds']),
                    'std': np.std(basic_stats['rotatable_bonds']),
                    'min': min(basic_stats['rotatable_bonds']),
                    'max': max(basic_stats['rotatable_bonds'])
                }
            },
            'token_frequencies': dict(basic_stats['token_counts']),
            'unique_tokens': list(basic_stats['unique_tokens'])
        },
        'atom_stats': {
            'atom_types': dict(atom_stats['atom_types']),
            'functional_groups': dict(atom_stats['functional_groups']),
            'ring_systems': dict(atom_stats['ring_systems'])
        }
    }
    
    with open(output_dir / 'smiles_statistics.json', 'w') as f:
        json.dump(stats_to_save, f, indent=2)

@click.command()
@click.option(
    '--data_dir',
    '-d',
    type=click.Path(path_type=Path),
    default=default_tokenized_dir,
    help='Directory containing tokenized data (tgt-*.txt files)'
)
@click.option(
    '--output_dir',
    '-o',
    type=click.Path(path_type=Path),
    default=Path('smiles_statistics'),
    help='Output directory for statistics and plots'
)
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True, path_type=Path),
    default=default_config_path,
    help='Path to config file'
)
def main(data_dir: Path, output_dir: Path, config: Path):
    """Analyze statistics of SMILES strings from the dataset."""
    # Load config if provided
    if config:
        cfg = load_config(config)
        if cfg and not data_dir:
            data_dir = Path(cfg['data'].get('tokenized_dir', 'tokenized_baseline/data'))
    
    print(f"\nLoading SMILES strings from {data_dir}...")
    smiles_list = []
    for split in ['train', 'val', 'test']:
        tgt_file = data_dir / f'tgt-{split}.txt'
        if tgt_file.exists():
            with open(tgt_file) as f:
                smiles_list.extend([line.strip() for line in f])
    print(f"Loaded {len(smiles_list)} SMILES strings")
    
    print("\nAnalyzing basic SMILES statistics...")
    basic_stats = analyze_smiles_basic_stats(smiles_list)
    print(f"Found {basic_stats['valid_smiles']} valid SMILES strings")
    print(f"Found {basic_stats['unique_smiles']} unique SMILES strings")
    print(f"Found {len(basic_stats['unique_tokens'])} unique tokens")
    
    print("\nAnalyzing atom distributions and substructures...")
    atom_stats = analyze_atom_distributions(smiles_list)
    print(f"Found {len(atom_stats['atom_types'])} different atom types")
    
    print("\nGenerating plots...")
    plot_statistics(basic_stats, atom_stats, output_dir)
    
    print("\nSaving statistics...")
    save_statistics(basic_stats, atom_stats, output_dir)
    
    print(f"\nResults saved to {output_dir}")
    print("\nDone!")

if __name__ == '__main__':
    main() 