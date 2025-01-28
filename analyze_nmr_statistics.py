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
import regex as re
import yaml

# Default paths from config
current_dir = os.path.dirname(os.path.realpath(__file__))
default_config_path = os.path.join(current_dir, 'configs', 'test_config.yaml')

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

# Set vocab path relative to tokenized directory
default_vocab_path = default_tokenized_dir.parent / 'vocab.json'

def load_nmr_vocabulary(vocab_path: Path) -> dict:
    """Load NMR vocabulary from JSON file."""
    if not vocab_path.exists():
        raise FileNotFoundError(f"NMR vocabulary not found at {vocab_path}")
    
    with open(vocab_path) as f:
        return json.load(f)

def analyze_hnmr_statistics(sources: list, vocab: dict) -> dict:
    """Analyze H-NMR statistics from tokenized sources."""
    stats = {
        'total_spectra': 0,
        'peak_counts': [],
        'chemical_shifts': [],
        'multiplicities': Counter(),
        'coupling_constants': [],
        'proton_counts': [],
        'tokens_per_spectrum': [],
        'unique_tokens': set()
    }
    
    for source in tqdm(sources, desc="Analyzing H-NMR"):
        # Find H-NMR section
        if "1HNMR" not in source:
            continue
            
        stats['total_spectra'] += 1
        tokens = source.split()
        
        # Find H-NMR section
        try:
            start_idx = tokens.index("1HNMR")
        except ValueError:
            continue
            
        # Find end of H-NMR section (next spectral type or end)
        end_idx = len(tokens)
        for i in range(start_idx + 1, len(tokens)):
            if tokens[i] in ["13CNMR", "IR", "E0Pos", "E0Neg"]:
                end_idx = i
                break
                
        hnmr_tokens = tokens[start_idx:end_idx]
        stats['tokens_per_spectrum'].append(len(hnmr_tokens))
        stats['unique_tokens'].update(hnmr_tokens)
        
        # Count peaks (separated by |)
        peaks = ' '.join(hnmr_tokens).split('|')
        stats['peak_counts'].append(len(peaks) - 1)  # -1 because split creates empty string at end
        
        # Analyze each peak
        for peak in peaks:
            if not peak.strip():
                continue
                
            peak_tokens = peak.strip().split()
            if len(peak_tokens) < 4:  # Need at least shift, category, nH
                continue
                
            # Get chemical shift (first two numbers are range)
            try:
                shift_max = float(peak_tokens[1])
                shift_min = float(peak_tokens[2])
                stats['chemical_shifts'].append((shift_max + shift_min) / 2)
            except (ValueError, IndexError):
                continue
                
            # Get multiplicity
            try:
                multiplicity = peak_tokens[3]
                if multiplicity != "H":  # Skip proton count token
                    stats['multiplicities'][multiplicity] += 1
            except IndexError:
                continue
                
            # Get proton count
            try:
                proton_idx = peak_tokens.index("H")
                if proton_idx > 0:
                    protons = float(peak_tokens[proton_idx-1])
                    stats['proton_counts'].append(protons)
            except (ValueError, IndexError):
                continue
                
            # Get coupling constants
            try:
                j_idx = peak_tokens.index("J")
                j_values = [float(v) for v in peak_tokens[j_idx+1:]]
                stats['coupling_constants'].extend(j_values)
            except (ValueError, IndexError):
                continue
    
    return stats

def analyze_cnmr_statistics(sources: list, vocab: dict) -> dict:
    """Analyze C-NMR statistics from tokenized sources."""
    stats = {
        'total_spectra': 0,
        'peak_counts': [],
        'chemical_shifts': [],
        'tokens_per_spectrum': [],
        'unique_tokens': set()
    }
    
    for source in tqdm(sources, desc="Analyzing C-NMR"):
        # Find C-NMR section
        if "13CNMR" not in source:
            continue
            
        stats['total_spectra'] += 1
        tokens = source.split()
        
        # Find C-NMR section
        try:
            start_idx = tokens.index("13CNMR")
        except ValueError:
            continue
            
        # Find end of C-NMR section (next spectral type or end)
        end_idx = len(tokens)
        for i in range(start_idx + 1, len(tokens)):
            if tokens[i] in ["1HNMR", "IR", "E0Pos", "E0Neg"]:
                end_idx = i
                break
                
        cnmr_tokens = tokens[start_idx:end_idx]
        stats['tokens_per_spectrum'].append(len(cnmr_tokens))
        stats['unique_tokens'].update(cnmr_tokens)
        
        # Each token after 13CNMR is a chemical shift
        shifts = []
        for token in cnmr_tokens[1:]:  # Skip 13CNMR token
            try:
                shift = float(token)
                shifts.append(shift)
            except ValueError:
                continue
                
        if shifts:
            stats['peak_counts'].append(len(shifts))
            stats['chemical_shifts'].extend(shifts)
    
    return stats

def plot_statistics(h_stats: dict, c_stats: dict, output_dir: Path):
    """Create plots of NMR statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="darkgrid")  # Use seaborn's darkgrid style
    
    # 1. H-NMR Plots
    if h_stats['total_spectra'] > 0:
        # Chemical shift distribution
        if h_stats['chemical_shifts']:
            plt.figure(figsize=(12, 6))
            sns.histplot(data=h_stats['chemical_shifts'], bins=50)
            plt.title('H-NMR Chemical Shift Distribution')
            plt.xlabel('Chemical Shift (ppm)')
            plt.ylabel('Count')
            plt.savefig(output_dir / 'hnmr_shifts.png')
            plt.close()
        
        # Multiplicities
        if h_stats['multiplicities']:
            plt.figure(figsize=(12, 6))
            mult_df = pd.DataFrame.from_dict(h_stats['multiplicities'], orient='index', columns=['count'])
            sns.barplot(data=mult_df.reset_index(), x='count', y='index')
            plt.title('H-NMR Multiplicity Distribution')
            plt.xlabel('Count')
            plt.tight_layout()
            plt.savefig(output_dir / 'hnmr_multiplicities.png')
            plt.close()
        
        # Coupling constants
        if h_stats['coupling_constants']:
            plt.figure(figsize=(12, 6))
            sns.histplot(data=h_stats['coupling_constants'], bins=50)
            plt.title('H-NMR Coupling Constants Distribution')
            plt.xlabel('Coupling Constant (Hz)')
            plt.ylabel('Count')
            plt.savefig(output_dir / 'hnmr_couplings.png')
            plt.close()
        
        # Peaks per spectrum
        if h_stats['peak_counts']:
            plt.figure(figsize=(12, 6))
            sns.histplot(data=h_stats['peak_counts'], bins=range(max(h_stats['peak_counts'])+2))
            plt.title('H-NMR Peaks per Spectrum')
            plt.xlabel('Number of Peaks')
            plt.ylabel('Count')
            plt.savefig(output_dir / 'hnmr_peaks.png')
            plt.close()
        
        # Proton counts
        if h_stats['proton_counts']:
            plt.figure(figsize=(12, 6))
            sns.histplot(data=h_stats['proton_counts'], bins=range(int(max(h_stats['proton_counts']))+2))
            plt.title('H-NMR Protons per Peak')
            plt.xlabel('Number of Protons')
            plt.ylabel('Count')
            plt.savefig(output_dir / 'hnmr_protons.png')
            plt.close()
    
    # 2. C-NMR Plots
    if c_stats['total_spectra'] > 0:
        # Chemical shift distribution
        if c_stats['chemical_shifts']:
            plt.figure(figsize=(12, 6))
            sns.histplot(data=c_stats['chemical_shifts'], bins=50)
            plt.title('C-NMR Chemical Shift Distribution')
            plt.xlabel('Chemical Shift (ppm)')
            plt.ylabel('Count')
            plt.savefig(output_dir / 'cnmr_shifts.png')
            plt.close()
        
        # Peaks per spectrum
        if c_stats['peak_counts']:
            plt.figure(figsize=(12, 6))
            sns.histplot(data=c_stats['peak_counts'], bins=range(max(c_stats['peak_counts'])+2))
            plt.title('C-NMR Peaks per Spectrum')
            plt.xlabel('Number of Peaks')
            plt.ylabel('Count')
            plt.savefig(output_dir / 'cnmr_peaks.png')
            plt.close()

def save_statistics(h_stats: dict, c_stats: dict, output_dir: Path):
    """Save statistics to JSON file."""
    # Convert sets to lists for JSON serialization
    h_stats['unique_tokens'] = list(h_stats['unique_tokens'])
    c_stats['unique_tokens'] = list(c_stats['unique_tokens'])
    
    # Convert Counter to dict for JSON serialization
    h_stats['multiplicities'] = dict(h_stats['multiplicities'])
    
    # Calculate additional statistics
    if h_stats['total_spectra'] > 0:
        h_stats['avg_peaks_per_spectrum'] = np.mean(h_stats['peak_counts'])
        h_stats['avg_protons_per_peak'] = np.mean(h_stats['proton_counts'])
        h_stats['avg_coupling_constant'] = np.mean(h_stats['coupling_constants'])
        h_stats['avg_tokens_per_spectrum'] = np.mean(h_stats['tokens_per_spectrum'])
    
    if c_stats['total_spectra'] > 0:
        c_stats['avg_peaks_per_spectrum'] = np.mean(c_stats['peak_counts'])
        c_stats['avg_tokens_per_spectrum'] = np.mean(c_stats['tokens_per_spectrum'])
    
    # Save to JSON
    with open(output_dir / 'nmr_statistics.json', 'w') as f:
        json.dump({
            'h_nmr': h_stats,
            'c_nmr': c_stats
        }, f, indent=2)

def analyze_token_statistics(sources: list, vocab: dict) -> dict:
    """Analyze token statistics across all spectra."""
    stats = {
        'token_counts': Counter(),  # Count of each token
        'tokens_per_spectrum': [],  # Number of tokens in each spectrum
        'unique_tokens_per_spectrum': [],  # Number of unique tokens per spectrum
        'sequence_lengths': {  # Length statistics for each type
            'h_nmr': [],
            'c_nmr': [],
            'total': []
        }
    }
    
    for source in tqdm(sources, desc="Analyzing tokens"):
        tokens = source.split()
        stats['tokens_per_spectrum'].append(len(tokens))
        stats['unique_tokens_per_spectrum'].append(len(set(tokens)))
        stats['token_counts'].update(tokens)
        
        # Track sequence lengths by type
        if "1HNMR" in tokens:
            start_idx = tokens.index("1HNMR")
            end_idx = len(tokens)
            for i in range(start_idx + 1, len(tokens)):
                if tokens[i] in ["13CNMR", "IR", "E0Pos", "E0Neg"]:
                    end_idx = i
                    break
            stats['sequence_lengths']['h_nmr'].append(end_idx - start_idx)
            
        if "13CNMR" in tokens:
            start_idx = tokens.index("13CNMR")
            end_idx = len(tokens)
            for i in range(start_idx + 1, len(tokens)):
                if tokens[i] in ["1HNMR", "IR", "E0Pos", "E0Neg"]:
                    end_idx = i
                    break
            stats['sequence_lengths']['c_nmr'].append(end_idx - start_idx)
        
        stats['sequence_lengths']['total'].append(len(tokens))
    
    return stats

def plot_token_statistics(token_stats: dict, output_dir: Path):
    """Plot token statistics."""
    # 1. Token frequency distribution (top 30)
    plt.figure(figsize=(15, 8))
    token_df = pd.DataFrame.from_dict(
        dict(token_stats['token_counts'].most_common(30)), 
        orient='index', 
        columns=['count']
    )
    sns.barplot(data=token_df.reset_index(), x='count', y='index')
    plt.title('Top 30 Most Common Tokens')
    plt.xlabel('Count')
    plt.ylabel('Token')
    plt.tight_layout()
    plt.savefig(output_dir / 'token_frequencies.png')
    plt.close()
    
    # 2. Tokens per spectrum distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=token_stats['tokens_per_spectrum'], bins=50)
    plt.title('Tokens per Spectrum Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'tokens_per_spectrum.png')
    plt.close()
    
    # 3. Unique tokens per spectrum
    plt.figure(figsize=(12, 6))
    sns.histplot(data=token_stats['unique_tokens_per_spectrum'], bins=50)
    plt.title('Unique Tokens per Spectrum Distribution')
    plt.xlabel('Number of Unique Tokens')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'unique_tokens_per_spectrum.png')
    plt.close()
    
    # 4. Sequence length distributions by type
    plt.figure(figsize=(12, 6))
    for key, lengths in token_stats['sequence_lengths'].items():
        if lengths:  # Only plot if we have data
            sns.histplot(data=lengths, bins=50, alpha=0.5, label=key)
    plt.title('Sequence Length Distribution by Type')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(output_dir / 'sequence_lengths.png')
    plt.close()

def save_token_statistics(token_stats: dict, output_dir: Path):
    """Save token statistics to JSON file."""
    # Convert Counter to dict for JSON serialization
    stats_to_save = {
        'total_unique_tokens': len(token_stats['token_counts']),
        'avg_tokens_per_spectrum': np.mean(token_stats['tokens_per_spectrum']),
        'avg_unique_tokens_per_spectrum': np.mean(token_stats['unique_tokens_per_spectrum']),
        'sequence_length_stats': {
            'h_nmr': {
                'mean': np.mean(token_stats['sequence_lengths']['h_nmr']) if token_stats['sequence_lengths']['h_nmr'] else 0,
                'max': max(token_stats['sequence_lengths']['h_nmr']) if token_stats['sequence_lengths']['h_nmr'] else 0,
                'min': min(token_stats['sequence_lengths']['h_nmr']) if token_stats['sequence_lengths']['h_nmr'] else 0
            },
            'c_nmr': {
                'mean': np.mean(token_stats['sequence_lengths']['c_nmr']) if token_stats['sequence_lengths']['c_nmr'] else 0,
                'max': max(token_stats['sequence_lengths']['c_nmr']) if token_stats['sequence_lengths']['c_nmr'] else 0,
                'min': min(token_stats['sequence_lengths']['c_nmr']) if token_stats['sequence_lengths']['c_nmr'] else 0
            },
            'total': {
                'mean': np.mean(token_stats['sequence_lengths']['total']),
                'max': max(token_stats['sequence_lengths']['total']),
                'min': min(token_stats['sequence_lengths']['total'])
            }
        },
        'token_frequencies': dict(token_stats['token_counts'])
    }
    
    with open(output_dir / 'token_statistics.json', 'w') as f:
        json.dump(stats_to_save, f, indent=2)

@click.command()
@click.option(
    '--data_dir',
    '-d',
    type=click.Path(path_type=Path),
    default=default_tokenized_dir,
    help='Directory containing tokenized data (src-*.txt files)'
)
@click.option(
    '--vocab_path',
    '-v',
    type=click.Path(path_type=Path),
    default=default_vocab_path,
    help='Path to NMR vocabulary JSON file'
)
@click.option(
    '--output_dir',
    '-o',
    type=click.Path(path_type=Path),
    default=Path('nmr_statistics'),
    help='Output directory for statistics and plots'
)
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True, path_type=Path),
    default=default_config_path,
    help='Path to config file'
)
def main(data_dir: Path, vocab_path: Path, output_dir: Path, config: Path):
    """Analyze statistics of tokenized NMR spectra."""
    # Load config if provided
    if config:
        cfg = load_config(config)
        if cfg and not data_dir:
            data_dir = Path(cfg['data'].get('tokenized_dir', 'tokenized_baseline/data'))
    
    print("\nLoading NMR vocabulary...")
    vocab = load_nmr_vocabulary(vocab_path)
    
    print(f"\nLoading source files from {data_dir}...")
    sources = []
    for split in ['train', 'val', 'test']:
        src_file = data_dir / f'src-{split}.txt'
        if src_file.exists():
            with open(src_file) as f:
                sources.extend([line.strip() for line in f])
    print(f"Loaded {len(sources)} total spectra")
    
    print("\nAnalyzing H-NMR spectra...")
    h_stats = analyze_hnmr_statistics(sources, vocab)
    print(f"Found {h_stats['total_spectra']} H-NMR spectra")
    
    print("\nAnalyzing C-NMR spectra...")
    c_stats = analyze_cnmr_statistics(sources, vocab)
    print(f"Found {c_stats['total_spectra']} C-NMR spectra")
    
    print("\nAnalyzing token statistics...")
    token_stats = analyze_token_statistics(sources, vocab)
    print(f"Found {len(token_stats['token_counts'])} unique tokens")
    
    print("\nGenerating plots...")
    plot_statistics(h_stats, c_stats, output_dir)
    plot_token_statistics(token_stats, output_dir)
    
    print("\nSaving statistics...")
    save_statistics(h_stats, c_stats, output_dir)
    save_token_statistics(token_stats, output_dir)
    
    print(f"\nResults saved to {output_dir}")
    print("\nDone!")

if __name__ == '__main__':
    main() 