import os
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

def process_ir(ir: np.ndarray, interpolation_points: int = 400) -> str:
    """
    Process an IR spectrum by reinterpolating over a fixed number of points,
    normalizing intensities to range [0,100] and rounding them to integer strings.
    Returns a string starting with an 'IR' prefix followed by space-separated token values.
    """
    # Use actual input length for original_x
    original_x = np.linspace(400, 4000, len(ir))
    interpolation_x = np.linspace(400, 4000, interpolation_points)
    interp = interp1d(original_x, ir)
    interp_ir = interp(interpolation_x)
    
    # Normalize
    interp_ir = interp_ir + abs(min(interp_ir))
    interp_ir = (interp_ir / max(interp_ir)) * 100
    interp_ir = np.round(interp_ir, decimals=0).astype(int).astype(str)
    
    return 'IR ' + ' '.join(interp_ir) + ' '

def process_and_save_ir_data(data_dir: Path, output_dir: Path, splits=['train', 'val', 'test']):
    """
    Process all IR spectra from .npy files, tokenize them, and save to text files.
    Also builds vocabulary from all tokens encountered.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_set = set(['<PAD>', '<UNK>', 'IR'])  # Initialize with special tokens
    
    for split in splits:
        # First load source file to get number of samples
        src_file = data_dir / f"src-{split}.txt"
        if not src_file.exists():
            print(f"Warning: Source file {src_file} not found, skipping...")
            continue
        with open(src_file) as f:
            num_samples = len([line.strip() for line in f])
        print(f"Found {num_samples} samples in source file")

        ir_file = data_dir / f"ir-{split}.npy"
        if not ir_file.exists():
            print(f"Warning: {ir_file} not found, skipping...")
            continue
            
        print(f"\nProcessing {split} split...")
        # Load IR data using memory mapping
        try:
            ir_data = np.memmap(
                ir_file,
                dtype='float32',
                mode='r',
                shape=None
            )
            # Get the actual shape from the memmap
            array_shape = ir_data.shape
            # Reshape if needed (should be 2D: [num_samples, features])
            if len(array_shape) == 1:
                # Calculate feature dimension based on total size and number of samples
                feature_dim = array_shape[0] // num_samples
                print(f"Calculated feature dimension: {feature_dim}")
                ir_data = ir_data.reshape(num_samples, feature_dim)
            print(f"Loaded IR data with shape: {ir_data.shape}")
        except Exception as e:
            print(f"Failed to load IR data: {e}")
            continue
        
        # Process each spectrum and save
        output_file = output_dir / f"ir-{split}.txt"
        with open(output_file, 'w') as f:
            for spectrum in tqdm(ir_data, desc=f"Processing {split} IR spectra"):
                # Copy the data from memmap to avoid issues
                spectrum = spectrum.copy()
                processed = process_ir(spectrum)
                f.write(processed + '\n')
                # Add tokens to vocabulary
                tokens = processed.strip().split()
                vocab_set.update(tokens)
        
        print(f"Saved processed IR data to {output_file}")
        # Clean up memmap
        del ir_data
    
    return sorted(list(vocab_set))

def build_ir_vocabulary(data_dir: Path, output_dir: Path, splits=['train', 'val', 'test']):
    """
    Process all IR data and build vocabulary.
    Saves both processed IR data and vocabulary files.
    
    Args:
        data_dir: Directory containing ir-{split}.npy files
        output_dir: Directory to save processed files and vocabulary
        splits: List of dataset splits to process
    """
    print("Processing IR data and building vocabulary...")
    
    # Process all IR data and get vocabulary
    vocab_list = process_and_save_ir_data(data_dir, output_dir, splits)
    
    # Save vocabulary as text file
    vocab_file = output_dir / "ir_vocab.txt"
    with open(vocab_file, "w") as f:
        for token in vocab_list:
            f.write(token + "\n")
    
    # Save vocabulary as JSON mapping
    token_to_id = {token: idx for idx, token in enumerate(vocab_list)}
    json_path = output_dir / "ir_vocab.json"
    with open(json_path, "w") as f:
        json.dump(token_to_id, f, indent=2)
    
    print(f"\nVocabulary files saved:")
    print(f"Text format: {vocab_file}")
    print(f"JSON format: {json_path}")
    print(f"Total vocabulary size: {len(vocab_list)}")
    
    return vocab_list, token_to_id

if __name__ == '__main__':
    # Example usage
    data_dir = Path('tokenized_baseline/data')  # Directory with raw .npy files
    output_dir = Path('tokenized_baseline/ir_processed')  # Directory for processed files
    vocab_list, token_to_id = build_ir_vocabulary(data_dir, output_dir) 