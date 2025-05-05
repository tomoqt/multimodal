'''
python data/reshape_tokenized_dataset.py \
    --input_dir data/tokenized_baseline/data \
    --output_dir data/reshaped_tokenized_data/data \
    --val_size 0.01 \
    --test_size 79000 \
    --seed 42
'''

import argparse
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_split_data(data_dir: Path, split: str):
    """Loads data for a specific split (train, val, test)."""
    src_path = data_dir / f"src-{split}.txt"
    tgt_path = data_dir / f"tgt-{split}.txt"
    ir_path = data_dir / f"ir-{split}.npy"

    sources, targets, ir_data = [], [], None

    if src_path.exists():
        with open(src_path, 'r') as f:
            sources = [line.strip() for line in f]
        logging.info(f"Loaded {len(sources)} source sequences from {src_path}")
    else:
        logging.warning(f"Source file not found: {src_path}")

    if tgt_path.exists():
        with open(tgt_path, 'r') as f:
            targets = [line.strip() for line in f]
        logging.info(f"Loaded {len(targets)} target sequences from {tgt_path}")
    else:
        logging.warning(f"Target file not found: {tgt_path}")

    # Basic consistency check
    if sources and targets and len(sources) != len(targets):
        logging.error(f"Mismatch between number of source ({len(sources)}) and target ({len(targets)}) sequences for split {split}. Aborting.")
        raise ValueError("Source and target sequence count mismatch.")

    if ir_path.exists():
        try:
            # Use np.memmap like in the training script
            ir_memmap = np.memmap(
                ir_path,
                dtype='float32',
                mode='r',
                shape=None # Infer shape
            )
            array_shape = ir_memmap.shape
            ir_data_loaded = None # Initialize

            # Reshape if loaded as 1D array (like in training script)
            if len(array_shape) == 1:
                num_samples = len(sources) # Assumes sources are already loaded
                if num_samples > 0:
                    if array_shape[0] % num_samples == 0: # Check divisibility
                        feature_dim = array_shape[0] // num_samples
                        # Reshape the memmap *view*
                        ir_data_loaded = ir_memmap.reshape(num_samples, feature_dim)
                        logging.info(f"Reshaped loaded 1D IR data to ({num_samples}, {feature_dim}) for {split}")
                    else:
                        logging.error(f"IR array size {array_shape[0]} not divisible by source sequence count {num_samples} for split {split}. Cannot reshape. Ignoring IR.")
                        ir_data_loaded = None
                else:
                     logging.warning(f"No source sequences loaded for split {split}. Cannot infer IR shape from 1D array. Ignoring IR.")
                     ir_data_loaded = None
            elif len(array_shape) >= 2:
                # Assume already has correct shape (e.g., N x Features)
                 ir_data_loaded = ir_memmap
            else:
                 logging.error(f"Loaded IR data has unexpected shape {array_shape} for split {split}. Ignoring IR.")
                 ir_data_loaded = None

            # If successfully loaded/reshaped, load fully into memory for processing
            if ir_data_loaded is not None:
                 # Check consistency with sources before full load
                 if len(sources) > 0 and ir_data_loaded.shape[0] != len(sources):
                     logging.error(f"Mismatch between number of source sequences ({len(sources)}) and IR samples ({ir_data_loaded.shape[0]}) for split {split} *after* potential reshape. IR data will be ignored.")
                     ir_data = None
                 else:
                    logging.info(f"Loading IR data {ir_data_loaded.shape} fully into memory from {ir_path}...")
                    ir_data = np.array(ir_data_loaded) # Load memmap fully into memory
                    logging.info(f"Successfully loaded IR data with shape {ir_data.shape} from {ir_path}")
            else:
                 ir_data = None # Loading/reshaping failed

            # Explicitly delete memmap object handle
            del ir_memmap

        except Exception as e:
            logging.error(f"Failed to load or process IR data from {ir_path}: {e}")
            ir_data = None
    else:
        logging.info(f"IR data file not found: {ir_path}")

    # Final check on loaded data
    if ir_data is not None and len(sources) > 0 and ir_data.shape[0] != len(sources):
        logging.error(f"Final check failed: Mismatch between source ({len(sources)}) and IR ({ir_data.shape[0]}) samples for split {split}. Discarding IR data.")
        ir_data = None

    if not sources: # If sources list is empty (e.g., file not found)
         return [], [], None

    return sources, targets, ir_data

def save_split_data(output_dir: Path, split: str, sources: list, targets: list, ir_data: np.ndarray = None):
    """Saves the data for a specific split."""
    output_dir.mkdir(parents=True, exist_ok=True)

    src_path = output_dir / f"src-{split}.txt"
    tgt_path = output_dir / f"tgt-{split}.txt"
    ir_path = output_dir / f"ir-{split}.npy"

    with open(src_path, 'w') as f:
        for line in sources:
            f.write(line + '\n')
    logging.info(f"Saved {len(sources)} source sequences to {src_path}")

    with open(tgt_path, 'w') as f:
        for line in targets:
            f.write(line + '\n')
    logging.info(f"Saved {len(targets)} target sequences to {tgt_path}")

    if ir_data is not None and len(ir_data) > 0:
        # Use tofile() to save raw binary data without the npy header
        ir_data.tofile(ir_path)
        logging.info(f"Saved raw IR data with shape {ir_data.shape} to {ir_path}")
    else:
        logging.info(f"No IR data to save for split {split}.")


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    val_size = args.val_size
    test_size = args.test_size

    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        return

    logging.info(f"Loading data from input directory: {input_dir}")

    # Load all existing splits
    all_sources, all_targets, all_ir = [], [], []
    for split in ['train', 'val', 'test']:
        sources, targets, ir_data = load_split_data(input_dir, split)
        if sources: # Only extend if data was loaded
            all_sources.extend(sources)
            all_targets.extend(targets)
            if ir_data is not None:
                 # Ensure ir_data is a list of arrays for consistent concatenation
                 if len(all_ir) == 0:
                     # Initialize with the first chunk
                      all_ir = ir_data
                 elif all_ir.shape[1:] == ir_data.shape[1:]: # Check if feature dimensions match
                      all_ir = np.concatenate((all_ir, ir_data), axis=0)
                 else:
                      logging.warning(f"IR data feature dimensions mismatch for split {split}. Skipping concatenation.")
            elif len(all_ir) > 0 and len(sources) > 0:
                 # Handle case where current split has no IR but previous splits did
                 # Need to decide how to handle this - e.g., pad with zeros or raise error
                 # For now, log a warning. This might indicate inconsistent data.
                 logging.warning(f"Missing IR data for split {split}, but previous splits had IR data. Shapes might become inconsistent.")


    total_samples = len(all_sources)
    if total_samples == 0:
        logging.error("No data loaded from the input directory. Aborting.")
        return

    logging.info(f"Total samples loaded: {total_samples}")
    if len(all_ir) > 0:
        logging.info(f"Total IR data shape after concatenation: {all_ir.shape}")
        if all_ir.shape[0] != total_samples:
             logging.warning(f"Final IR sample count ({all_ir.shape[0]}) does not match source/target count ({total_samples}). IR data might be incomplete or inconsistent.")
             # Decide on handling: either discard all IR or proceed with caution.
             # Discarding for safety:
             logging.warning("Discarding all IR data due to count mismatch.")
             all_ir = None


    # Create indices for splitting
    indices = np.arange(total_samples)

    # Calculate split sizes
    if test_size < 1.0:
        test_count = int(total_samples * test_size)
    else:
        test_count = int(test_size)

    if val_size < 1.0:
         # Calculate val_count based on the *remaining* data after test split
        remaining_after_test = total_samples - test_count
        val_count = int(remaining_after_test * val_size)
    else:
        val_count = int(val_size)

    train_count = total_samples - test_count - val_count

    if train_count <= 0 or test_count <= 0 or val_count <= 0:
         logging.error(f"Calculated split counts are invalid: Train={train_count}, Val={val_count}, Test={test_count}. Check split sizes.")
         return

    logging.info(f"Calculated split counts: Train={train_count}, Validation={val_count}, Test={test_count}")

    # Split test set first
    if test_count > 0:
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_count,
            random_state=args.seed,
            shuffle=True
        )
    else:
        train_val_indices = indices
        test_indices = np.array([], dtype=int)


    # Split train and validation from the remainder
    if val_count > 0 and len(train_val_indices) > 0:
         # Calculate val proportion relative to the train_val set
         val_proportion = val_count / len(train_val_indices)
         train_indices, val_indices = train_test_split(
             train_val_indices,
             test_size=val_proportion, # Use proportion here
             random_state=args.seed, # Use the same seed for reproducibility
             shuffle=True
        )
    else:
         train_indices = train_val_indices
         val_indices = np.array([], dtype=int)

    # --- Data Extraction based on Indices ---
    train_sources = [all_sources[i] for i in train_indices]
    train_targets = [all_targets[i] for i in train_indices]
    train_ir = all_ir[train_indices] if all_ir is not None and len(all_ir) > 0 else None

    val_sources = [all_sources[i] for i in val_indices]
    val_targets = [all_targets[i] for i in val_indices]
    val_ir = all_ir[val_indices] if all_ir is not None and len(all_ir) > 0 else None

    test_sources = [all_sources[i] for i in test_indices]
    test_targets = [all_targets[i] for i in test_indices]
    test_ir = all_ir[test_indices] if all_ir is not None and len(all_ir) > 0 else None

    logging.info(f"Final split sizes: Train={len(train_sources)}, Validation={len(val_sources)}, Test={len(test_sources)}")

    # Save the new splits
    logging.info(f"Saving reshaped data to output directory: {output_dir}")
    save_split_data(output_dir, 'train', train_sources, train_targets, train_ir)
    save_split_data(output_dir, 'val', val_sources, val_targets, val_ir)
    save_split_data(output_dir, 'test', test_sources, test_targets, test_ir)

    logging.info("Dataset reshaping completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reshape a tokenized dataset into new train/validation/test splits.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the existing tokenized dataset (e.g., 'data/tokenized_baseline/data').")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the reshaped dataset will be saved.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction (0.0-1.0) or absolute number of samples for the validation set.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction (0.0-1.0) or absolute number of samples for the test set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")

    args = parser.parse_args()
    main(args) 