# ============================
# File: create_training_data.py
# ============================
"""
Example script that demonstrates:
1) How to load spectral data + SMILES from a Parquet file (with metadata).
2) How to optionally store them as easy-to-retrieve binaries (e.g., .pt or .npy)
   for faster I/O during training.
3) Or, if feasible, directly load from Parquet in the training loop (often the
   best option when chunk-loading or partial reading is supported).

Note: For large datasets, using Parquet directly with chunked reads or Dask / 
PyArrow can be very effective. If you need random access to specific rows, 
storing row-wise binaries might help at the cost of more files on disk.
"""

import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from pathlib import Path
from tqdm import tqdm

def load_parquet_with_metadata(parquet_path, meta_path, columns):
    """
    Loads a Parquet file containing the specified columns, plus metadata from JSON.

    Parameters
    ----------
    parquet_path : Path-like
        Path to the .parquet file.
    meta_path : Path-like
        Path to the metadata JSON file.
    columns : list of str
        Columns to load from the Parquet (e.g. ['smiles','ir_spectra','h_nmr_spectra',...]).
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with loaded columns.
    meta : dict
        Metadata loaded from JSON.
    """
    df = pd.read_parquet(parquet_path, columns=columns, engine='fastparquet')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return df, meta


def convert_row_to_tensor(row, meta, device='cpu'):
    """
    Converts a single row of spectral data + SMILES into PyTorch tensors.

    row : pd.Series
        A row from the DataFrame containing (e.g. 'smiles', 'ir_spectra', etc.).
    meta : dict
        Dictionary with spectral dimension info, etc.
    device : str
        'cpu' or 'cuda' or similar.

    Returns
    -------
    smiles_str : str
        The SMILES string for the molecule.
    data_dict : dict of torch.Tensor
        Dictionary of Tensors for each spectral modality keyed by the name.
    """
    smiles_str = row['smiles']

    # Example: IR
    if 'ir_spectra' in row and row['ir_spectra'] is not None:
        ir_data = torch.tensor(np.array(row['ir_spectra']), dtype=torch.float, device=device)
    else:
        ir_data = None

    # Example: 1H NMR
    if 'h_nmr_spectra' in row and row['h_nmr_spectra'] is not None:
        h_nmr_data = torch.tensor(np.array(row['h_nmr_spectra']), dtype=torch.float, device=device)
    else:
        h_nmr_data = None

    # Example: 13C NMR
    if 'c_nmr_spectra' in row and row['c_nmr_spectra'] is not None:
        c_nmr_data = torch.tensor(np.array(row['c_nmr_spectra']), dtype=torch.float, device=device)
    else:
        c_nmr_data = None

    # Example: HSQC
    if 'hsqc_nmr_spectrum' in row and row['hsqc_nmr_spectrum'] is not None:
        # Could be 1D flattened or already 2D
        arr = np.array(row['hsqc_nmr_spectrum'])
        hsqc_data = torch.tensor(arr, dtype=torch.float, device=device)
    else:
        hsqc_data = None

    data_dict = {
        'ir': ir_data,
        'h_nmr': h_nmr_data,
        'c_nmr': c_nmr_data,
        'hsqc': hsqc_data
    }
    return smiles_str, data_dict


def store_as_binary(row_index, smiles_str, data_dict, output_dir, shard_size=1000):
    """
    Store data in sharded files (e.g., shard_0.pt contains rows 0-999)
    
    Parameters
    ----------
    row_index : int
        Index of the row in the DataFrame
    smiles_str : str
        SMILES string for the molecule
    data_dict : dict
        Dictionary of spectral tensors
    output_dir : Path-like
        Directory to store shards
    shard_size : int
        Number of samples per shard file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shard_index = row_index // shard_size
    shard_path = output_dir / f"shard_{shard_index}.pt"
    
    # Load existing shard or create new one
    if shard_path.exists():
        shard_data = torch.load(shard_path)
    else:
        shard_data = {}
    
    # Add new data to shard
    shard_data[row_index] = {
        'smiles': smiles_str,
        'spectra': {k: v.cpu() for k, v in data_dict.items() if v is not None}
    }
    
    torch.save(shard_data, shard_path)


def create_training_binaries(
    parquet_path, meta_path, columns, output_dir,
    limit=None, device='cpu', start_index=0
):
    """
    Demonstrates reading a Parquet, converting each row to Tensors, and storing
    them in .pt format for faster offline usage.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the Parquet file.
    meta_path : str or Path
        Path to the JSON metadata.
    columns : list of str
        Columns to load from the parquet (must include 'smiles' and any spectra).
    output_dir : str or Path
        Where to store the .pt files.
    limit : int or None
        If not None, limit processing to the first N rows (useful for tests).
    device : str
        Which device to load the data onto initially (e.g. 'cpu' or 'cuda').
    start_index : int
        Starting index for the binary files to avoid overwriting previous data.
    """
    df, meta = load_parquet_with_metadata(parquet_path, meta_path, columns=columns)

    n = len(df) if limit is None else min(limit, len(df))
    for i in tqdm(range(n), desc=f"Processing {Path(parquet_path).name}", leave=True):
        row = df.iloc[i]
        smiles_str, data_dict = convert_row_to_tensor(row, meta, device=device)
        store_as_binary(i + start_index, smiles_str, data_dict, output_dir)

    print(f"Created binary .pt files for {n} rows from {parquet_path}")


def load_training_binary(binary_file, device='cpu'):
    """
    Loads a single .pt file created by 'store_as_binary'.

    Parameters
    ----------
    binary_file : str or Path
        Path to the .pt file on disk.
    device : str
        'cpu' or 'cuda'. Tensors will be moved to that device.

    Returns
    -------
    smiles_str : str
    spectra_dict : dict of Tensors
    """
    data = torch.load(binary_file, map_location=device)
    smiles_str = data['smiles']
    spectra_dict = data['spectra']
    return smiles_str, spectra_dict


def main():
    """
    Example usage:
    1) We read all Parquet files in the directory
    2) We create .pt binaries for each row in a given folder.
    3) We load one .pt file back in as a demonstration.
    """
    data_path = Path("data_extraction/multimodal_spectroscopic_dataset")
    parquet_files = list(data_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in '{data_path}'.")

    meta_path = data_path / "meta_data" / "meta_data_dict.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta_data_dict.json found in '{meta_path.parent}'.")

    # Columns we plan to load for training
    columns = ['smiles', 'ir_spectra', 'h_nmr_spectra', 'c_nmr_spectra', 'hsqc_nmr_spectrum']

    # Process all parquet files with tqdm
    total_processed = 0
    for parquet_file in tqdm(parquet_files, desc="Processing parquet files", position=0):
        print(f"\nProcessing {parquet_file.name}...")
        
        # Calculate starting index for this file to avoid overwriting previous data
        if total_processed > 0:
            start_index = total_processed
        else:
            start_index = 0
            
        # Create training binaries
        create_training_binaries(
            parquet_path=parquet_file,
            meta_path=meta_path,
            columns=columns,
            output_dir="training_binaries",
            limit=None,
            device='cpu',
            start_index=start_index
        )
        
        # Update total count
        df_count = pd.read_parquet(parquet_file, columns=['smiles']).shape[0]
        total_processed += df_count
        print(f"Total processed so far: {total_processed} spectra")

    print(f"\nFinished processing all files. Total spectra processed: {total_processed}")


if __name__ == "__main__":
    main()
