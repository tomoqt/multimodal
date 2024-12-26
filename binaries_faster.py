import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from pathlib import Path
from tqdm import tqdm

def process_parquet_in_row_groups(
    parquet_path,
    meta_path,
    output_dir,
    columns=None,
    device='cpu',
    shard_size=None
):
    """
    Read a Parquet file in row-group chunks and rewrite each chunk 
    into a single .pt shard.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the .parquet file.
    meta_path : str or Path
        Path to a JSON metadata file (if needed).
    output_dir : str or Path
        Directory to store shards.
    columns : list of str
        Which columns to load from Parquet (must include 'smiles' and any needed spectra).
    device : str
        'cpu' or 'cuda' for loading the data.
    shard_size : int or None
        If you want to further split row groups internally, you can chunk them further
        (e.g., store smaller shards). Otherwise, each row group => one shard.
    """
    # Load metadata if needed
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    parquet_file = pq.ParquetFile(parquet_path)
    num_row_groups = parquet_file.num_row_groups
    file_basename = Path(parquet_path).stem  # e.g. 'part-0000'

    total_rows = 0
    for rg_index in tqdm(range(num_row_groups), desc=f"RowGroups {file_basename}"):
        # 1) Read only this row group
        table = parquet_file.read_row_group(rg_index, columns=columns)
        df = table.to_pandas()
        num_rows_rg = len(df)

        # 2) Convert all rows in this row group
        shard_data = {}
        for i in range(num_rows_rg):
            row = df.iloc[i]
            smiles_str, data_dict = convert_row_to_tensor(row, meta, device=device)
            # Store each row into shard_data
            global_row_index = total_rows + i
            shard_data[global_row_index] = {
                'smiles': smiles_str,
                'spectra': {k: v.cpu() for k, v in data_dict.items() if v is not None}
            }

        # 3) Write the entire row group as one .pt file
        shard_name = f"{file_basename}_rg{rg_index}.pt"
        shard_path = Path(output_dir) / shard_name
        torch.save(shard_data, shard_path)

        total_rows += num_rows_rg

    print(f"[{parquet_path}] => Processed {total_rows} rows into {num_row_groups} shards.")

def convert_row_to_tensor(row, meta, device='cpu'):
    """
    Example of row => (smiles, dict_of_tensors).
    Modify as needed.
    """
    for col in ['ir_spectra', 'h_nmr_spectra', 'c_nmr_spectra', 'hsqc_nmr_spectra']:
        if col in row:
            print(f"\n{col} type:", type(row[col]))
            if isinstance(row[col], (list, np.ndarray)):
                print(f"{col} shape/length:", np.shape(row[col]))
            print(f"{col} first few elements:", str(row[col])[:100])
    
    smiles_str = row['smiles']

    def to_tensor(x):
        if x is None:
            return None
        # Convert to numpy array and ensure float32 type
        try:
            # Handle lists/nested structures by explicitly converting to float array
            arr = np.asarray(x, dtype=np.float32)
            return torch.tensor(arr, dtype=torch.float, device=device)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert data to tensor. Data type: {type(x)}")
            if isinstance(x, (list, np.ndarray)):
                print(f"Shape/Length: {np.shape(x)}")
            return None

    data_dict = {
        'ir': to_tensor(row.get('ir_spectra')),
        'h_nmr': to_tensor(row.get('h_nmr_spectra')),
        'c_nmr': to_tensor(row.get('c_nmr_spectra')),
        'hsqc': to_tensor(row.get('hsqc_nmr_spectra')),  # Also fixed the key name to match
    }
    return smiles_str, data_dict

def main():
    data_path = Path("data_extraction/multimodal_spectroscopic_dataset")
    parquet_files = list(data_path.glob("*.parquet"))
    meta_path = data_path / "meta_data" / "meta_data_dict.json"
    output_dir = Path("training_binaries")
    output_dir.mkdir(exist_ok=True, parents=True)

    columns = ['smiles','ir_spectra','h_nmr_spectra','c_nmr_spectra','hsqc_nmr_spectrum']

    for parquet_file in parquet_files:
        process_parquet_in_row_groups(
            parquet_path=parquet_file,
            meta_path=meta_path,
            output_dir=output_dir,
            columns=columns,
            device='cpu',
            shard_size=None  # or specify if you want further splitting
        )

if __name__ == "__main__":
    main()
