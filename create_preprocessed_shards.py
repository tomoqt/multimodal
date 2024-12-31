import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from pathlib import Path
from tqdm import tqdm
from models.spectral_encoder import SpectralPreprocessor

def convert_row_to_tensor(row, meta, preprocessor, device='cpu'):
    """Convert raw row data to preprocessed tensors."""
    smiles_str = row['smiles']
    
    # Helper function for 1D spectra
    def process_1d_spectrum(data, domain, spectrum_type):
        if data is None:
            return None
        try:
            data = np.asarray(data, dtype=np.float32)
            # Create evenly spaced domain if not provided
            if domain is None:
                if spectrum_type == 'ir':
                    domain = np.linspace(400, 4000, len(data))
                elif spectrum_type == 'h_nmr':
                    domain = np.linspace(0, 12, len(data))
                else:  # c_nmr
                    domain = np.linspace(0, 200, len(data))
            domain = np.asarray(domain, dtype=np.float32)
            
            # Return as tuple of (data, domain) to match encoder expectations
            return (torch.tensor(data, dtype=torch.float, device=device),
                   torch.tensor(domain, dtype=torch.float, device=device))
            
        except Exception as e:
            print(f"Warning: Failed to process {spectrum_type} spectrum: {e}")
            return None

    # Process each spectrum type
    data_dict = {
        'ir': process_1d_spectrum(row.get('ir_spectra'), None, 'ir'),
        'h_nmr': process_1d_spectrum(row.get('h_nmr_spectra'), None, 'h_nmr'),
        'c_nmr': process_1d_spectrum(row.get('c_nmr_spectra'), None, 'c_nmr')
    }

    return smiles_str, data_dict

def process_parquet_in_row_groups(
    parquet_path,
    meta_path,
    output_dir,
    columns=None,
    device='cpu',
    resample_size=1000
):
    """Process Parquet file into preprocessed shards."""
    # Load metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Get domain ranges from metadata
    ir_range = [
        min(meta["ir_spectra"]["dimensions"]),
        max(meta["ir_spectra"]["dimensions"])
    ]
    h_nmr_range = [
        min(meta["h_nmr_spectra"]["dimensions"]),
        max(meta["h_nmr_spectra"]["dimensions"])
    ]
    c_nmr_range = [
        min(meta["c_nmr_spectra"]["dimensions"]),
        max(meta["c_nmr_spectra"]["dimensions"])
    ]

    # Initialize preprocessor with domain ranges
    preprocessor = SpectralPreprocessor(
        resample_size=resample_size,
        process_nmr=True,
        process_ir=True,
        process_c_nmr=True,
        nmr_window=h_nmr_range,
        ir_window=ir_range,
        c_nmr_window=c_nmr_range
    )

    parquet_file = pq.ParquetFile(parquet_path)
    num_row_groups = parquet_file.num_row_groups
    file_basename = Path(parquet_path).stem

    total_rows = 0
    for rg_index in tqdm(range(num_row_groups), desc=f"Processing {file_basename}"):
        # Read row group
        table = parquet_file.read_row_group(rg_index, columns=columns)
        df = table.to_pandas()
        num_rows_rg = len(df)

        # Process rows
        shard_data = {}
        for i in range(num_rows_rg):
            row = df.iloc[i]
            smiles_str, data_dict = convert_row_to_tensor(row, meta, preprocessor, device=device)
            
            # Store processed data - handle (data, domain) tuples properly
            global_row_index = total_rows + i
            shard_data[global_row_index] = {
                'smiles': smiles_str,
                'spectra': {
                    k: (v[0].cpu(), v[1].cpu()) if v is not None else None 
                    for k, v in data_dict.items()
                }
            }

        # Save shard
        shard_name = f"{file_basename}_rg{rg_index}_processed.pt"
        shard_path = Path(output_dir) / shard_name
        torch.save(shard_data, shard_path)

        total_rows += num_rows_rg

    print(f"[{parquet_path}] => Processed {total_rows} rows into {num_row_groups} shards.")

def main():
    data_path = Path("data_extraction/multimodal_spectroscopic_dataset")
    parquet_files = list(data_path.glob("*.parquet"))
    meta_path = data_path / "meta_data" / "meta_data_dict.json"
    output_dir = Path("preprocessed_binaries")
    output_dir.mkdir(exist_ok=True, parents=True)

    columns = ['smiles', 'ir_spectra', 'h_nmr_spectra', 'c_nmr_spectra', 'hsqc_nmr_spectrum']

    for parquet_file in parquet_files:
        process_parquet_in_row_groups(
            parquet_path=parquet_file,
            meta_path=meta_path,
            output_dir=output_dir,
            columns=columns,
            device='cpu',
            resample_size=1000  # Adjust as needed
        )

if __name__ == "__main__":
    main() 