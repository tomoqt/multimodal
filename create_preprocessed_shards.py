import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from pathlib import Path
from tqdm import tqdm
from models.preprocessing import GlobalWindowResampler, GlobalWindowResampler2D, Normalizer

class SpectralPreprocessor:
    def __init__(self, resample_size=1000):
        self.nmr_processor = GlobalWindowResampler(target_size=resample_size, window=[0, 12])
        self.ir_processor = GlobalWindowResampler(target_size=resample_size, window=[400, 4000])
        self.hsqc_processor = GlobalWindowResampler2D(
            target_size=(resample_size, resample_size),
            window_h=[0, 12],
            window_c=[0, 200],
            method='linear'
        )
        self.normalizer = Normalizer()
        self.resample_size = resample_size

    def process_nmr(self, intensities, domain):
        arr = self.nmr_processor(intensities, domain)
        return self.normalizer(arr)

    def process_ir(self, intensities, domain):
        arr = self.ir_processor(intensities, domain)
        return self.normalizer(arr)

    def process_hsqc(self, intensities, domain_h, domain_c):
        arr = self.hsqc_processor(intensities, domain_h, domain_c)
        return self.normalizer(arr)

def convert_row_to_tensor(row, meta, preprocessor, device='cpu'):
    """Convert raw row data to preprocessed tensors."""
    smiles_str = row['smiles']
    
    # Helper function for 1D spectra
    def process_1d_spectrum(data, domain, mode='ir'):
        if data is None:
            return None
        try:
            data = np.asarray(data, dtype=np.float32)
            # Create evenly spaced domain if not provided
            if domain is None:
                if mode == 'ir':
                    domain = np.linspace(400, 4000, len(data))
                else:  # nmr
                    domain = np.linspace(0, 12, len(data))
            domain = np.asarray(domain, dtype=np.float32)
            
            if mode == 'ir':
                arr = preprocessor.process_ir(data, domain)
            else:
                arr = preprocessor.process_nmr(data, domain)
            return torch.tensor(arr, dtype=torch.float, device=device)
        except Exception as e:
            print(f"Warning: Failed to process {mode} spectrum: {e}")
            return None

    # Helper function for 2D HSQC
    def process_hsqc_spectrum(data):
        if data is None:
            return None
        try:
            # Assuming data is a 2D array or can be converted to one
            data = np.asarray(data, dtype=np.float32)
            if data.ndim != 2:
                print(f"Warning: HSQC data has unexpected shape: {data.shape}")
                return None
                
            # Create default domains if needed
            domain_h = np.linspace(0, 12, data.shape[0])
            domain_c = np.linspace(0, 200, data.shape[1])
            
            arr = preprocessor.process_hsqc(data, domain_h, domain_c)
            return torch.tensor(arr, dtype=torch.float, device=device)
        except Exception as e:
            print(f"Warning: Failed to process HSQC spectrum: {e}")
            return None

    # Process each spectrum type
    data_dict = {
        'ir': process_1d_spectrum(row.get('ir_spectra'), None, mode='ir'),
        'h_nmr': process_1d_spectrum(row.get('h_nmr_spectra'), None, mode='nmr'),
        'c_nmr': process_1d_spectrum(row.get('c_nmr_spectra'), None, mode='nmr'),
        'hsqc': process_hsqc_spectrum(row.get('hsqc_nmr_spectra'))
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

    # Initialize preprocessor
    preprocessor = SpectralPreprocessor(resample_size=resample_size)

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
            
            # Store processed data
            global_row_index = total_rows + i
            shard_data[global_row_index] = {
                'smiles': smiles_str,
                'spectra': {k: v.cpu() for k, v in data_dict.items() if v is not None}
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