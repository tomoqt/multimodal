#!/usr/bin/env python

"""
Offline Dataset Creation Script (Chunked-Writing Version)

This script processes multiple Parquet files containing IR and NMR spectral data along with SMILES strings,
then consolidates them into efficient binary (.bin) files with an accompanying index for quick access during training.

Usage:
    python build_mmap_dataset.py
    # Or specify custom paths:
    python build_mmap_dataset.py --parquet_dir path/to/parquet_files \
                                 --meta_path path/to/meta_data_dict.json \
                                 --out_dir path/to/output_directory
"""

import argparse
import json
import numpy as np
import pyarrow.parquet as pq
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

##############################################################################
# Utility: Write a float32 array directly to an open binary file.
# Returns (start_offset, length_in_floats).
# If arr is None/empty, returns sentinel (-1, 0).
##############################################################################
def write_float_array_to_file(file_obj, arr):
    """
    Write float32 array arr to the open file_obj.
    Returns:
      (start_offset, length)
    where start_offset is the offset in "float32 units" (not bytes).
    """
    if arr is None or len(arr) == 0:
        return -1, 0

    # current position in bytes => convert to float32 offset
    start_offset = file_obj.tell() // 4  # each float32 = 4 bytes
    arr = arr.astype(np.float32, copy=False)  # ensure float32, no extra copy if possible
    file_obj.write(arr.tobytes())
    length = len(arr)
    return start_offset, length

##############################################################################
# Helper: Generate a domain array if not given in the row (optional).
##############################################################################
def generate_domain_if_none(spectrum_data, spectrum_type):
    if spectrum_data is None or len(spectrum_data) == 0:
        return None
    length = len(spectrum_data)
    # Example domain creation:
    if spectrum_type == 'ir':
        return np.linspace(400, 4000, length, dtype=np.float32)
    elif spectrum_type == 'h_nmr':
        return np.linspace(0, 12, length, dtype=np.float32)
    elif spectrum_type == 'c_nmr':
        return np.linspace(0, 200, length, dtype=np.float32)
    else:
        return None  # or raise NotImplementedError

##############################################################################
# Main function to build memory-mapped dataset
##############################################################################
def build_mmap_dataset(parquet_dir, meta_file, out_dir):
    """
    Reads multiple Parquet files of spectral data + SMILES, writes:
      - spectra_data.bin   (float32 arrays for IR, H-NMR, C-NMR + domains)
      - smiles_data.bin    (raw bytes for SMILES)
      - spectra_index.npy  (index array with offsets, shape = [num_rows, 14])

    Index format per row (total 14 columns):
      0: IR_data_off,    1: IR_data_len,
      2: IR_dom_off,     3: IR_dom_len,
      4: HNMR_data_off,  5: HNMR_data_len,
      6: HNMR_dom_off,   7: HNMR_dom_len,
      8: CNMR_data_off,  9: CNMR_data_len,
     10: CNMR_dom_off,  11: CNMR_dom_len,
     12: SMILES_off,     13: SMILES_len
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load metadata (if needed for reference)
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    # 2) Gather all Parquet files
    parquet_path = Path(parquet_dir)
    parquet_files = sorted(parquet_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_dir}")
    print(f"Found {len(parquet_files)} Parquet files.")

    # 3) Prepare output binary files
    spectra_data_path = out_dir / "spectra_data.bin"
    smiles_data_path = out_dir / "smiles_data.bin"
    spectra_index_path = out_dir / "spectra_index.npy"

    # We'll store only offsets in memory
    index_entries = []
    global_row_count = 0

    # Open output files in binary write mode
    with open(spectra_data_path, "wb") as spectra_f, open(smiles_data_path, "wb") as smiles_f:
        # Iterate over all Parquet files
        for parquet_file in tqdm(parquet_files, desc="Processing Parquet files"):
            parquet_obj = pq.ParquetFile(parquet_file)
            num_row_groups = parquet_obj.num_row_groups

            # Iterate row groups
            for rg_idx in range(num_row_groups):
                table = parquet_obj.read_row_group(rg_idx, columns=[
                    'smiles',
                    'ir_spectra',
                    'h_nmr_spectra',
                    'c_nmr_spectra'
                ])
                df = table.to_pandas()

                # Iterate rows
                for i in range(len(df)):
                    row = df.iloc[i]
                    smiles_str = row['smiles'] if pd.notnull(row['smiles']) else ""
                    # Convert SMILES to bytes
                    smiles_bytes = smiles_str.encode('utf-8')
                    smiles_off = smiles_f.tell()
                    smiles_f.write(smiles_bytes)
                    smiles_len = len(smiles_bytes)

                    # IR
                    ir_data = row.get('ir_spectra', None)
                    if isinstance(ir_data, str):
                        ir_data = json.loads(ir_data)
                    ir_data = np.array(ir_data, dtype=np.float32) if ir_data is not None else None
                    ir_dom = generate_domain_if_none(ir_data, 'ir')

                    # H-NMR
                    h_nmr_data = row.get('h_nmr_spectra', None)
                    if isinstance(h_nmr_data, str):
                        h_nmr_data = json.loads(h_nmr_data)
                    h_nmr_data = np.array(h_nmr_data, dtype=np.float32) if h_nmr_data is not None else None
                    h_nmr_dom = generate_domain_if_none(h_nmr_data, 'h_nmr')

                    # C-NMR
                    c_nmr_data = row.get('c_nmr_spectra', None)
                    if isinstance(c_nmr_data, str):
                        c_nmr_data = json.loads(c_nmr_data)
                    c_nmr_data = np.array(c_nmr_data, dtype=np.float32) if c_nmr_data is not None else None
                    c_nmr_dom = generate_domain_if_none(c_nmr_data, 'c_nmr')

                    # Write IR data + domain directly to file
                    ir_data_off, ir_data_len = write_float_array_to_file(spectra_f, ir_data)
                    ir_dom_off, ir_dom_len   = write_float_array_to_file(spectra_f, ir_dom)

                    # Write H-NMR data + domain
                    hnm_data_off, hnm_data_len = write_float_array_to_file(spectra_f, h_nmr_data)
                    hnm_dom_off,  hnm_dom_len  = write_float_array_to_file(spectra_f, h_nmr_dom)

                    # Write C-NMR data + domain
                    cnm_data_off, cnm_data_len = write_float_array_to_file(spectra_f, c_nmr_data)
                    cnm_dom_off,  cnm_dom_len  = write_float_array_to_file(spectra_f, c_nmr_dom)

                    # Build index for this row (14 columns)
                    index_entries.append([
                        ir_data_off,    ir_data_len,
                        ir_dom_off,     ir_dom_len,
                        hnm_data_off,   hnm_data_len,
                        hnm_dom_off,    hnm_dom_len,
                        cnm_data_off,   cnm_data_len,
                        cnm_dom_off,    cnm_dom_len,
                        smiles_off,     smiles_len
                    ])

                    global_row_count += 1

    # After writing all data, save the index as a NumPy array
    index_array = np.array(index_entries, dtype=np.int64)
    np.save(spectra_index_path, index_array)

    print(f"\nDone! Processed {global_row_count} rows.")
    print(f" - Wrote all float data directly to: {spectra_data_path}")
    print(f" - Wrote all SMILES to:             {smiles_data_path}")
    print(f" - Float data offset/index entries: {index_array.shape[0]}")
    print(f" - Final index saved at:            {spectra_index_path}")

##############################################################################
# CLI / Main
##############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Build memory-mapped dataset from Parquet files (chunked-writing).")
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default="data_extraction/multimodal_spectroscopic_dataset",
        help="Directory containing Parquet files"
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="data_extraction/multimodal_spectroscopic_dataset/meta_data/meta_data_dict.json",
        help="Path to meta_data_dict.json (for domain info)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="preprocessed_binaries",
        help="Directory to store output .bin and .npy files"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    build_mmap_dataset(
        parquet_dir=args.parquet_dir,
        meta_file=args.meta_path,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()
