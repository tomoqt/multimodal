from pathlib import Path
import click
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Union
from sklearn.model_selection import train_test_split
import regex as re
from scipy.interpolate import interp1d
import numpy as np
import pyarrow.parquet as pq
from rxn.chemutils.tokenization import tokenize_smiles
import tempfile
import uuid
import json
import os

################################################################################
# Utility functions
################################################################################

def split_data(data: pd.DataFrame, seed: int, val_size: float = 0.002) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/test/val sets."""
    train, test = train_test_split(data, test_size=0.001, random_state=seed, shuffle=True)
    train, val = train_test_split(train, test_size=val_size, random_state=seed, shuffle=True)
    return train, test, val


def tokenize_formula(formula: str) -> str:
    """Tokenize molecular formula with spaces between atoms/numbers."""
    return ' '.join(re.findall(r"[A-Z][a-z]?|\d+|.", formula)) + ' '


def process_hnmr(multiplets: List[Dict[str, Union[str, float, int]]]) -> str:
    """Process H-NMR multiplets into a tokenized string."""
    parts = ["1HNMR"]
    for peak in multiplets:
        range_max = float(peak["rangeMax"]) 
        range_min = float(peak["rangeMin"]) 
        peak_str = f"{range_max:.2f} {range_min:.2f} {peak['category']} {peak['nH']}H"
        
        js = str(peak["j_values"])
        if js != "None":
            split_js = js.split("_")
            split_js = list(filter(None, split_js))
            processed_js = [f"{float(j):.2f}" for j in split_js]
            j_part = "J " + " ".join(processed_js)
            peak_str += " " + j_part

        parts.append(peak_str)
    return " | ".join(parts)


def process_cnmr(carbon_nmr: List[Dict[str, Union[str, float, int]]]) -> str:
    """Process C-NMR peaks into a tokenized string."""
    parts = ["13CNMR"]
    for peak in carbon_nmr:
        val = round(float(peak["delta (ppm)"]), 1)
        parts.append(str(val))
    return " ".join(parts)


def process_ir(
    ir: np.ndarray,
    original_x: np.ndarray,
    tokenize: bool = False,
    interpolation_points: int = 1000
) -> Union[np.ndarray, str]:
    """
    Process IR spectrum with interpolation.

    - If tokenize=True, returns a string tokenization (0–100 scaling).
    - Otherwise returns a float numpy array normalized row by row.
    """
    # Pre-build target_x (only do it once if you prefer; done here for clarity)
    target_x = np.linspace(original_x[0], original_x[-1], interpolation_points)

    try:
        # Try a linear interpolation
        f = interp1d(original_x, ir, kind='linear', bounds_error=False, fill_value=0)
        interp_ir = f(target_x)
    except Exception:
        # Fall back if needed
        interp_ir = np.zeros_like(target_x)

    if tokenize:
        # Normalize to 0-100
        interp_ir = interp_ir + abs(min(interp_ir))
        max_val = np.max(interp_ir)
        if max_val > 0:
            interp_ir = (interp_ir / max_val) * 100
        
        # Round and convert to int -> string
        interp_ir = np.round(interp_ir, 0).astype(int).astype(str)
        return 'IR ' + ' '.join(interp_ir) + ' '
    else:
        # For model input as continuous array:
        # Normalize only non-zero portion (avoid dividing by zero on the entire array).
        non_zero = (interp_ir != 0)
        if non_zero.any():
            min_val = np.min(interp_ir[non_zero])
            max_val = np.max(interp_ir[non_zero])
            if not np.isclose(min_val, max_val):
                interp_ir[non_zero] = (interp_ir[non_zero] - min_val) / (max_val - min_val)
        return interp_ir


def process_msms(msms: List[List[float]]) -> str:
    """Process MS/MS peaks into a tokenized string."""
    parts = []
    for peak in msms:
        mz = round(peak[0], 1)
        intensity = round(peak[1], 1)
        parts.append(f"{mz} {intensity}")
    return " ".join(parts)


################################################################################
# Core parquet processing
################################################################################

def process_parquet_file(
    parquet_file: Path,
    h_nmr: bool,
    c_nmr: bool,
    ir: bool,
    pos_msms: bool,
    neg_msms: bool,
    formula: bool,
    original_x: np.ndarray,
    tokenize_ir: bool = False
) -> pd.DataFrame:
    """
    Process a single parquet file in chunks (row groups),
    returning a DataFrame containing tokenized results or raw IR arrays.
    """
    parquet_obj = pq.ParquetFile(parquet_file)
    
    columns = ['molecular_formula', 'smiles']
    if h_nmr:
        columns.append('h_nmr_peaks')
    if c_nmr:
        columns.append('c_nmr_peaks') 
    if ir:
        columns.append('ir_spectra')
    if pos_msms:
        columns.extend(['msms_positive_10ev', 'msms_positive_20ev', 'msms_positive_40ev'])
    if neg_msms:
        columns.extend(['msms_negative_10ev', 'msms_negative_20ev', 'msms_negative_40ev'])

    row_group_results = []

    for rg_idx in tqdm(range(parquet_obj.num_row_groups), desc=f"Processing {parquet_file.name}"):
        table = parquet_obj.read_row_group(rg_idx, columns=columns)
        chunk_df = table.to_pandas()

        chunk_results = []
        for row in chunk_df.itertuples(index=False):
            # Build tokenized input as a list, then join at the end
            token_parts = []

            # Add formula
            if formula:
                token_parts.append(tokenize_formula(row.molecular_formula))

            # Add H-NMR
            if h_nmr:
                h_nmr_string = process_hnmr(row.h_nmr_peaks)
                token_parts.append(h_nmr_string)

            # Add C-NMR
            if c_nmr:
                c_nmr_string = process_cnmr(row.c_nmr_peaks)
                token_parts.append(c_nmr_string)

            # IR: either tokenize or store in a separate column
            if ir:
                ir_data = process_ir(row.ir_spectra, original_x, tokenize=tokenize_ir)
                if tokenize_ir:
                    token_parts.append(ir_data)
                else:
                    # We'll store the IR array in a separate column, skip final chunk append
                    chunk_results.append({
                        'source': " ".join(token_parts).strip(),
                        'target': ' '.join(tokenize_smiles(row.smiles)),
                        'ir_data': ir_data
                    })
                    continue

            # Positive MS/MS
            if pos_msms:
                pos_msms_string = []
                pos_msms_string.append("E0Pos " + process_msms(row.msms_positive_10ev))
                pos_msms_string.append("E1Pos " + process_msms(row.msms_positive_20ev))
                pos_msms_string.append("E2Pos " + process_msms(row.msms_positive_40ev))
                token_parts.append(" ".join(pos_msms_string))

            # Negative MS/MS
            if neg_msms:
                neg_msms_string = []
                neg_msms_string.append("E0Neg " + process_msms(row.msms_negative_10ev))
                neg_msms_string.append("E1Neg " + process_msms(row.msms_negative_20ev))
                neg_msms_string.append("E2Neg " + process_msms(row.msms_negative_40ev))
                token_parts.append(" ".join(neg_msms_string))

            # If IR was tokenized or IR not included, store text result
            chunk_results.append({
                'source': " ".join(token_parts).strip(),
                'target': ' '.join(tokenize_smiles(row.smiles))
            })

        if chunk_results:
            row_group_results.append(pd.DataFrame(chunk_results))
    
    if row_group_results:
        return pd.concat(row_group_results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['source','target','ir_data'])


def save_set(data_set: pd.DataFrame, out_path: Path, set_type: str, pred_spectra: bool) -> None:
    """Save tokenized data (and IR arrays if present) to disk."""
    out_path.mkdir(parents=True, exist_ok=True)

    smiles_list = list(data_set['target'])
    spectra_list = list(data_set['source'])

    # Write source file
    with (out_path / f"src-{set_type}.txt").open("w") as f:
        if pred_spectra:
            for smi in smiles_list:
                f.write(f"{smi}\n")
        else:
            for spec in spectra_list:
                f.write(f"{spec}\n")
    
    # Write target file
    with (out_path / f"tgt-{set_type}.txt").open("w") as f:
        if pred_spectra:
            for spec in spectra_list:
                f.write(f"{spec}\n")
        else:
            for smi in smiles_list:
                f.write(f"{smi}\n")

    # Save IR data if it exists in this split
    if 'ir_data' in data_set.columns:
        print(f"Saving IR data for {set_type} set...")

        BATCH_SIZE = 1000
        total_rows = len(data_set)

        # Build a small initial batch to determine shape
        first_batch = []
        for i in range(min(BATCH_SIZE, total_rows)):
            ir_data = data_set.ir_data.iloc[i]
            if isinstance(ir_data, np.ndarray):
                first_batch.append(ir_data)
            elif isinstance(ir_data, list):
                first_batch.append(np.array(ir_data))
            else:
                # Attempt to parse from string if needed (rare)
                try:
                    arr = np.fromstring(ir_data.strip('[]'), sep=' ')
                    if len(arr) > 0:
                        first_batch.append(arr)
                except Exception as e:
                    print(f"Warning: Could not parse IR data: {e}")

        if not first_batch:
            print(f"Warning: No valid IR data found for {set_type} set")
            return
        
        # Stack the first batch to determine shape
        first_batch = np.stack(first_batch)
        array_shape = (total_rows, first_batch.shape[1])

        # Create memory-mapped array
        fp = np.memmap(out_path / f"ir-{set_type}.npy", dtype='float32', mode='w+', shape=array_shape)
        fp[:len(first_batch)] = first_batch

        # Process remaining data in batches
        for start_idx in range(BATCH_SIZE, total_rows, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_rows)
            print(f"Processing IR data batch {start_idx//BATCH_SIZE + 1}/{(total_rows + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            batch = []
            for i in range(start_idx, end_idx):
                ir_data = data_set.ir_data.iloc[i]
                if isinstance(ir_data, np.ndarray):
                    batch.append(ir_data)
                elif isinstance(ir_data, list):
                    batch.append(np.array(ir_data))
                else:
                    # Attempt string parse
                    try:
                        arr = np.fromstring(ir_data.strip('[]'), sep=' ')
                        if len(arr) > 0:
                            batch.append(arr)
                        else:
                            batch.append(np.zeros(first_batch.shape[1]))
                    except Exception as e:
                        print(f"Warning: Could not parse IR data at index {i}: {e}")
                        batch.append(np.zeros(first_batch.shape[1]))
            
            if batch:
                batch_array = np.stack(batch)
                fp[start_idx:end_idx] = batch_array

        # Flush changes to disk
        fp.flush()
        del fp
        print(f"Saved {total_rows} IR spectra with shape {array_shape}")


################################################################################
# Main CLI
################################################################################

@click.command()
@click.option(
    "--analytical_data",
    "-n",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the analytical data parquet files"
)
@click.option(
    "--out_path",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for tokenized data"
)
@click.option("--h_nmr", is_flag=True, help="Include H-NMR data")
@click.option("--c_nmr", is_flag=True, help="Include C-NMR data")
@click.option("--ir", is_flag=True, help="Include IR data")
@click.option("--pos_msms", is_flag=True, help="Include positive MS/MS data")
@click.option("--neg_msms", is_flag=True, help="Include negative MS/MS data")
@click.option("--formula", is_flag=True, help="Include molecular formula")
@click.option("--pred_spectra", is_flag=True, help="Predict spectra from SMILES")
@click.option("--seed", type=int, default=3245, help="Random seed for splitting")
@click.option("--tokenize_ir", is_flag=True, default=False, help="Tokenize IR data instead of returning raw values")
@click.option("--test_mode", is_flag=True, default=False, help="Process only first 3 files for testing")
def main(
    analytical_data: Path,
    out_path: Path,
    h_nmr: bool = False,
    c_nmr: bool = False,
    ir: bool = False,
    pos_msms: bool = False,
    neg_msms: bool = False,
    formula: bool = True,
    pred_spectra: bool = False,
    seed: int = 3245,
    tokenize_ir: bool = False,
    test_mode: bool = False
):
    """
    Create tokenized training data from analytical spectra
    in a memory-efficient manner by processing row groups 
    and writing intermediate results to disk.
    """
    print("\nProcessing analytical data...")

    # If IR is requested, read spectrum_dimensions.json once
    if ir:
        dimensions_path = os.path.join(
            'data_extraction', 'multimodal_spectroscopic_dataset', 
            'meta_data', 'spectrum_dimensions.json'
        )
        with open(dimensions_path, 'r') as f:
            spectrum_dims = json.load(f)
        original_x = np.array(spectrum_dims['ir_spectra']['dimensions'])
    else:
        original_x = None

    # Temporary dir for intermediate chunk results
    temp_dir = Path(tempfile.gettempdir()) / f"tokenized_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temporary chunk results will be stored in: {temp_dir}")

    # 1) Process each Parquet file
    chunk_files = []
    parquet_files = list(analytical_data.glob("*.parquet"))
    if test_mode:
        print("Running in test mode - processing only first 3 files")
        parquet_files = parquet_files[:3]

    for parquet_file in parquet_files:
        print(f"\nProcessing {parquet_file.name}...")
        df_chunk = process_parquet_file(
            parquet_file, h_nmr, c_nmr, ir, pos_msms, neg_msms,
            formula, original_x, tokenize_ir
        )
        chunk_file = temp_dir / f"{parquet_file.stem}_{uuid.uuid4().hex}.csv"
        df_chunk.to_csv(chunk_file, index=False)
        chunk_files.append(chunk_file)

    # 2) Combine chunk files
    print("\nCombining chunk files...")
    tokenised_data_list = []
    for cfile in chunk_files:
        tokenised_data_list.append(pd.read_csv(cfile))
    tokenised_data = pd.concat(tokenised_data_list, ignore_index=True)

    # 3) De-duplicate
    tokenised_data = tokenised_data.drop_duplicates(subset="source")
    print(f"\nTotal samples after processing: {len(tokenised_data)}")

    # 4) Split data
    print("\nSplitting into train/val/test sets...")
    train_set, test_set, val_set = split_data(tokenised_data, seed)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # 5) Save splits
    print("\nSaving tokenized data...")
    out_data_path = out_path / "data"
    save_set(test_set, out_data_path, "test", pred_spectra)
    save_set(train_set, out_data_path, "train", pred_spectra)
    save_set(val_set, out_data_path, "val", pred_spectra)
    
    print(f"\nTokenized data saved to {out_data_path}")
    print(f"Cleaning up temp directory {temp_dir}...")

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Done.")


if __name__ == '__main__':
    main()
