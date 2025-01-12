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

def split_data(data: pd.DataFrame, seed: int, val_size: float = 0.002) -> Tuple[pd.DataFrame]:
    """Split data into train/test/val sets."""
    train, test = train_test_split(data, test_size=0.1, random_state=seed, shuffle=True)
    train, val = train_test_split(train, test_size=val_size, random_state=seed, shuffle=True)
    return train, test, val

def tokenize_formula(formula: str) -> str:
    """Tokenize molecular formula with spaces between atoms/numbers."""
    return ' '.join(re.findall("[A-Z][a-z]?|\d+|.", formula)) + ' '

def process_hnmr(multiplets: List[Dict[str, Union[str, float, int]]]) -> str:
    """Process H-NMR multiplets into a tokenized string."""
    multiplet_str = "1HNMR "
    for peak in multiplets:
        range_max = float(peak["rangeMax"]) 
        range_min = float(peak["rangeMin"]) 

        formatted_peak = "{:.2f} {:.2f} {} {}H ".format(
            range_max, 
            range_min,
            peak["category"],
            peak["nH"]
        )
        
        js = str(peak["j_values"])
        if js != "None":
            split_js = js.split("_")
            split_js = list(filter(None, split_js))
            processed_js = ["{:.2f}".format(float(j)) for j in split_js]
            formatted_js = "J " + " ".join(processed_js)
            formatted_peak += formatted_js

        multiplet_str += formatted_peak.strip() + " | "

    return multiplet_str[:-2]  # Remove last separator

def process_cnmr(carbon_nmr: List[Dict[str, Union[str, float, int]]]) -> str:
    """Process C-NMR peaks into a tokenized string."""
    nmr_string = "13CNMR "
    for peak in carbon_nmr:
        nmr_string += str(round(float(peak["delta (ppm)"]), 1)) + " "
    return nmr_string

def process_ir(ir: np.ndarray, tokenize: bool = False, interpolation_points: int = 1000) -> Union[tuple[np.ndarray, np.ndarray], str]:
    """Process IR spectrum with cubic spline interpolation (falls back to linear)."""
    # Load the actual domain from spectrum_dimensions.json
    dimensions_path = os.path.join('data_extraction', 'multimodal_spectroscopic_dataset', 
                                 'meta_data', 'spectrum_dimensions.json')
    with open(dimensions_path, 'r') as f:
        spectrum_dims = json.load(f)
    original_x = np.array(spectrum_dims['ir_spectra']['dimensions'])
    
    if tokenize:
        # Original tokenization logic for text format
        interpolation_x = np.linspace(min(original_x), max(original_x), interpolation_points)
        interp = interp1d(original_x, ir)
        interp_ir = interp(interpolation_x)
        
        # Normalize to 0-100 range
        interp_ir = interp_ir + abs(min(interp_ir))
        interp_ir = (interp_ir / max(interp_ir)) * 100 
        interp_ir = np.round(interp_ir, decimals=0).astype(int).astype(str)
        return 'IR ' + ' '.join(interp_ir) + ' '
    else:
        # Interpolate for model input using the actual domain range
        target_x = np.linspace(min(original_x), max(original_x), interpolation_points)
        
        try:
            # Try cubic spline first
            if len(original_x) > 3:  # Need at least 4 points for cubic
                eps = 1e-10
                if not np.any(np.diff(original_x) < eps):
                    f = interp1d(original_x, ir, kind='cubic', bounds_error=False, fill_value=0)
                    interp_ir = f(target_x)
                    # Normalize
                    non_zero = interp_ir != 0
                    if non_zero.any():
                        min_val = np.min(interp_ir[non_zero])
                        max_val = np.max(interp_ir[non_zero])
                        if not np.isclose(min_val, max_val):
                            interp_ir[non_zero] = (interp_ir[non_zero] - min_val) / (max_val - min_val)
                    return interp_ir
        except:
            pass  # Fall through to linear interpolation
            
        # Linear interpolation as fallback
        f = interp1d(original_x, ir, kind='linear', bounds_error=False, fill_value=0)
        interp_ir = f(target_x)
        
        # Normalize
        non_zero = interp_ir != 0
        if non_zero.any():
            min_val = np.min(interp_ir[non_zero])
            max_val = np.max(interp_ir[non_zero])
            if not np.isclose(min_val, max_val):
                interp_ir[non_zero] = (interp_ir[non_zero] - min_val) / (max_val - min_val)
        
        return interp_ir

def process_msms(msms: List[List[float]]) -> str:
    """Process MS/MS peaks into a tokenized string."""
    msms_string = ''
    for peak in msms:
        msms_string = msms_string + "{:.1f} {:.1f} ".format(
            round(peak[0], 1), round(peak[1], 1)
        )
    return msms_string

def process_parquet_file(parquet_file: Path, h_nmr: bool, c_nmr: bool, ir: bool, 
                        pos_msms: bool, neg_msms: bool, formula: bool, tokenize_ir: bool = False) -> pd.DataFrame:
    """
    Process a single parquet file in chunks (row groups),
    returning a DataFrame containing tokenized results or raw IR.
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

    # Read each row group so we don't load the entire file into memory at once
    for rg_idx in tqdm(range(parquet_obj.num_row_groups), desc=f"Processing {parquet_file.name}"):
        table = parquet_obj.read_row_group(rg_idx, columns=columns)
        chunk_df = table.to_pandas()
        
        chunk_results = []
        for i in range(len(chunk_df)):
            row = chunk_df.iloc[i]
            tokenized_formula_str = tokenize_formula(row['molecular_formula']) if formula else ''
            tokenized_input = tokenized_formula_str

            if h_nmr:
                h_nmr_string = process_hnmr(row['h_nmr_peaks'])
                tokenized_input += h_nmr_string

            if c_nmr:
                c_nmr_string = process_cnmr(row['c_nmr_peaks'])
                tokenized_input += c_nmr_string

            if ir:
                ir_data = process_ir(row['ir_spectra'], tokenize=tokenize_ir)
                if tokenize_ir:
                    # Add the tokenized IR to the string
                    tokenized_input += ir_data
                else:
                    # Store raw IR data in a separate column
                    chunk_results.append({
                        'source': tokenized_input.strip(),
                        'target': ' '.join(tokenize_smiles(row['smiles'])),
                        'ir_data': ir_data
                    })
                    continue  # Move on

            if pos_msms:
                pos_msms_string = ''
                pos_msms_string += "E0Pos " + process_msms(row["msms_positive_10ev"])
                pos_msms_string += "E1Pos " + process_msms(row["msms_positive_20ev"])
                pos_msms_string += "E2Pos " + process_msms(row["msms_positive_40ev"])
                tokenized_input += pos_msms_string

            if neg_msms:
                neg_msms_string = ''
                neg_msms_string += "E0Neg " + process_msms(row["msms_negative_10ev"])
                neg_msms_string += "E1Neg " + process_msms(row["msms_negative_20ev"])
                neg_msms_string += "E2Neg " + process_msms(row["msms_negative_40ev"])
                tokenized_input += neg_msms_string

            # If we've tokenized IR, or if IR isn't included, store the normal data
            chunk_results.append({
                'source': tokenized_input.strip(),
                'target': ' '.join(tokenize_smiles(row['smiles']))
            })

        if chunk_results:
            row_group_results.append(pd.DataFrame(chunk_results))
    
    if row_group_results:
        return pd.concat(row_group_results, ignore_index=True)
    else:
        # In case the file had no data or columns
        return pd.DataFrame(columns=['source','target','ir_data'])

def save_set(data_set: pd.DataFrame, out_path: Path, set_type: str, pred_spectra: bool) -> None:
    """Save tokenized data to files."""
    out_path.mkdir(parents=True, exist_ok=True)

    smiles = list(data_set.target)
    spectra = list(data_set.source)

    # Write source file
    with (out_path / f"src-{set_type}.txt").open("w") as f:
        src = smiles if pred_spectra else spectra
        for item in src:
            f.write(f"{item}\n")
    
    # Write target file
    with (out_path / f"tgt-{set_type}.txt").open("w") as f:
        tgt = spectra if pred_spectra else smiles
        for item in tgt:
            f.write(f"{item}\n")

    # Save IR data if it exists
    if 'ir_data' in data_set.columns:
        print(f"Saving IR data for {set_type} set...")
        ir_arrays = []
        for ir_data in data_set.ir_data:
            if isinstance(ir_data, (np.ndarray, list)):
                ir_arrays.append(np.array(ir_data))
            elif isinstance(ir_data, str):
                try:
                    # Convert string representation of array to numpy array
                    ir_data = ir_data.strip('[]')
                    ir_array = np.fromstring(ir_data, sep=' ')
                    if len(ir_array) > 0:  # Only append if we got valid data
                        ir_arrays.append(ir_array)
                except Exception as e:
                    print(f"Warning: Could not parse IR data: {e}")
                    continue
        
        if ir_arrays:
            # Verify all arrays have the same length
            lengths = [len(arr) for arr in ir_arrays]
            if len(set(lengths)) > 1:
                print(f"Warning: Inconsistent IR array lengths found: {set(lengths)}")
                # Use the most common length
                from collections import Counter
                target_length = Counter(lengths).most_common(1)[0][0]
                ir_arrays = [arr for arr in ir_arrays if len(arr) == target_length]
                
            if ir_arrays:  # If we still have valid arrays
                ir_array = np.stack(ir_arrays)
                np.save(out_path / f"ir-{set_type}.npy", ir_array)
                print(f"Saved {len(ir_arrays)} IR spectra with shape {ir_array.shape}")
            else:
                print(f"Warning: No valid IR arrays to save for {set_type} set")
        else:
            print(f"Warning: No valid IR data found for {set_type} set")

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

    # Create a temporary directory to store intermediate chunk results
    temp_dir = Path(tempfile.gettempdir()) / f"tokenized_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temporary chunk results will be stored in: {temp_dir}")

    # 1) For each parquet, process row-groups and store the results into chunk-based files
    chunk_files = []
    parquet_files = list(analytical_data.glob("*.parquet"))
    
    if test_mode:
        print("Running in test mode - processing only first 3 files")
        parquet_files = parquet_files[:3]
    
    for parquet_file in parquet_files:
        print(f"\nProcessing {parquet_file.name}...")
        df_chunk = process_parquet_file(parquet_file, h_nmr, c_nmr, ir, pos_msms, neg_msms, formula, tokenize_ir)
        # Write each parquet's result to a temporary CSV (or parquet) on disk
        chunk_file = temp_dir / f"{parquet_file.stem}_{uuid.uuid4().hex}.csv"
        df_chunk.to_csv(chunk_file, index=False)
        chunk_files.append(chunk_file)

    # 2) Load all chunk-files back into a single DataFrame
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

    # If desired, remove temporary directory with all chunk files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Done.")

if __name__ == '__main__':
    main()
#python create_tokenized_dataset_smallram.py --analytical_data "data_extraction/multimodal_spectroscopic_dataset" --out_path "tokenized_baseline" --h_nmr --c_nmr --ir --formula