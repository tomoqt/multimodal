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

def split_data(data: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame]:
    """Split data into train/test/val sets."""
    train, test = train_test_split(data, test_size=0.1, random_state=seed, shuffle=True)
    train, val = train_test_split(train, test_size=0.05, random_state=seed, shuffle=True)
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

def process_ir(ir: np.ndarray, tokenize: bool = False, interpolation_points: int = 400) -> Union[tuple[np.ndarray, np.ndarray], str]:
    """Process IR spectrum either as tokenized string or as raw data for SpectralEncoder.
    
    Args:
        ir: Input IR spectrum
        tokenize: If True, returns tokenized string. If False, returns tuple for SpectralEncoder
        interpolation_points: Number of points to interpolate to if tokenizing
    
    Returns:
        Either:
            - tuple[np.ndarray, np.ndarray]: (intensities, wavenumbers) for SpectralEncoder
            - str: Tokenized IR string if tokenize=True
    """
    # Generate wavenumber domain (assuming 1800 points from 400-4000 cm⁻¹)
    original_x = np.linspace(400, 4000, 1800)
    
    if tokenize:
        # Original tokenization logic
        interpolation_x = np.linspace(400, 4000, interpolation_points)
        interp = interp1d(original_x, ir)
        interp_ir = interp(interpolation_x)

        # Normalize to 0-100 range
        interp_ir = interp_ir + abs(min(interp_ir))
        interp_ir = (interp_ir / max(interp_ir)) * 100 
        interp_ir = np.round(interp_ir, decimals=0).astype(int).astype(str)
        
        return 'IR ' + ' '.join(interp_ir) + ' '
    else:
        # Return format expected by SpectralEncoder
        return ir, original_x

def process_msms(msms: List[List[float]]) -> str:
    """Process MS/MS peaks into a tokenized string."""
    msms_string = ''
    for peak in msms:
        msms_string = msms_string + "{:.1f} {:.1f} ".format(
            round(peak[0], 1), round(peak[1], 1)
        )
    return msms_string

def process_parquet_file(parquet_file: Path, h_nmr: bool, c_nmr: bool, ir: bool, 
                        pos_msms: bool, neg_msms: bool, formula: bool, tokenize_ir: bool = False) -> List[Dict]:
    """Process a single parquet file in chunks using row groups."""
    input_list = []
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

    for rg_idx in tqdm(range(parquet_obj.num_row_groups), desc=f"Processing {parquet_file.name}"):
        table = parquet_obj.read_row_group(rg_idx, columns=columns)
        chunk_df = table.to_pandas()
        
        for i in range(len(chunk_df)):
            row = chunk_df.iloc[i]
            tokenized_formula = tokenize_formula(row['molecular_formula'])
            tokenized_input = tokenized_formula if formula else ''

            if h_nmr:
                h_nmr_string = process_hnmr(row['h_nmr_peaks'])
                tokenized_input += h_nmr_string

            if c_nmr:
                c_nmr_string = process_cnmr(row['c_nmr_peaks'])
                tokenized_input += c_nmr_string

            if ir:
                ir_data = process_ir(row['ir_spectra'], tokenize=tokenize_ir)
                if tokenize_ir:
                    tokenized_input += ir_data
                else:
                    # Store the tuple directly in the dictionary
                    input_list.append({
                        'source': tokenized_input.strip(),
                        'target': ' '.join(tokenize_smiles(row['smiles'])),
                        'ir_data': ir_data
                    })
                    continue  # Skip the rest of this iteration

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

            # Only add the basic dictionary if we're tokenizing IR or not using IR
            if tokenize_ir or not ir:
                input_list.append({
                    'source': tokenized_input.strip(),
                    'target': ' '.join(tokenize_smiles(row['smiles']))
                })

    return input_list

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
    tokenize_ir: bool = False
):
    """Create tokenized training data from analytical spectra."""
    print("\nProcessing analytical data...")
    
    # Process all parquet files
    tokenised_data_list = []
    for parquet_file in analytical_data.glob("*.parquet"):
        print(f"\nProcessing {parquet_file.name}...")
        chunk_data = process_parquet_file(
            parquet_file, h_nmr, c_nmr, ir, pos_msms, neg_msms, formula, tokenize_ir
        )
        tokenised_data_list.extend(chunk_data)

    tokenised_data = pd.DataFrame(tokenised_data_list)
    tokenised_data = tokenised_data.drop_duplicates(subset="source")
    print(f"\nTotal samples after processing: {len(tokenised_data)}")

    # Split data
    print("\nSplitting into train/val/test sets...")
    train_set, test_set, val_set = split_data(tokenised_data, seed)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Save splits
    print("\nSaving tokenized data...")
    out_data_path = out_path / "data"
    save_set(test_set, out_data_path, "test", pred_spectra)
    save_set(train_set, out_data_path, "train", pred_spectra)
    save_set(val_set, out_data_path, "val", pred_spectra)
    
    print(f"\nTokenized data saved to {out_data_path}")

if __name__ == '__main__':
    main()

"""
Standard usage:
    # Multi-line version:
    python create_tokenized_data.py \
        --analytical_data "data_extraction/multimodal_spectroscopic_dataset" \
        --out_path "tokenized_baseline" \
        --h_nmr --c_nmr --ir --formula

    # Single-line version:
    python create_tokenized_data.py --analytical_data "data_extraction/multimodal_spectroscopic_dataset" --out_path "tokenized_baseline" --h_nmr --c_nmr --ir --formula

This will:
1. Read Parquet files from: data_extraction/multimodal_spectroscopic_dataset/
2. Create tokenized output in: tokenized_baseline/data/
3. Include H-NMR, C-NMR, IR spectra and molecular formula in tokenization
"""