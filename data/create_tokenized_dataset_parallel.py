from pathlib import Path
import click
import pandas as pd
import numpy as np
import tempfile
import uuid
import json
import os

from create_tokenized_dataset_faster import (
    split_data, tokenize_formula, process_hnmr, process_cnmr, process_ir, process_msms,
    process_parquet_file, save_set
)

from concurrent.futures import ProcessPoolExecutor, as_completed


# TODO: hynopump@ parallelize

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

    # 1) Process each Parquet file in parallel
    chunk_files = []
    parquet_files = list(analytical_data.glob("*.parquet"))
    if test_mode:
        print("Running in test mode - processing only first 3 files")
        parquet_files = parquet_files[:3]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        # Submit tasks to the executor and store futures in a dictionary
        future_to_file = {}
        for parquet_file in parquet_files:
            print(f"Submitting {parquet_file.name} for processing...")
            future = executor.submit(
                process_parquet_file,
                parquet_file, h_nmr, c_nmr, ir, pos_msms, neg_msms,
                formula, original_x, tokenize_ir
            )
            future_to_file[future] = parquet_file

        # Collect results as they complete
        for future in as_completed(future_to_file):
            parquet_file = future_to_file[future]
            try:
                df_chunk = future.result()
                chunk_file = temp_dir / f"{parquet_file.stem}_{uuid.uuid4().hex}.csv"
                df_chunk.to_csv(chunk_file, index=False)
                chunk_files.append(chunk_file)
                print(f"Completed processing {parquet_file.name}")
            except Exception as e:
                print(f"Error processing {parquet_file.name}: {e}")

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
