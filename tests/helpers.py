import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data.build_vocab import build_vocabulary
from data.create_tokenized_dataset_faster import process_parquet_file, save_set


def build_raw_dataframe(num_rows: int = 24, spectrum_points: int = 64) -> pd.DataFrame:
    smiles_bank = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
    rows = []
    for i in range(num_rows):
        smiles = smiles_bank[i % len(smiles_bank)]
        scale = 0.5 + (i % 5) * 0.1
        ir = (np.linspace(0.0, 1.0, spectrum_points) * scale).tolist()
        rows.append(
            {
                "molecular_formula": "C2H6O",
                "smiles": smiles,
                "h_nmr_peaks": [
                    {
                        "rangeMax": 1.2 + 0.01 * i,
                        "rangeMin": 1.0 + 0.01 * i,
                        "category": "d",
                        "nH": 3,
                        "j_values": "7.0_",
                    }
                ],
                "c_nmr_peaks": [{"delta (ppm)": 12.3 + 0.01 * i}],
                "ir_spectra": ir,
            }
        )
    return pd.DataFrame(rows)


def materialize_tokenized_dataset(base_dir: Path, num_rows: int = 24) -> Tuple[Path, Dict[str, int]]:
    """
    Build a minimal tokenized dataset from synthetic raw parquet input.

    Returns:
      tokenized_data_dir, nmr_tokenizer
    """
    raw_dir = base_dir / "raw"
    out_dir = base_dir / "tokenized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = build_raw_dataframe(num_rows=num_rows)
    parquet_file = raw_dir / "chunk.parquet"
    raw_df.to_parquet(parquet_file, index=False)

    processed = process_parquet_file(
        parquet_file=parquet_file,
        h_nmr=True,
        c_nmr=True,
        ir=True,
        pos_msms=False,
        neg_msms=False,
        formula=True,
        original_x=np.linspace(400, 4000, len(raw_df.iloc[0]["ir_spectra"])),
        tokenize_ir=False,
    )

    # Deterministic tiny split for tests.
    train = processed.iloc[:-4].reset_index(drop=True)
    val = processed.iloc[-4:-2].reset_index(drop=True)
    test = processed.iloc[-2:].reset_index(drop=True)

    tokenized_data_dir = out_dir / "data"
    save_set(train, tokenized_data_dir, "train", pred_spectra=False)
    save_set(val, tokenized_data_dir, "val", pred_spectra=False)
    save_set(test, tokenized_data_dir, "test", pred_spectra=False)

    vocab_txt = out_dir / "vocab.txt"
    build_vocabulary(tokenized_dir=tokenized_data_dir, output_vocab_file=vocab_txt, add_special_tokens=True, save_json=True)
    vocab_json = vocab_txt.with_suffix(".json")
    with vocab_json.open("r", encoding="utf-8") as f:
        nmr_tokenizer = json.load(f)
    return tokenized_data_dir, nmr_tokenizer
