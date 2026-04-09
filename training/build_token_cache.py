#!/usr/bin/env python3
"""
Build persistent token and optional IR caches for train/val/test splits.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.smiles_tokenizer import SmilesTokenizer
from training.core_dataset import SpectralSmilesDataset
from training.train_autoregressive import load_config, load_nmr_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persistent token/IR caches for tokenized splits.")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    tokenized_dir = Path(data_cfg["tokenized_dir"])
    if not tokenized_dir.exists():
        raise FileNotFoundError(f"Tokenized directory not found: {tokenized_dir}")

    smiles_tokenizer = SmilesTokenizer(vocab_file=str(Path(__file__).with_name("vocab.txt")))
    nmr_tokenizer = load_nmr_tokenizer(tokenized_dir)

    cache_dir = data_cfg.get("cache_dir")
    splits = ("train", "val", "test")
    total_start = time.perf_counter()
    for split in splits:
        start = time.perf_counter()
        dataset = SpectralSmilesDataset(
            data_dir=str(tokenized_dir),
            smiles_tokenizer=smiles_tokenizer,
            spectral_tokenizer=nmr_tokenizer,
            split=split,
            max_smiles_len=model_cfg["max_seq_length"],
            max_nmr_len=model_cfg["max_nmr_length"],
            pretokenize=True,
            preload_ir=data_cfg.get("preload_ir", False),
            use_disk_cache=True,
            write_disk_cache=True,
            cache_dir=cache_dir,
            ir_cache_mode=data_cfg.get("ir_cache_mode", "none"),
            ir_cache_dtype=data_cfg.get("ir_cache_dtype", "float32"),
        )
        elapsed = time.perf_counter() - start
        token_cache_path = str(dataset.token_cache_path) if dataset.token_cache_path else "n/a"
        token_state = "hit" if dataset.token_cache_hit else "built"
        line = (
            f"[cache] split={split} samples={len(dataset)} "
            f"token={token_state} token_path={token_cache_path} "
            f"time_s={elapsed:.2f}"
        )
        if data_cfg.get("ir_cache_mode", "none") == "pt":
            ir_state = "hit" if dataset.ir_cache_hit else "built"
            ir_cache_path = str(dataset.ir_cache_path) if dataset.ir_cache_path else "n/a"
            line += f" ir={ir_state} ir_path={ir_cache_path} ir_dtype={data_cfg.get('ir_cache_dtype', 'float32')}"
        print(line)

    total_elapsed = time.perf_counter() - total_start
    print(f"[cache] done total_time_s={total_elapsed:.2f}")


if __name__ == "__main__":
    main()
