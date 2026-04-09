#!/usr/bin/env python3
"""
Create modality-filtered or size-limited variants of an existing tokenized dataset.

Typical usage:

python data/filter_tokenized_modalities.py \
  --input_dir data/tokenized_full_candidate_newsplit \
  --output_dir data/tokenized_official_nmiracle_like \
  --drop_formula

python data/filter_tokenized_modalities.py \
  --input_dir data/tokenized_official_nmiracle_like \
  --output_dir data/tokenized_official_nmiracle_like_smoke \
  --max_train 2048 --max_val 256 --max_test 256
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def split_source_tokens(source_text: str) -> Tuple[List[str], List[str], List[str]]:
    tokens = source_text.split()
    idx_1h = tokens.index("1HNMR") if "1HNMR" in tokens else -1
    idx_13c = tokens.index("13CNMR") if "13CNMR" in tokens else -1

    if idx_1h >= 0:
        formula_tokens = tokens[:idx_1h]
    elif idx_13c >= 0:
        formula_tokens = tokens[:idx_13c]
    else:
        formula_tokens = tokens

    one_h_tokens: List[str] = []
    if idx_1h >= 0:
        end_1h = idx_13c if idx_13c >= 0 else len(tokens)
        one_h_tokens = tokens[idx_1h:end_1h]

    carbon_tokens: List[str] = []
    if idx_13c >= 0:
        carbon_tokens = tokens[idx_13c:]

    return formula_tokens, one_h_tokens, carbon_tokens


def filter_source_text(source_text: str, keep_formula: bool, keep_1h: bool, keep_13c: bool) -> str:
    formula_tokens, one_h_tokens, carbon_tokens = split_source_tokens(source_text)
    out: List[str] = []
    if keep_formula:
        out.extend(formula_tokens)
    if keep_1h:
        out.extend(one_h_tokens)
    if keep_13c:
        out.extend(carbon_tokens)
    return " ".join(out).strip()


def read_lines(path: Path, max_rows: int | None = None) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        if max_rows is None:
            return [line.rstrip("\n") for line in f]
        lines: List[str] = []
        for idx, line in enumerate(f):
            if idx >= max_rows:
                break
            lines.append(line.rstrip("\n"))
        return lines


def write_lines(path: Path, rows: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row}\n")


def infer_ir_shape(ir_path: Path, num_rows: int) -> Tuple[int, int]:
    raw = np.memmap(ir_path, dtype="float32", mode="r")
    if num_rows <= 0:
        raise ValueError(f"Cannot infer IR shape with num_rows={num_rows}")
    if raw.size % num_rows != 0:
        raise ValueError(
            f"IR file {ir_path} with {raw.size} float32 entries is incompatible with {num_rows} rows."
        )
    feat_dim = raw.size // num_rows
    return num_rows, feat_dim


def copy_ir_subset(src_ir: Path, dst_ir: Path, src_rows: int, keep_rows: int) -> None:
    _, feat_dim = infer_ir_shape(src_ir, src_rows)
    src = np.memmap(src_ir, dtype="float32", mode="r", shape=(src_rows, feat_dim))
    dst = np.memmap(dst_ir, dtype="float32", mode="w+", shape=(keep_rows, feat_dim))
    block = 4096
    for start in range(0, keep_rows, block):
        end = min(start + block, keep_rows)
        dst[start:end] = src[start:end]
    dst.flush()
    del dst
    del src


def symlink_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def maybe_copy_vocab(input_root: Path, output_root: Path, symlink: bool) -> None:
    for name in ("vocab.json", "vocab.txt"):
        src = input_root / name
        if src.exists():
            symlink_or_copy(src, output_root / name, symlink=symlink)


def process_split(
    input_data_dir: Path,
    output_data_dir: Path,
    split: str,
    *,
    keep_formula: bool,
    keep_1h: bool,
    keep_13c: bool,
    keep_ir: bool,
    max_rows: int | None,
    symlink_passthrough: bool,
) -> Tuple[int, int]:
    src_path = input_data_dir / f"src-{split}.txt"
    tgt_path = input_data_dir / f"tgt-{split}.txt"
    ir_path = input_data_dir / f"ir-{split}.npy"

    sources = read_lines(src_path, max_rows=max_rows)
    targets = read_lines(tgt_path, max_rows=max_rows)
    if len(sources) != len(targets):
        raise ValueError(
            f"Split {split}: source/target mismatch ({len(sources)} vs {len(targets)}) in {input_data_dir}"
        )

    filtered_sources = [
        filter_source_text(row, keep_formula=keep_formula, keep_1h=keep_1h, keep_13c=keep_13c)
        for row in sources
    ]
    write_lines(output_data_dir / f"src-{split}.txt", filtered_sources)

    out_tgt = output_data_dir / f"tgt-{split}.txt"
    if max_rows is None and symlink_passthrough:
        symlink_or_copy(tgt_path, out_tgt, symlink=True)
    else:
        write_lines(out_tgt, targets)

    if ir_path.exists() and keep_ir:
        out_ir = output_data_dir / f"ir-{split}.npy"
        if max_rows is None and symlink_passthrough:
            symlink_or_copy(ir_path, out_ir, symlink=True)
        else:
            full_rows = sum(1 for _ in src_path.open("r", encoding="utf-8"))
            copy_ir_subset(ir_path, out_ir, src_rows=full_rows, keep_rows=len(sources))

    return len(sources), len(filtered_sources)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter tokenized dataset modalities and/or truncate split sizes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Dataset root containing vocab files and data/.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dataset root.")
    parser.add_argument("--drop_formula", action="store_true", help="Remove formula tokens from source text.")
    parser.add_argument("--drop_1h", action="store_true", help="Remove 1H NMR tokens from source text.")
    parser.add_argument("--drop_13c", action="store_true", help="Remove 13C NMR tokens from source text.")
    parser.add_argument("--drop_ir", action="store_true", help="Drop IR arrays from output dataset.")
    parser.add_argument("--max_train", type=int, default=0, help="Optional row cap for train split.")
    parser.add_argument("--max_val", type=int, default=0, help="Optional row cap for val split.")
    parser.add_argument("--max_test", type=int, default=0, help="Optional row cap for test split.")
    parser.add_argument("--copy_vocab", action="store_true", help="Copy vocab files instead of symlinking.")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    input_data_dir = input_root / "data" if (input_root / "data").exists() else input_root
    output_root = Path(args.output_dir)
    output_data_dir = output_root / "data"

    output_data_dir.mkdir(parents=True, exist_ok=True)
    maybe_copy_vocab(input_root if (input_root / "vocab.json").exists() else input_data_dir.parent, output_root, symlink=not args.copy_vocab)

    keep_formula = not args.drop_formula
    keep_1h = not args.drop_1h
    keep_13c = not args.drop_13c
    keep_ir = not args.drop_ir

    limits = {
        "train": args.max_train if args.max_train > 0 else None,
        "val": args.max_val if args.max_val > 0 else None,
        "test": args.max_test if args.max_test > 0 else None,
    }

    for split in ("train", "val", "test"):
        rows_in, rows_out = process_split(
            input_data_dir,
            output_data_dir,
            split,
            keep_formula=keep_formula,
            keep_1h=keep_1h,
            keep_13c=keep_13c,
            keep_ir=keep_ir,
            max_rows=limits[split],
            symlink_passthrough=True,
        )
        print(
            f"[{split}] rows={rows_out} formula={keep_formula} 1H={keep_1h} 13C={keep_13c} IR={keep_ir}"
        )

    print(f"Saved filtered dataset to {output_root}")


if __name__ == "__main__":
    main()
