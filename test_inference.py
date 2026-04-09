#!/usr/bin/env python3
"""
Lean inference/evaluation script for the core architecture.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from inference.inference import DecodingStrategy, ModelInference
from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from training.core_dataset import SpectralSmilesDataset, collate_spectral_smiles


def load_config(config_path: str = None) -> Dict:
    config = {
        "model": {
            "max_seq_length": 128,
            "max_nmr_length": 256,
            "max_memory_length": 128,
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "dropout": 0.1,
            "ir_encoder_type": "regular",
            "ir_as_prompt": False,
            "use_stablemax": False,
            "use_rmsnorm": False,
        },
        "data": {"tokenized_dir": "data/tokenized_baseline/data"},
    }
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        for section, section_val in user_cfg.items():
            if isinstance(section_val, dict) and section in config:
                config[section].update(section_val)
            else:
                config[section] = section_val
    return config


def load_nmr_tokenizer(tokenized_dir: Path) -> Dict[str, int]:
    vocab_json = tokenized_dir.parent / "vocab.json"
    if not vocab_json.exists():
        raise FileNotFoundError(f"Missing NMR vocabulary: {vocab_json}")
    with vocab_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_strategy(name: str) -> DecodingStrategy:
    mapping = {
        "greedy": DecodingStrategy.GREEDY,
        "beam": DecodingStrategy.BEAM,
        "sampling": DecodingStrategy.SAMPLING,
        "nucleus": DecodingStrategy.NUCLEUS,
    }
    if name not in mapping:
        raise ValueError(f"Unknown strategy '{name}'. Valid values: {', '.join(mapping)}")
    return mapping[name]


def evaluate(
    inference: ModelInference,
    dataset: SpectralSmilesDataset,
    strategies: List[str],
    max_examples: int,
    max_len: int,
    beam_width: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Dict[str, Dict[str, float]]:
    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    results: Dict[str, Dict[str, float]] = {}

    for strat_name in strategies:
        strategy = decode_strategy(strat_name)
        exact = 0
        for idx in range(n):
            target_tokens, ir_data, nmr_tokens = dataset[idx]
            if ir_data is not None:
                ir_data = ir_data.to(inference.device)
            if nmr_tokens is not None:
                nmr_tokens = nmr_tokens.to(inference.device)

            decoded = inference.decode(
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                strategy=strategy,
                max_len=max_len,
                beam_width=beam_width,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )[0]

            tgt = target_tokens.tolist()
            try:
                eos_idx = tgt.index(inference.eos_token_id)
                tgt = tgt[:eos_idx]
            except ValueError:
                pass
            target_smiles = inference.tokenizer.decode(tgt[1:]).replace(" ", "").strip()
            exact += int(decoded == target_smiles)

        results[strat_name] = {
            "num_examples": float(n),
            "exact_match_rate": exact / max(n, 1),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lean inference evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--strategies", type=str, default="greedy,beam,sampling,nucleus")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_csv", type=str, default="inference_results/core_eval.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenized_dir = Path(config["data"]["tokenized_dir"])
    nmr_tokenizer = load_nmr_tokenizer(tokenized_dir)
    smiles_tokenizer = SmilesTokenizer(vocab_file=str(Path("training/vocab.txt")))

    model = MultiModalToSMILESModel(
        smiles_vocab_size=len(smiles_tokenizer),
        nmr_vocab_size=max(nmr_tokenizer.values()) + 1,
        max_seq_length=config["model"]["max_seq_length"],
        max_nmr_length=config["model"]["max_nmr_length"],
        max_memory_length=config["model"]["max_memory_length"],
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        verbose=False,
        use_stablemax=config["model"].get("use_stablemax", False),
        ir_encoder_type=config["model"].get("ir_encoder_type", "regular"),
        ir_as_prompt=config["model"].get("ir_as_prompt", False),
        use_rmsnorm=config["model"].get("use_rmsnorm", False),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    dataset = SpectralSmilesDataset(
        data_dir=str(tokenized_dir),
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split=args.split,
        max_smiles_len=config["model"]["max_seq_length"],
        max_nmr_len=config["model"]["max_nmr_length"],
    )
    inference = ModelInference(model, smiles_tokenizer, device=device, ir_as_prompt=config["model"].get("ir_as_prompt", False))

    strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
    metrics = evaluate(
        inference=inference,
        dataset=dataset,
        strategies=strategies,
        max_examples=args.max_examples,
        max_len=args.max_len,
        beam_width=args.beam_width,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "num_examples", "exact_match_rate"])
        for strategy, vals in metrics.items():
            writer.writerow([strategy, int(vals["num_examples"]), f"{vals['exact_match_rate']:.6f}"])

    for strategy, vals in metrics.items():
        print(f"{strategy}: exact_match_rate={vals['exact_match_rate']:.4f} on n={int(vals['num_examples'])}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
