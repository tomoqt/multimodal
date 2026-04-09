#!/usr/bin/env python3
"""
Evaluate chemistry-aware generation metrics for a trained checkpoint.

Metrics:
- valid_smiles
- exact_match_all
- exact_match_valid_pairs
- avg_tanimoto
- avg_ecfp6_iou
- avg_mcs_over_target
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.diffusion_inference import DiffusionInference
from inference.inference import DecodingStrategy, ModelInference
from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from training.core_dataset import SpectralSmilesDataset
from training.train_autoregressive import load_nmr_tokenizer
from training.train_diffusion import build_diffusion_model, load_diffusion_config


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_cfg(path: Path, model_type: str) -> Dict:
    if model_type == "diffusion":
        return load_diffusion_config(str(path))
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _strip_and_decode_target(tokenizer: SmilesTokenizer, target_tokens: torch.Tensor) -> str:
    ids = target_tokens.tolist()
    try:
        eos_idx = ids.index(tokenizer.sep_token_id)
        ids = ids[:eos_idx]
    except ValueError:
        pass
    return tokenizer.decode(ids[1:]).replace(" ", "").strip()


def _pair_metrics(pred: str, target: str) -> Dict[str, float]:
    out = {
        "valid_pred": False,
        "valid_target": False,
        "exact_match": False,
        "tanimoto": 0.0,
        "ecfp6_iou": 0.0,
        "mcs_over_target": 0.0,
    }
    pred = pred.replace(" ", "")
    target = target.replace(" ", "")

    mol_pred = Chem.MolFromSmiles(pred)
    mol_target = Chem.MolFromSmiles(target)
    out["valid_pred"] = mol_pred is not None
    out["valid_target"] = mol_target is not None
    if not (out["valid_pred"] and out["valid_target"]):
        return out

    can_pred = Chem.MolToSmiles(mol_pred, canonical=True)
    can_target = Chem.MolToSmiles(mol_target, canonical=True)
    out["exact_match"] = can_pred == can_target

    fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, radius=2)
    fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, radius=2)
    out["tanimoto"] = float(DataStructs.TanimotoSimilarity(fp_pred, fp_target))

    fp3_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, radius=3, nBits=1024)
    fp3_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, radius=3, nBits=1024)
    intersection = sum((fp3_pred & fp3_target))
    union = sum((fp3_pred | fp3_target))
    out["ecfp6_iou"] = float(intersection / union) if union > 0 else 0.0

    mcs = rdFMCS.FindMCS([mol_pred, mol_target])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
    if mcs_mol is not None:
        target_atoms = mol_target.GetNumAtoms()
        out["mcs_over_target"] = float(mcs_mol.GetNumAtoms() / target_atoms) if target_atoms > 0 else 0.0

    return out


def _infer_model_type(cli_value: str, cfg: Dict) -> str:
    if cli_value != "auto":
        return cli_value
    if "diffusion" in cfg:
        return "diffusion"
    return "ar"


def _build_ar_model_from_checkpoint(cfg: Dict, checkpoint: Dict, tokenizer: SmilesTokenizer, device: torch.device):
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    smiles_vocab_size = int(state_dict["decoder.smiles_embed.weight"].shape[0])
    nmr_vocab_size = int(state_dict["decoder.nmr_embed.weight"].shape[0])
    model_cfg = cfg["model"]

    model = MultiModalToSMILESModel(
        smiles_vocab_size=smiles_vocab_size,
        nmr_vocab_size=nmr_vocab_size,
        max_seq_length=int(model_cfg["max_seq_length"]),
        max_nmr_length=int(model_cfg["max_nmr_length"]),
        max_memory_length=int(model_cfg["max_memory_length"]),
        embed_dim=int(model_cfg["embed_dim"]),
        num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        verbose=False,
        use_stablemax=bool(model_cfg.get("use_stablemax", False)),
        ir_encoder_type=str(model_cfg.get("ir_encoder_type", "regular")),
        ir_as_prompt=bool(model_cfg.get("ir_as_prompt", False)),
        use_rmsnorm=bool(model_cfg.get("use_rmsnorm", False)),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _build_diffusion_model_from_checkpoint(
    cfg: Dict,
    checkpoint: Dict,
    tokenizer: SmilesTokenizer,
    nmr_tokenizer: Dict[str, int],
    device: torch.device,
):
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model = build_diffusion_model(cfg, tokenizer, nmr_tokenizer).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def evaluate_split(
    model,
    model_type: str,
    tokenizer: SmilesTokenizer,
    nmr_tokenizer: Dict[str, int],
    data_dir: Path,
    split: str,
    max_examples: int,
    max_len: int,
    device: torch.device,
    ir_as_prompt: bool,
    sampling_steps: int,
    sampling_block_length: int,
    sampling_temperature: float,
    sampling_remasking: str,
    cfg_scale: float,
) -> Dict:
    dataset = SpectralSmilesDataset(
        data_dir=str(data_dir),
        smiles_tokenizer=tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split=split,
        max_smiles_len=max_len,
        max_nmr_len=max_len * 2,
        pretokenize=True,
        preload_ir=False,
        use_disk_cache=True,
        write_disk_cache=True,
    )
    total_n = len(dataset) if max_examples <= 0 else min(max_examples, len(dataset))
    ar_inference = None
    diff_inference = None
    if model_type == "ar":
        ar_inference = ModelInference(model, tokenizer, device=device, ir_as_prompt=ir_as_prompt)
    else:
        diff_inference = DiffusionInference(model, tokenizer, device=device, ir_as_prompt=ir_as_prompt)

    valid_pred = 0
    valid_pairs = 0
    exact_all = 0
    exact_valid = 0
    tanimoto_sum = 0.0
    ecfp6_sum = 0.0
    mcs_sum = 0.0

    for idx in range(total_n):
        target_tokens, ir_data, nmr_tokens = dataset[idx]
        if nmr_tokens is not None:
            nmr_tokens = nmr_tokens.to(device)
        if ir_data is not None:
            ir_data = ir_data.to(device)

        if model_type == "ar":
            pred = ar_inference.decode(
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                strategy=DecodingStrategy.GREEDY,
                max_len=max_len,
            )[0]
        else:
            pred = diff_inference.decode(
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                max_len=max_len,
                steps=sampling_steps,
                block_length=sampling_block_length,
                temperature=sampling_temperature,
                remasking=sampling_remasking,
                cfg_scale=cfg_scale,
            )[0]
        target = _strip_and_decode_target(tokenizer, target_tokens)
        m = _pair_metrics(pred, target)

        if m["valid_pred"]:
            valid_pred += 1
        if m["exact_match"]:
            exact_all += 1
        if m["valid_pred"] and m["valid_target"]:
            valid_pairs += 1
            exact_valid += int(m["exact_match"])
            tanimoto_sum += m["tanimoto"]
            ecfp6_sum += m["ecfp6_iou"]
            mcs_sum += m["mcs_over_target"]

        if (idx + 1) % 100 == 0 or (idx + 1) == total_n:
            print(f"[{split}] {idx + 1}/{total_n}", flush=True)

    return {
        "model_type": model_type,
        "split": split,
        "num_examples": int(total_n),
        "valid_smiles": float(valid_pred / max(total_n, 1)),
        "exact_match_all": float(exact_all / max(total_n, 1)),
        "exact_match_valid_pairs": float(exact_valid / max(valid_pairs, 1)),
        "valid_pairs": int(valid_pairs),
        "avg_tanimoto": float(tanimoto_sum / max(valid_pairs, 1)),
        "avg_ecfp6_iou": float(ecfp6_sum / max(valid_pairs, 1)),
        "avg_mcs_over_target": float(mcs_sum / max(valid_pairs, 1)),
        "sampling_steps": int(sampling_steps) if model_type == "diffusion" else None,
        "sampling_block_length": int(sampling_block_length) if model_type == "diffusion" else None,
        "sampling_temperature": float(sampling_temperature) if model_type == "diffusion" else None,
        "sampling_remasking": sampling_remasking if model_type == "diffusion" else None,
        "cfg_scale": float(cfg_scale) if model_type == "diffusion" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chemistry-aware metrics for a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Path to training config yaml")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to tokenized data dir containing src/tgt/ir split files")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_examples", type=int, default=0, help="0 means full split")
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "ar", "diffusion"])
    parser.add_argument("--sampling_steps", type=int, default=0)
    parser.add_argument("--sampling_block_length", type=int, default=0)
    parser.add_argument("--sampling_temperature", type=float, default=0.0)
    parser.add_argument("--sampling_remasking", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--output_json", type=str, default="artifacts/chemical_eval.json")
    args = parser.parse_args()

    prelim_cfg = _load_cfg(Path(args.config), "diffusion" if args.model_type == "diffusion" else "ar")
    model_type = _infer_model_type(args.model_type, prelim_cfg)
    cfg = _load_cfg(Path(args.config), model_type)
    model_cfg = cfg["model"]
    device = _default_device()
    print(f"[setup] device={device} model_type={model_type}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    tokenizer = SmilesTokenizer(vocab_file=str(Path("training/vocab.txt")))
    data_dir = Path(args.data_dir)
    nmr_tokenizer = load_nmr_tokenizer(data_dir)

    if model_type == "ar":
        model = _build_ar_model_from_checkpoint(cfg, checkpoint, tokenizer, device)
    else:
        model = _build_diffusion_model_from_checkpoint(cfg, checkpoint, tokenizer, nmr_tokenizer, device)

    diffusion_cfg = cfg.get("diffusion", {})
    sampling_steps = int(args.sampling_steps) if args.sampling_steps > 0 else int(diffusion_cfg.get("sampling_steps", int(model_cfg["max_seq_length"]) - 1))
    sampling_block_length = (
        int(args.sampling_block_length)
        if args.sampling_block_length > 0
        else int(diffusion_cfg.get("sampling_block_length", int(model_cfg["max_seq_length"]) - 1))
    )
    sampling_remasking = args.sampling_remasking or str(diffusion_cfg.get("sampling_remasking", "low_confidence"))
    cfg_scale = float(args.cfg_scale) if args.cfg_scale is not None else float(diffusion_cfg.get("cfg_scale", 0.0))

    result = evaluate_split(
        model=model,
        model_type=model_type,
        tokenizer=tokenizer,
        nmr_tokenizer=nmr_tokenizer,
        data_dir=data_dir,
        split=args.split,
        max_examples=int(args.max_examples),
        max_len=int(model_cfg["max_seq_length"]),
        device=device,
        ir_as_prompt=bool(model_cfg.get("ir_as_prompt", False)),
        sampling_steps=sampling_steps,
        sampling_block_length=sampling_block_length,
        sampling_temperature=float(args.sampling_temperature),
        sampling_remasking=sampling_remasking,
        cfg_scale=cfg_scale,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
