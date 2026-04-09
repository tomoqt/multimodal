#!/usr/bin/env python3
"""
Build per-example per-position criticality scores from a pretrained AR model.

These scores can then bias the diffusion masking scheduler toward positions that
the AR model finds uncertain or brittle under free-running decode.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inference import DecodingStrategy, ModelInference
from models.smiles_tokenizer import SmilesTokenizer
from scripts.eval_chemical_metrics import _build_ar_model_from_checkpoint
from training.core_dataset import SpectralSmilesDataset
from training.train_autoregressive import load_config, load_nmr_tokenizer


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pad_token_ids(token_ids: List[int], pad_token_id: int, max_len: int) -> torch.Tensor:
    padded = token_ids[:max_len] + [pad_token_id] * max(0, max_len - len(token_ids))
    return torch.tensor(padded[:max_len], dtype=torch.long)


def _greedy_prediction_tokens(
    inference: ModelInference,
    tokenizer: SmilesTokenizer,
    nmr_tokens: torch.Tensor,
    ir_data,
    max_len: int,
) -> torch.Tensor:
    pred = inference.decode(
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        strategy=DecodingStrategy.GREEDY,
        max_len=max_len,
    )[0]
    pred_ids = tokenizer.encode(pred, add_special_tokens=True, max_length=max_len, truncation=True)
    return _pad_token_ids(pred_ids, tokenizer.pad_token_id, max_len)


def _normalize_valid(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(values, dtype=torch.float32)
    if not bool(valid_mask.any()):
        return out
    valid_vals = values[valid_mask].to(dtype=torch.float32)
    vmin = valid_vals.min()
    vmax = valid_vals.max()
    if float((vmax - vmin).abs().item()) < 1e-8:
        out[valid_mask] = 0.0
    else:
        out[valid_mask] = (valid_vals - vmin) / (vmax - vmin)
    return out


@torch.no_grad()
def _build_scores_for_example(
    model,
    inference: ModelInference,
    tokenizer: SmilesTokenizer,
    target_tokens: torch.Tensor,
    nmr_tokens: torch.Tensor,
    ir_data,
    nmr_pad_token_id: int,
    device: torch.device,
    mismatch_radius: int,
    teacher_nll_weight: float,
    entropy_weight: float,
    teacher_error_weight: float,
    mismatch_weight: float,
) -> Dict[str, torch.Tensor]:
    target_tokens = target_tokens.to(device).unsqueeze(0)
    nmr_tokens = nmr_tokens.to(device).unsqueeze(0)
    if ir_data is not None:
        ir_data = ir_data.to(device).unsqueeze(0)

    logits = model(
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        target_seq=target_tokens[:, :-1],
        nmr_padding_mask=nmr_tokens.eq(nmr_pad_token_id),
    )
    next_targets = target_tokens[:, 1:]
    valid_next = next_targets.ne(tokenizer.pad_token_id)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    gold_log_probs = torch.gather(log_probs, dim=-1, index=next_targets.unsqueeze(-1)).squeeze(-1)
    nll = (-gold_log_probs).squeeze(0)
    entropy = (-(probs * log_probs).sum(dim=-1)).squeeze(0)
    teacher_pred = torch.argmax(logits, dim=-1)
    teacher_error = (teacher_pred != next_targets).squeeze(0) & valid_next.squeeze(0)

    pred_tokens = _greedy_prediction_tokens(
        inference=inference,
        tokenizer=tokenizer,
        nmr_tokens=nmr_tokens.squeeze(0),
        ir_data=None if ir_data is None else ir_data.squeeze(0),
        max_len=target_tokens.size(1),
    ).to(device)

    valid_target = target_tokens.squeeze(0).ne(tokenizer.pad_token_id) & target_tokens.squeeze(0).ne(tokenizer.cls_token_id)
    mismatch = (pred_tokens != target_tokens.squeeze(0)) & valid_target
    mismatch_neighborhood = torch.zeros_like(valid_target, dtype=torch.float32)
    if bool(mismatch.any()):
        first_mismatch = int(torch.nonzero(mismatch, as_tuple=False)[0].item())
        start = max(1, first_mismatch - mismatch_radius)
        end = min(target_tokens.size(1), first_mismatch + mismatch_radius + 1)
        mismatch_neighborhood[start:end] = 1.0

    score_row = torch.zeros(target_tokens.size(1), dtype=torch.float32, device=device)
    valid_positions = valid_target
    if target_tokens.size(1) > 1:
        score_row[1:] = (
            teacher_nll_weight * _normalize_valid(nll, valid_next.squeeze(0))
            + entropy_weight * _normalize_valid(entropy, valid_next.squeeze(0))
            + teacher_error_weight * teacher_error.to(dtype=torch.float32)
        )
    score_row = score_row + mismatch_weight * mismatch_neighborhood
    score_row = _normalize_valid(score_row, valid_positions)

    return {
        "scores": score_row.cpu(),
        "teacher_error_rate": teacher_error.to(dtype=torch.float32).mean().cpu(),
        "has_greedy_mismatch": torch.tensor(float(bool(mismatch.any())), dtype=torch.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AR-derived criticality cache for diffusion masking.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to AR checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Path to AR config yaml")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to tokenized dataset dir")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_examples", type=int, default=0, help="0 means full split")
    parser.add_argument("--device", type=str, default=None, help="cpu, mps, or cuda")
    parser.add_argument("--mismatch_radius", type=int, default=2)
    parser.add_argument("--teacher_nll_weight", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.35)
    parser.add_argument("--teacher_error_weight", type=float, default=0.5)
    parser.add_argument("--mismatch_weight", type=float, default=0.8)
    parser.add_argument("--log_every", type=int, default=10, help="Progress print cadence in examples.")
    parser.add_argument("--output", type=str, required=True, help="Output .pt path")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else _default_device()
    cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    tokenizer = SmilesTokenizer(vocab_file=str(Path("training/vocab.txt")))
    data_dir = Path(args.data_dir)
    nmr_tokenizer = load_nmr_tokenizer(data_dir)
    model = _build_ar_model_from_checkpoint(cfg, checkpoint, tokenizer, device)
    inference = ModelInference(model, tokenizer, device=device, ir_as_prompt=bool(cfg["model"].get("ir_as_prompt", False)))

    dataset = SpectralSmilesDataset(
        data_dir=str(data_dir),
        smiles_tokenizer=tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split=args.split,
        max_smiles_len=int(cfg["model"]["max_seq_length"]),
        max_nmr_len=int(cfg["model"]["max_nmr_length"]),
        pretokenize=True,
        preload_ir=False,
        use_disk_cache=True,
        write_disk_cache=True,
    )
    total_n = len(dataset) if args.max_examples <= 0 else min(int(args.max_examples), len(dataset))
    max_len = int(cfg["model"]["max_seq_length"])
    scores = torch.zeros(total_n, max_len, dtype=torch.float16)
    teacher_error_rates: List[float] = []
    greedy_mismatch_flags: List[float] = []

    nmr_pad_token_id = int(nmr_tokenizer.get("<PAD>", 0))
    for idx in range(total_n):
        row = dataset[idx]
        if len(row) == 4:
            target_tokens, ir_data, nmr_tokens, _ = row
        else:
            target_tokens, ir_data, nmr_tokens = row
        metrics = _build_scores_for_example(
            model=model,
            inference=inference,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            nmr_pad_token_id=nmr_pad_token_id,
            device=device,
            mismatch_radius=int(args.mismatch_radius),
            teacher_nll_weight=float(args.teacher_nll_weight),
            entropy_weight=float(args.entropy_weight),
            teacher_error_weight=float(args.teacher_error_weight),
            mismatch_weight=float(args.mismatch_weight),
        )
        scores[idx] = metrics["scores"].to(dtype=torch.float16)
        teacher_error_rates.append(float(metrics["teacher_error_rate"].item()))
        greedy_mismatch_flags.append(float(metrics["has_greedy_mismatch"].item()))
        if (idx + 1) % max(1, int(args.log_every)) == 0 or (idx + 1) == total_n:
            print(f"[cache] {idx + 1}/{total_n}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scores": scores,
        "metadata": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config": str(Path(args.config).resolve()),
            "data_dir": str(data_dir.resolve()),
            "split": args.split,
            "num_examples": int(total_n),
            "max_seq_length": int(max_len),
            "mismatch_radius": int(args.mismatch_radius),
            "teacher_nll_weight": float(args.teacher_nll_weight),
            "entropy_weight": float(args.entropy_weight),
            "teacher_error_weight": float(args.teacher_error_weight),
            "mismatch_weight": float(args.mismatch_weight),
            "mean_teacher_error_rate": float(sum(teacher_error_rates) / max(len(teacher_error_rates), 1)),
            "mean_has_greedy_mismatch": float(sum(greedy_mismatch_flags) / max(len(greedy_mismatch_flags), 1)),
        },
    }
    torch.save(payload, output_path)
    print(json.dumps(payload["metadata"], indent=2))
    print(f"[done] wrote {output_path}")


if __name__ == "__main__":
    main()
