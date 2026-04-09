#!/usr/bin/env python3
"""
Train a diffusion model on one fixed batch and measure whether it can memorize
that batch under both one-shot reconstruction and iterative decoding.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.diffusion_inference import DiffusionInference
from models.smiles_tokenizer import SmilesTokenizer
from training.core_dataset import SpectralSmilesDataset, collate_spectral_smiles
from training.diffusion_core import (
    block_diffusion_train_step,
    build_smiles_priority_token_ids,
    iter_generated_blocks,
    prepare_block_diffusion_targets,
    diffusion_train_step,
)
from training.diffusion_scheduler import load_ar_criticality_cache, resolve_position_scores, unpack_batch
from training.train_autoregressive import load_nmr_tokenizer
from training.train_diffusion import build_diffusion_model, load_diffusion_config


def _decode_target(tokenizer: SmilesTokenizer, token_row: torch.Tensor) -> str:
    ids = token_row.tolist()
    out: List[int] = []
    for token_id in ids:
        if token_id == tokenizer.sep_token_id:
            break
        if token_id in (tokenizer.cls_token_id, tokenizer.pad_token_id):
            continue
        out.append(token_id)
    return tokenizer.decode(out).replace(" ", "").strip()


@torch.no_grad()
def evaluate_one_shot(
    model: torch.nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tokenizer: SmilesTokenizer,
    nmr_pad_token_id: int,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    target_batch, ir_batch, nmr_batch = batch
    target_batch = target_batch.to(device)
    nmr_batch = nmr_batch.to(device)
    if ir_batch is not None:
        ir_batch = ir_batch.to(device)

    if getattr(model, "supports_block_diffusion", False):
        pred_ids = target_batch.clone()
        valid = target_batch.ne(tokenizer.pad_token_id) & target_batch.ne(tokenizer.cls_token_id)
        token_correct = 0.0
        token_total = 0.0
        block_size = max(1, int(getattr(model, "block_size", 8)))
        for block_start, block_end in iter_generated_blocks(target_batch.size(1), block_size):
            noisy_tokens, _, _, valid_positions = prepare_block_diffusion_targets(
                target_tokens=target_batch,
                block_start=block_start,
                block_end=block_end,
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.cls_token_id,
                mask_prob_floor=1.0,
                mask_prob_ceiling=1.0,
            )
            if not valid_positions.any():
                continue
            logits = model(
                nmr_tokens=nmr_batch,
                ir_data=ir_batch,
                target_seq=noisy_tokens,
                target_padding_mask=noisy_tokens.eq(tokenizer.pad_token_id),
                nmr_padding_mask=nmr_batch.eq(nmr_pad_token_id),
                block_start=block_start,
                block_end=block_end,
            )
            block_pred = logits[:, block_start:block_end, :].argmax(dim=-1)
            block_valid = valid_positions[:, block_start:block_end]
            token_correct += float((block_pred[block_valid] == target_batch[:, block_start:block_end][block_valid]).sum().item())
            token_total += float(block_valid.sum().item())
            pred_ids[:, block_start:block_end] = torch.where(
                block_valid,
                block_pred,
                pred_ids[:, block_start:block_end],
            )
        token_acc = token_correct / max(token_total, 1.0)
    else:
        masked = target_batch.clone()
        valid = masked.ne(tokenizer.pad_token_id) & masked.ne(tokenizer.cls_token_id)
        masked[valid] = tokenizer.mask_token_id
        logits = model(
            nmr_tokens=nmr_batch,
            ir_data=ir_batch,
            target_seq=masked,
            target_padding_mask=target_batch.eq(tokenizer.pad_token_id),
            nmr_padding_mask=nmr_batch.eq(nmr_pad_token_id),
        )
        pred_ids = logits.argmax(dim=-1)
        token_acc = float((pred_ids[valid] == target_batch[valid]).float().mean().item()) if valid.any() else 0.0
        pred_ids = torch.where(valid, pred_ids, target_batch)

    rows = []
    exact = 0
    for idx in range(target_batch.size(0)):
        pred = _decode_target(tokenizer, pred_ids[idx].cpu())
        target = _decode_target(tokenizer, target_batch[idx].cpu())
        is_exact = pred == target
        exact += int(is_exact)
        rows.append({"idx": idx, "pred": pred, "target": target, "exact": is_exact})

    model.train()
    return {"exact": exact, "count": int(target_batch.size(0)), "token_acc": token_acc, "rows": rows}


@torch.no_grad()
def evaluate_iterative(
    model: torch.nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tokenizer: SmilesTokenizer,
    device: torch.device,
    max_len: int,
    remasking: str,
    sampling_steps: int,
    sampling_block_length: int,
    cfg_scale: float,
) -> Dict[str, object]:
    model.eval()
    infer = DiffusionInference(model, tokenizer, device=device)
    target_batch, ir_batch, nmr_batch = batch

    rows = []
    exact = 0
    for idx in range(target_batch.size(0)):
        pred = infer.decode(
            nmr_tokens=nmr_batch[idx],
            ir_data=None if ir_batch is None else ir_batch[idx],
            max_len=max_len,
            steps=sampling_steps,
            block_length=sampling_block_length,
            temperature=0.0,
            remasking=remasking,
            cfg_scale=cfg_scale,
        )[0]
        target = _decode_target(tokenizer, target_batch[idx])
        is_exact = pred == target
        exact += int(is_exact)
        rows.append({"idx": idx, "pred": pred, "target": target, "exact": is_exact})

    model.train()
    return {"exact": exact, "count": int(target_batch.size(0)), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Overfit one batch with the diffusion model and compare remasking.")
    parser.add_argument("--config", type=str, required=True, help="Diffusion YAML config path.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu, mps, cuda.")
    parser.add_argument("--batch-size", type=int, default=8, help="Fixed batch size for the overfit batch.")
    parser.add_argument("--steps", type=int, default=300, help="Number of optimizer updates on the same batch.")
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluation interval.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Learning rate for overfit test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model init and training noise.")
    parser.add_argument(
        "--condition-drop-prob",
        type=float,
        default=None,
        help="Optional override for diffusion condition dropout probability.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Optional guidance scale used only for iterative decode evaluation.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=0,
        help="Optional override for diffusion sampling steps. Uses config value when <= 0.",
    )
    parser.add_argument(
        "--sampling-block-length",
        type=int,
        default=0,
        help="Optional override for diffusion sampling block length. Uses config value when <= 0.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="cross_attention",
        choices=["cross_attention", "block_diffusion", "prefix_encoder", "unified_encoder"],
        help="Diffusion architecture to test.",
    )
    parser.add_argument(
        "--mask-schedule",
        type=str,
        default=None,
        choices=["uniform", "token_priority", "ar_criticality"],
        help="Optional diffusion masking scheduler override.",
    )
    parser.add_argument(
        "--priority-weight",
        type=float,
        default=None,
        help="Optional scheduler emphasis weight for token_priority or ar_criticality.",
    )
    parser.add_argument(
        "--ar-criticality-cache",
        type=str,
        default=None,
        help="Optional AR criticality cache path used when mask schedule is ar_criticality.",
    )
    parser.add_argument(
        "--remasking",
        type=str,
        nargs="+",
        default=["low_confidence", "random"],
        help="Iterative remasking strategies to compare.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    cfg = load_diffusion_config(args.config)
    cfg["diffusion"]["architecture"] = args.architecture
    if args.mask_schedule is not None:
        cfg["diffusion"]["mask_schedule"] = str(args.mask_schedule)
    if args.priority_weight is not None:
        cfg["diffusion"]["priority_weight"] = float(args.priority_weight)
    if args.ar_criticality_cache is not None:
        cfg["diffusion"]["ar_criticality_cache_path"] = str(Path(args.ar_criticality_cache).resolve())
    if args.condition_drop_prob is not None:
        cfg["diffusion"]["condition_dropout_prob"] = float(args.condition_drop_prob)
    if args.cfg_scale is not None:
        cfg["diffusion"]["cfg_scale"] = float(args.cfg_scale)
    if args.sampling_steps > 0:
        cfg["diffusion"]["sampling_steps"] = int(args.sampling_steps)
    if args.sampling_block_length > 0:
        cfg["diffusion"]["sampling_block_length"] = int(args.sampling_block_length)
    if args.architecture == "block_diffusion":
        cfg["diffusion"]["block_size"] = int(cfg["diffusion"].get("sampling_block_length", 8))
    device = torch.device(args.device or str(cfg["training"].get("device", "cpu")))
    torch.manual_seed(args.seed)

    tokenized_dir = Path(cfg["data"]["tokenized_dir"])
    smiles_tokenizer = SmilesTokenizer(vocab_file=str(Path(__file__).resolve().parents[1] / "training" / "vocab.txt"))
    nmr_tokenizer = load_nmr_tokenizer(tokenized_dir)
    mask_schedule = str(cfg["diffusion"].get("mask_schedule", "uniform")).lower()
    priority_token_ids = build_smiles_priority_token_ids(smiles_tokenizer) if mask_schedule == "token_priority" else None
    ar_criticality_cache = (
        load_ar_criticality_cache(str(cfg["diffusion"]["ar_criticality_cache_path"]))
        if mask_schedule == "ar_criticality"
        else None
    )

    dataset = SpectralSmilesDataset(
        data_dir=str(tokenized_dir),
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split="train",
        max_smiles_len=int(cfg["model"]["max_seq_length"]),
        max_nmr_len=int(cfg["model"]["max_nmr_length"]),
        pretokenize=bool(cfg["data"].get("pretokenize", True)),
        preload_ir=bool(cfg["data"].get("preload_ir", False)),
        use_disk_cache=bool(cfg["data"].get("use_disk_cache", True)),
        write_disk_cache=bool(cfg["data"].get("write_disk_cache", True)),
        cache_dir=cfg["data"].get("cache_dir"),
        ir_cache_mode=str(cfg["data"].get("ir_cache_mode", "none")),
        ir_cache_dtype=str(cfg["data"].get("ir_cache_dtype", "float32")),
        return_index=mask_schedule == "ar_criticality",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_spectral_smiles(
            b,
            smiles_pad_token_id=smiles_tokenizer.pad_token_id,
            nmr_pad_token_id=nmr_tokenizer["<PAD>"],
        ),
    )
    batch = next(iter(loader))
    target_batch, ir_batch, nmr_batch, example_indices = unpack_batch(batch)
    batch_eval = (target_batch, ir_batch, nmr_batch)
    batch_position_scores = resolve_position_scores(
        target_tokens=target_batch,
        mask_schedule=mask_schedule,
        priority_token_ids=priority_token_ids,
        ar_criticality_cache=ar_criticality_cache,
        example_indices=example_indices,
    )

    model = build_diffusion_model(cfg, smiles_tokenizer, nmr_tokenizer).to(device)
    if args.architecture == "block_diffusion":
        model.block_size = int(cfg["diffusion"].get("block_size", 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    results: Dict[str, object] = {
        "config_path": str(Path(args.config).resolve()),
        "device": str(device),
        "architecture": args.architecture,
        "batch_size": int(batch[0].size(0)),
        "train_steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "mask_schedule": mask_schedule,
        "priority_weight": float(cfg["diffusion"].get("priority_weight", 2.0)),
        "ar_criticality_cache_path": cfg["diffusion"].get("ar_criticality_cache_path"),
        "sampling_steps": int(cfg["diffusion"]["sampling_steps"]),
        "sampling_block_length": int(cfg["diffusion"]["sampling_block_length"]),
        "condition_dropout_prob": float(cfg["diffusion"].get("condition_dropout_prob", 0.0)),
        "cfg_scale": float(cfg["diffusion"].get("cfg_scale", 0.0)),
        "remasking": list(args.remasking),
        "timeline": [],
    }

    initial_one_shot = evaluate_one_shot(model, batch_eval, smiles_tokenizer, nmr_tokenizer["<PAD>"], device)
    initial_iterative = {
        remasking: evaluate_iterative(
            model=model,
            batch=batch_eval,
            tokenizer=smiles_tokenizer,
            device=device,
            max_len=int(cfg["model"]["max_seq_length"]),
            remasking=remasking,
            sampling_steps=int(cfg["diffusion"]["sampling_steps"]),
            sampling_block_length=int(cfg["diffusion"]["sampling_block_length"]),
            cfg_scale=float(cfg["diffusion"].get("cfg_scale", 0.0)),
        )
        for remasking in args.remasking
    }
    results["initial"] = {
        "one_shot_exact": initial_one_shot["exact"],
        "one_shot_token_acc": initial_one_shot["token_acc"],
        "iterative_exact": {k: v["exact"] for k, v in initial_iterative.items()},
        "sample_rows": {k: v["rows"][:3] for k, v in initial_iterative.items()},
    }
    print(
        f"[overfit] init one_shot={initial_one_shot['exact']}/{initial_one_shot['count']} "
        f"one_shot_token_acc={initial_one_shot['token_acc']:.4f} "
        + " ".join(
            f"{name}={payload['exact']}/{payload['count']}" for name, payload in initial_iterative.items()
        ),
        flush=True,
    )

    for step in range(1, args.steps + 1):
        if args.architecture == "block_diffusion":
            loss = block_diffusion_train_step(
                model=model,
                optimizer=optimizer,
                batch=batch_eval,
                mask_token_id=smiles_tokenizer.mask_token_id,
                pad_token_id=smiles_tokenizer.pad_token_id,
                bos_token_id=smiles_tokenizer.cls_token_id,
                device=device,
                block_size=int(cfg["diffusion"].get("block_size", 8)),
                mask_prob_floor=float(cfg["diffusion"]["mask_prob_floor"]),
                mask_prob_ceiling=float(cfg["diffusion"].get("mask_prob_ceiling", 1.0)),
                nmr_pad_token_id=nmr_tokenizer["<PAD>"],
                condition_dropout_prob=float(cfg["diffusion"].get("condition_dropout_prob", 0.0)),
                position_scores=batch_position_scores,
                priority_weight=float(cfg["diffusion"].get("priority_weight", 2.0)),
            )
        else:
            loss = diffusion_train_step(
                model=model,
                optimizer=optimizer,
                batch=batch_eval,
                mask_token_id=smiles_tokenizer.mask_token_id,
                pad_token_id=smiles_tokenizer.pad_token_id,
                bos_token_id=smiles_tokenizer.cls_token_id,
                device=device,
                mask_prob_floor=float(cfg["diffusion"]["mask_prob_floor"]),
                mask_prob_ceiling=float(cfg["diffusion"].get("mask_prob_ceiling", 1.0)),
                nmr_pad_token_id=nmr_tokenizer["<PAD>"],
                condition_dropout_prob=float(cfg["diffusion"].get("condition_dropout_prob", 0.0)),
                position_scores=batch_position_scores,
                priority_weight=float(cfg["diffusion"].get("priority_weight", 2.0)),
                complementary_masking=bool(cfg["diffusion"].get("complementary_masking", False)),
            )
        if step % args.eval_every != 0:
            continue

        one_shot = evaluate_one_shot(model, batch_eval, smiles_tokenizer, nmr_tokenizer["<PAD>"], device)
        iterative = {
            remasking: evaluate_iterative(
                model=model,
                batch=batch_eval,
                tokenizer=smiles_tokenizer,
                device=device,
                max_len=int(cfg["model"]["max_seq_length"]),
                remasking=remasking,
                sampling_steps=int(cfg["diffusion"]["sampling_steps"]),
                sampling_block_length=int(cfg["diffusion"]["sampling_block_length"]),
                cfg_scale=float(cfg["diffusion"].get("cfg_scale", 0.0)),
            )
            for remasking in args.remasking
        }
        snapshot = {
            "step": int(step),
            "loss": float(loss),
            "one_shot_exact": int(one_shot["exact"]),
            "one_shot_token_acc": float(one_shot["token_acc"]),
            "iterative_exact": {k: int(v["exact"]) for k, v in iterative.items()},
        }
        results["timeline"].append(snapshot)
        print(
            f"[overfit] step={step} loss={loss:.4f} one_shot={one_shot['exact']}/{one_shot['count']} "
            f"one_shot_token_acc={one_shot['token_acc']:.4f} "
            + " ".join(f"{name}={payload['exact']}/{payload['count']}" for name, payload in iterative.items()),
            flush=True,
        )

    final_one_shot = evaluate_one_shot(model, batch_eval, smiles_tokenizer, nmr_tokenizer["<PAD>"], device)
    final_iterative = {
        remasking: evaluate_iterative(
            model=model,
            batch=batch_eval,
            tokenizer=smiles_tokenizer,
            device=device,
            max_len=int(cfg["model"]["max_seq_length"]),
            remasking=remasking,
            sampling_steps=int(cfg["diffusion"]["sampling_steps"]),
            sampling_block_length=int(cfg["diffusion"]["sampling_block_length"]),
            cfg_scale=float(cfg["diffusion"].get("cfg_scale", 0.0)),
        )
        for remasking in args.remasking
    }
    results["final"] = {
        "one_shot_exact": final_one_shot["exact"],
        "one_shot_token_acc": final_one_shot["token_acc"],
        "one_shot_rows": final_one_shot["rows"],
        "iterative_exact": {k: v["exact"] for k, v in final_iterative.items()},
        "iterative_rows": {k: v["rows"] for k, v in final_iterative.items()},
    }
    print(
        f"[overfit] final one_shot={final_one_shot['exact']}/{final_one_shot['count']} "
        f"one_shot_token_acc={final_one_shot['token_acc']:.4f} "
        + " ".join(f"{name}={payload['exact']}/{payload['count']}" for name, payload in final_iterative.items()),
        flush=True,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[overfit] wrote {output_path}", flush=True)
    else:
        print(json.dumps(results, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
