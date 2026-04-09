#!/usr/bin/env python3
"""
Train the autoregressive model on one fixed batch and measure how quickly it
memorizes that batch under teacher-forced next-token prediction and greedy
decoding.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inference import DecodingStrategy, ModelInference
from models.multimodal_prefix_to_smiles import MultiModalPrefixToSMILESModel
from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from training.core_dataset import SpectralSmilesDataset, collate_spectral_smiles
from training.core_train import compute_next_token_loss, train_step
from training.train_autoregressive import load_config, load_nmr_tokenizer


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
def evaluate_teacher_forced(
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

    logits = model(
        nmr_tokens=nmr_batch,
        ir_data=ir_batch,
        target_seq=target_batch[:, :-1],
        target_padding_mask=target_batch[:, :-1].eq(tokenizer.pad_token_id),
        nmr_padding_mask=nmr_batch.eq(nmr_pad_token_id),
    )
    loss = compute_next_token_loss(logits, target_batch, tokenizer.pad_token_id)
    pred_ids = logits.argmax(dim=-1)
    expected = target_batch[:, 1:]
    valid = expected.ne(tokenizer.pad_token_id)

    token_correct = (pred_ids.eq(expected) & valid).sum().item()
    token_total = valid.sum().item()
    seq_exact = 0
    rows = []
    for idx in range(target_batch.size(0)):
        row_valid = valid[idx]
        if row_valid.any():
            is_exact = bool(torch.equal(pred_ids[idx][row_valid], expected[idx][row_valid]))
        else:
            is_exact = True
        seq_exact += int(is_exact)
        pred = _decode_target(tokenizer, torch.cat([target_batch[idx, :1].cpu(), pred_ids[idx].cpu()], dim=0))
        target = _decode_target(tokenizer, target_batch[idx].cpu())
        rows.append({"idx": idx, "pred": pred, "target": target, "exact": is_exact})

    model.train()
    return {
        "loss": float(loss.item()),
        "token_correct": int(token_correct),
        "token_total": int(token_total),
        "seq_exact": int(seq_exact),
        "count": int(target_batch.size(0)),
        "rows": rows,
    }


@torch.no_grad()
def evaluate_greedy_decode(
    model: torch.nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tokenizer: SmilesTokenizer,
    device: torch.device,
    max_len: int,
) -> Dict[str, object]:
    model.eval()
    infer = ModelInference(model, tokenizer, device=device, ir_as_prompt=False)
    target_batch, ir_batch, nmr_batch = batch

    exact = 0
    rows = []
    for idx in range(target_batch.size(0)):
        pred = infer.decode(
            nmr_tokens=nmr_batch[idx],
            ir_data=None if ir_batch is None else ir_batch[idx],
            strategy=DecodingStrategy.GREEDY,
            max_len=max_len,
        )[0]
        target = _decode_target(tokenizer, target_batch[idx])
        is_exact = pred == target
        exact += int(is_exact)
        rows.append({"idx": idx, "pred": pred, "target": target, "exact": is_exact})

    model.train()
    return {"exact": int(exact), "count": int(target_batch.size(0)), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Overfit one batch with the autoregressive model.")
    parser.add_argument("--config", type=str, required=True, help="AR YAML config path.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu, mps, cuda.")
    parser.add_argument("--batch-size", type=int, default=8, help="Fixed batch size.")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimizer updates.")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluation interval.")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Learning rate for overfit test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model init and training.")
    parser.add_argument(
        "--architecture",
        type=str,
        default="cross_attention",
        choices=["cross_attention", "prefix"],
        help="AR architecture to test.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["architecture"] = args.architecture
    device = torch.device(args.device or str(cfg["training"].get("device", "cpu")))
    torch.manual_seed(args.seed)

    tokenized_dir = Path(cfg["data"]["tokenized_dir"])
    smiles_tokenizer = SmilesTokenizer(vocab_file=str(Path(__file__).resolve().parents[1] / "training" / "vocab.txt"))
    nmr_tokenizer = load_nmr_tokenizer(tokenized_dir)

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

    model_cfg = cfg["model"]
    model_cls = MultiModalToSMILESModel if args.architecture == "cross_attention" else MultiModalPrefixToSMILESModel
    model = model_cls(
        smiles_vocab_size=len(smiles_tokenizer),
        nmr_vocab_size=max(nmr_tokenizer.values()) + 1,
        max_seq_length=int(model_cfg["max_seq_length"]),
        max_nmr_length=int(model_cfg["max_nmr_length"]),
        max_memory_length=int(model_cfg["max_memory_length"]),
        embed_dim=int(model_cfg["embed_dim"]),
        num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        verbose=False,
        ir_encoder_type=str(model_cfg.get("ir_encoder_type", "regular")),
        ir_as_prompt=bool(model_cfg.get("ir_as_prompt", False)),
        use_stablemax=bool(model_cfg.get("use_stablemax", False)),
        use_rmsnorm=bool(model_cfg.get("use_rmsnorm", False)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    results: Dict[str, object] = {
        "config_path": str(Path(args.config).resolve()),
        "device": str(device),
        "batch_size": int(batch[0].size(0)),
        "train_steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "architecture": args.architecture,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "timeline": [],
    }

    initial_teacher = evaluate_teacher_forced(model, batch, smiles_tokenizer, nmr_tokenizer["<PAD>"], device)
    initial_decode = evaluate_greedy_decode(
        model=model,
        batch=batch,
        tokenizer=smiles_tokenizer,
        device=device,
        max_len=int(cfg["model"]["max_seq_length"]),
    )
    results["initial"] = {
        "teacher_forced_loss": initial_teacher["loss"],
        "teacher_forced_seq_exact": initial_teacher["seq_exact"],
        "teacher_forced_token_acc": initial_teacher["token_correct"] / max(initial_teacher["token_total"], 1),
        "greedy_exact": initial_decode["exact"],
        "sample_rows": initial_decode["rows"][:3],
    }
    print(
        f"[ar_overfit] init teacher_seq={initial_teacher['seq_exact']}/{initial_teacher['count']} "
        f"token_acc={initial_teacher['token_correct']}/{initial_teacher['token_total']} "
        f"greedy={initial_decode['exact']}/{initial_decode['count']}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        loss = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            pad_token_id=smiles_tokenizer.pad_token_id,
            device=device,
        )
        if step % args.eval_every != 0:
            continue

        teacher = evaluate_teacher_forced(model, batch, smiles_tokenizer, nmr_tokenizer["<PAD>"], device)
        decode = evaluate_greedy_decode(
            model=model,
            batch=batch,
            tokenizer=smiles_tokenizer,
            device=device,
            max_len=int(cfg["model"]["max_seq_length"]),
        )
        snapshot = {
            "step": int(step),
            "train_step_loss": float(loss),
            "teacher_forced_loss": float(teacher["loss"]),
            "teacher_forced_seq_exact": int(teacher["seq_exact"]),
            "teacher_forced_token_acc": teacher["token_correct"] / max(teacher["token_total"], 1),
            "greedy_exact": int(decode["exact"]),
        }
        results["timeline"].append(snapshot)
        print(
            f"[ar_overfit] step={step} step_loss={loss:.4f} teacher_seq={teacher['seq_exact']}/{teacher['count']} "
            f"token_acc={teacher['token_correct']}/{teacher['token_total']} "
            f"greedy={decode['exact']}/{decode['count']}",
            flush=True,
        )

    final_teacher = evaluate_teacher_forced(model, batch, smiles_tokenizer, nmr_tokenizer["<PAD>"], device)
    final_decode = evaluate_greedy_decode(
        model=model,
        batch=batch,
        tokenizer=smiles_tokenizer,
        device=device,
        max_len=int(cfg["model"]["max_seq_length"]),
    )
    results["final"] = {
        "teacher_forced_loss": final_teacher["loss"],
        "teacher_forced_seq_exact": final_teacher["seq_exact"],
        "teacher_forced_token_acc": final_teacher["token_correct"] / max(final_teacher["token_total"], 1),
        "teacher_forced_rows": final_teacher["rows"],
        "greedy_exact": final_decode["exact"],
        "greedy_rows": final_decode["rows"],
    }
    print(
        f"[ar_overfit] final teacher_seq={final_teacher['seq_exact']}/{final_teacher['count']} "
        f"token_acc={final_teacher['token_correct']}/{final_teacher['token_total']} "
        f"greedy={final_decode['exact']}/{final_decode['count']}",
        flush=True,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[ar_overfit] wrote {output_path}", flush=True)
    else:
        print(json.dumps(results, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
