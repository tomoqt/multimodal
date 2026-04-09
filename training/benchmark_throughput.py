#!/usr/bin/env python3
"""
Measure steady-state training throughput (non-pad target tokens/sec).

Supports single process and DDP launch via torchrun.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.smiles_tokenizer import SmilesTokenizer
from training.core_train import compute_next_token_loss
from training.diffusion_core import apply_condition_dropout, compute_masked_diffusion_loss, sample_noisy_targets
from training.train_autoregressive import (
    DistContext,
    _autocast_context,
    _cleanup_dist,
    _infer_dist_context,
    _reduce_pair,
    _seed_everything,
    _set_perf_flags,
    _build_model,
    create_loaders,
    load_config,
    load_nmr_tokenizer,
)
from training.train_diffusion import build_diffusion_model, load_diffusion_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Throughput micro-benchmark for multimodal pretraining")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--mode", type=str, choices=["ar", "diffusion"], default="ar")
    parser.add_argument("--batch-size", type=int, required=True, help="Per-rank train batch size override")
    parser.add_argument("--steps", type=int, default=80, help="Measured optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps before timing")
    parser.add_argument("--device", type=str, default=None, help="Override config training.device")
    parser.add_argument("--num-workers", type=int, default=None, help="Override config training.num_workers")
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile even if enabled in config",
    )
    return parser.parse_args()


def _max_reduce(value: float, device: torch.device, dist_ctx: DistContext) -> float:
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    if dist_ctx.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _maybe_enable_compile(model: torch.nn.Module, training_cfg: Dict, disable_compile: bool) -> torch.nn.Module:
    compile_enabled = bool(training_cfg.get("compile", False)) and not disable_compile
    if not compile_enabled:
        return model
    if not hasattr(torch, "compile"):
        return model
    return torch.compile(
        model,
        mode=str(training_cfg.get("compile_mode", "max-autotune")),
        dynamic=bool(training_cfg.get("compile_dynamic", False)),
        fullgraph=bool(training_cfg.get("compile_fullgraph", False)),
    )


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config) if args.mode == "ar" else load_diffusion_config(args.config)
    training_cfg = cfg["training"]

    training_cfg["batch_size"] = int(args.batch_size)
    training_cfg["test_batch_size"] = int(args.batch_size)
    training_cfg["num_epochs"] = 1
    training_cfg["log_every_steps"] = 0
    training_cfg["drop_last"] = True

    if args.device is not None:
        training_cfg["device"] = str(args.device).lower()
    if args.num_workers is not None:
        training_cfg["num_workers"] = int(args.num_workers)

    requested_device_name = str(training_cfg.get("device", "cpu")).lower()
    dist_ctx = DistContext(enabled=False)

    try:
        dist_ctx = _infer_dist_context(training_cfg, requested_device_name)

        if requested_device_name == "cuda":
            cuda_index = dist_ctx.local_rank if dist_ctx.enabled else 0
            device = torch.device(f"cuda:{cuda_index}")
        elif requested_device_name == "cpu":
            device = torch.device("cpu")
        elif requested_device_name == "mps":
            if dist_ctx.enabled:
                raise RuntimeError("MPS DDP is unsupported for this benchmark. Use CUDA + NCCL for multi-GPU.")
            device = torch.device("mps")
        else:
            raise ValueError(f"Unsupported device '{requested_device_name}'.")

        _set_perf_flags(training_cfg)
        base_seed = int(training_cfg.get("seed", 1337))
        _seed_everything(base_seed + dist_ctx.rank)

        tokenized_dir = Path(cfg["data"]["tokenized_dir"])
        if not tokenized_dir.exists():
            raise FileNotFoundError(f"Tokenized directory not found: {tokenized_dir}")

        smiles_tokenizer = SmilesTokenizer(vocab_file=str(Path(__file__).with_name("vocab.txt")))
        nmr_tokenizer = load_nmr_tokenizer(tokenized_dir)
        nmr_pad_token_id = int(nmr_tokenizer.get("<PAD>", 0))

        train_loader, _, _ = create_loaders(
            tokenized_dir,
            smiles_tokenizer,
            nmr_tokenizer,
            cfg,
            dist_ctx=dist_ctx,
            device=device,
        )

        if args.mode == "ar":
            model = _build_model(cfg, smiles_tokenizer, nmr_tokenizer, device)
        else:
            model = build_diffusion_model(cfg, smiles_tokenizer, nmr_tokenizer).to(device)
        model = _maybe_enable_compile(model, training_cfg, disable_compile=bool(args.disable_compile))

        if dist_ctx.enabled:
            model = DDP(
                model,
                device_ids=[dist_ctx.local_rank] if device.type == "cuda" else None,
                output_device=dist_ctx.local_rank if device.type == "cuda" else None,
                find_unused_parameters=bool(training_cfg.get("ddp_find_unused_parameters", False)),
                gradient_as_bucket_view=bool(training_cfg.get("ddp_gradient_as_bucket_view", True)),
                static_graph=bool(training_cfg.get("ddp_static_graph", True)),
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        )

        precision = str(training_cfg.get("precision", "fp32")).lower()
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unsupported precision '{precision}'. Use one of: fp32, bf16, fp16.")

        use_grad_scaler = device.type == "cuda" and precision == "fp16"
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

        pad_token_id = smiles_tokenizer.pad_token_id
        bos_token_id = smiles_tokenizer.cls_token_id
        mask_token_id = smiles_tokenizer.mask_token_id
        mask_prob_floor = float(cfg.get("diffusion", {}).get("mask_prob_floor", 1e-3))
        condition_dropout_prob = float(cfg.get("diffusion", {}).get("condition_dropout_prob", 0.0))
        non_blocking = device.type == "cuda"
        warmup_steps = max(0, int(args.warmup_steps))
        measured_steps = max(1, int(args.steps))
        total_steps = warmup_steps + measured_steps

        iterator = iter(train_loader)
        local_measured_tokens = 0.0
        local_loss_sum = 0.0
        timed_start = None
        model.train()
        optimizer.zero_grad(set_to_none=True)

        oom_happened = False

        for step_idx in range(total_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            target_tokens, ir_data, nmr_tokens = batch
            target_tokens = target_tokens.to(device, non_blocking=non_blocking)
            nmr_tokens = nmr_tokens.to(device, non_blocking=non_blocking)
            if ir_data is not None:
                ir_data = ir_data.to(device, non_blocking=non_blocking)

            if step_idx == warmup_steps:
                _sync_if_needed(device)
                timed_start = time.perf_counter()

            try:
                with _autocast_context(device, precision):
                    if args.mode == "ar":
                        logits = model(
                            nmr_tokens=nmr_tokens,
                            ir_data=ir_data,
                            target_seq=target_tokens[:, :-1],
                        )
                        loss = compute_next_token_loss(logits, target_tokens, pad_token_id)
                        token_count = float((target_tokens[:, 1:] != pad_token_id).sum().item())
                    else:
                        noisy_tokens, masked_indices, p_mask, valid_positions = sample_noisy_targets(
                            target_tokens=target_tokens,
                            mask_token_id=mask_token_id,
                            pad_token_id=pad_token_id,
                            bos_token_id=bos_token_id,
                            mask_prob_floor=mask_prob_floor,
                        )
                        cond_nmr_tokens, cond_ir_data, cond_nmr_padding_mask, _ = apply_condition_dropout(
                            nmr_tokens=nmr_tokens,
                            ir_data=ir_data,
                            condition_dropout_prob=condition_dropout_prob,
                            nmr_pad_token_id=nmr_pad_token_id,
                        )
                        logits = model(
                            nmr_tokens=cond_nmr_tokens,
                            ir_data=cond_ir_data,
                            target_seq=noisy_tokens,
                            target_padding_mask=target_tokens.eq(pad_token_id),
                            nmr_padding_mask=cond_nmr_padding_mask,
                        )
                        loss = compute_masked_diffusion_loss(
                            logits=logits,
                            target_tokens=target_tokens,
                            masked_indices=masked_indices,
                            p_mask=p_mask,
                            valid_positions=valid_positions,
                        )
                        token_count = float(valid_positions.sum().item())

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    oom_happened = True
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise

            if step_idx >= warmup_steps:
                local_loss_sum += float(loss.item())
                local_measured_tokens += token_count

        local_oom = 1.0 if oom_happened else 0.0
        global_oom = _max_reduce(local_oom, device, dist_ctx)
        global_batch = int(args.batch_size) * dist_ctx.world_size

        if global_oom > 0:
            if dist_ctx.is_main:
                print(
                    f"[result] status=oom mode={args.mode} batch_size={int(args.batch_size)} "
                    f"world_size={dist_ctx.world_size} global_batch={global_batch}"
                )
            return 3

        _sync_if_needed(device)
        if timed_start is None:
            timed_start = time.perf_counter()
        elapsed = max(time.perf_counter() - timed_start, 1e-9)
        elapsed_max = _max_reduce(elapsed, device, dist_ctx)

        global_tokens, _ = _reduce_pair(local_measured_tokens, 0.0, device, dist_ctx)
        global_loss_sum, global_count = _reduce_pair(local_loss_sum, float(measured_steps), device, dist_ctx)
        tok_s = global_tokens / elapsed_max
        mean_loss = global_loss_sum / max(global_count, 1.0)
        step_ms = (elapsed_max / measured_steps) * 1000.0

        max_mem_gib = 0.0
        if device.type == "cuda":
            max_mem_gib = torch.cuda.max_memory_allocated(device=device) / (1024**3)
        max_mem_gib = _max_reduce(max_mem_gib, device, dist_ctx)

        if dist_ctx.is_main:
            payload = {
                "status": "ok",
                "mode": args.mode,
                "batch_size": int(args.batch_size),
                "world_size": dist_ctx.world_size,
                "global_batch": global_batch,
                "tok_s": round(tok_s, 2),
                "step_ms": round(step_ms, 3),
                "elapsed_s": round(elapsed_max, 3),
                "measured_steps": measured_steps,
                "mean_loss": round(mean_loss, 6),
                "max_mem_gib": round(max_mem_gib, 3),
            }
            print(
                "[result] "
                f"status=ok mode={payload['mode']} batch_size={payload['batch_size']} "
                f"world_size={payload['world_size']} global_batch={payload['global_batch']} "
                f"tok_s={payload['tok_s']:.2f} step_ms={payload['step_ms']:.3f} "
                f"elapsed_s={payload['elapsed_s']:.3f} measured_steps={payload['measured_steps']} "
                f"mean_loss={payload['mean_loss']:.6f} max_mem_gib={payload['max_mem_gib']:.3f}"
            )
            print("[result_json] " + json.dumps(payload, sort_keys=True))
        return 0
    finally:
        _cleanup_dist(dist_ctx)


if __name__ == "__main__":
    raise SystemExit(main())
