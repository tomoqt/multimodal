#!/usr/bin/env python3
"""
Lean LLADA-style diffusion pretraining entrypoint for multimodal SMILES.
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.diffusion_inference import DiffusionInference
from models.multimodal_block_diffusion import MultiModalBlockDiffusionModel
from models.multimodal_diffusion import MultiModalDiffusionModel
from models.multimodal_prefix_diffusion import MultiModalPrefixDiffusionModel
from models.multimodal_unified_diffusion import MultiModalUnifiedDiffusionModel
from models.smiles_tokenizer import SmilesTokenizer
from training.diffusion_core import (
    apply_condition_dropout,
    build_smiles_priority_token_ids,
    compute_masked_diffusion_loss,
    iter_generated_blocks,
    prepare_block_diffusion_targets,
    prepare_diffusion_targets,
)
from training.diffusion_scheduler import load_ar_criticality_cache, resolve_position_scores, unpack_batch
from training.train_autoregressive import (
    DistContext,
    _autocast_context,
    _cleanup_dist,
    _compute_decode_metrics,
    _decode_target_smiles,
    _infer_dist_context,
    _reduce_pair,
    _seed_everything,
    _set_perf_flags,
    _write_run_artifacts,
    create_loaders,
    load_config,
    load_nmr_tokenizer,
)


def load_diffusion_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config(path)
    diffusion_cfg = cfg.setdefault("diffusion", {})
    diffusion_cfg.setdefault("architecture", "cross_attention")
    diffusion_cfg.setdefault("mask_schedule", "uniform")
    diffusion_cfg.setdefault("mask_prob_floor", 1e-3)
    diffusion_cfg.setdefault("mask_prob_ceiling", 1.0)
    diffusion_cfg.setdefault("priority_weight", 2.0)
    diffusion_cfg.setdefault("complementary_masking", False)
    diffusion_cfg.setdefault("ar_criticality_cache_path", None)
    diffusion_cfg.setdefault("sampling_steps", max(1, int(cfg["model"]["max_seq_length"]) - 1))
    diffusion_cfg.setdefault("sampling_block_length", max(1, int(cfg["model"]["max_seq_length"]) - 1))
    diffusion_cfg.setdefault("sampling_temperature", 0.0)
    diffusion_cfg.setdefault("sampling_remasking", "low_confidence")
    diffusion_cfg.setdefault("block_size", 8)
    diffusion_cfg.setdefault("cfg_scale", 0.0)
    diffusion_cfg.setdefault("condition_dropout_prob", 0.0)
    diffusion_cfg.setdefault("decode_validate_examples", 0)
    return cfg


def build_diffusion_model(
    cfg: Dict[str, Any],
    smiles_tokenizer: SmilesTokenizer,
    nmr_tokenizer: Dict[str, int],
) -> torch.nn.Module:
    model_cfg = cfg["model"]
    diffusion_cfg = cfg["diffusion"]
    arch = str(diffusion_cfg.get("architecture", "cross_attention")).lower()

    common_kwargs = dict(
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
    )

    if arch == "cross_attention":
        return MultiModalDiffusionModel(**common_kwargs)
    if arch == "block_diffusion":
        return MultiModalBlockDiffusionModel(**common_kwargs)
    if arch == "prefix_encoder":
        return MultiModalPrefixDiffusionModel(**common_kwargs)
    if arch == "unified_encoder":
        return MultiModalUnifiedDiffusionModel(**common_kwargs)
    raise ValueError(f"Unsupported diffusion architecture: {arch}")


@torch.no_grad()
def evaluate_diffusion_loss(
    model: torch.nn.Module,
    loader,
    pad_token_id: int,
    bos_token_id: int,
    mask_token_id: int,
    mask_prob_floor: float,
    mask_prob_ceiling: float,
    nmr_pad_token_id: int,
    device: torch.device,
    precision: str,
    dist_ctx: DistContext,
    mask_schedule: str,
    priority_weight: float,
    complementary_masking: bool,
    priority_token_ids,
    ar_criticality_cache,
    architecture: str,
    block_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0.0
    non_blocking = device.type == "cuda"

    for batch in loader:
        target_tokens, ir_data, nmr_tokens, example_indices = unpack_batch(batch)
        target_tokens = target_tokens.to(device, non_blocking=non_blocking)
        nmr_tokens = nmr_tokens.to(device, non_blocking=non_blocking)
        if ir_data is not None:
            ir_data = ir_data.to(device, non_blocking=non_blocking)
        position_scores = resolve_position_scores(
            target_tokens=target_tokens,
            mask_schedule=mask_schedule,
            priority_token_ids=priority_token_ids,
            ar_criticality_cache=ar_criticality_cache,
            example_indices=example_indices,
        )

        if architecture == "block_diffusion":
            block_losses = []
            for block_start, block_end in iter_generated_blocks(target_tokens.size(1), block_size):
                noisy_tokens, masked_indices, p_mask, valid_positions = prepare_block_diffusion_targets(
                    target_tokens=target_tokens,
                    block_start=block_start,
                    block_end=block_end,
                    mask_token_id=mask_token_id,
                    pad_token_id=pad_token_id,
                    bos_token_id=bos_token_id,
                    mask_prob_floor=mask_prob_floor,
                    mask_prob_ceiling=mask_prob_ceiling,
                    position_scores=position_scores,
                    priority_weight=priority_weight,
                )
                if not valid_positions.any():
                    continue
                with _autocast_context(device, precision):
                    logits = model(
                        nmr_tokens=nmr_tokens,
                        ir_data=ir_data,
                        target_seq=noisy_tokens,
                        target_padding_mask=noisy_tokens.eq(pad_token_id),
                        nmr_padding_mask=nmr_tokens.eq(nmr_pad_token_id),
                        block_start=block_start,
                        block_end=block_end,
                    )
                    block_loss = compute_masked_diffusion_loss(
                        logits=logits,
                        target_tokens=target_tokens,
                        masked_indices=masked_indices,
                        p_mask=p_mask,
                        valid_positions=valid_positions,
                    )
                block_losses.append(block_loss)
            loss = torch.stack(block_losses).mean() if block_losses else torch.zeros((), device=device)
        else:
            noisy_tokens, masked_indices, p_mask, valid_positions = prepare_diffusion_targets(
                target_tokens=target_tokens,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                mask_prob_floor=mask_prob_floor,
                mask_prob_ceiling=mask_prob_ceiling,
                position_scores=position_scores,
                priority_weight=priority_weight,
                complementary_masking=complementary_masking,
            )
            if complementary_masking:
                target_tokens = torch.cat([target_tokens, target_tokens], dim=0)
                nmr_tokens = torch.cat([nmr_tokens, nmr_tokens], dim=0)
                if ir_data is not None:
                    ir_data = torch.cat([ir_data, ir_data], dim=0)
            with _autocast_context(device, precision):
                logits = model(
                    nmr_tokens=nmr_tokens,
                    ir_data=ir_data,
                    target_seq=noisy_tokens,
                    target_padding_mask=target_tokens.eq(pad_token_id),
                    nmr_padding_mask=nmr_tokens.eq(nmr_pad_token_id),
                )
                loss = compute_masked_diffusion_loss(
                    logits=logits,
                    target_tokens=target_tokens,
                    masked_indices=masked_indices,
                    p_mask=p_mask,
                    valid_positions=valid_positions,
                )
        total_loss += float(loss.item())
        count += 1.0

    global_total, global_count = _reduce_pair(total_loss, count, device, dist_ctx)
    return global_total / max(global_count, 1.0)


@torch.no_grad()
def evaluate_decode_metrics(
    model: torch.nn.Module,
    dataset,
    tokenizer: SmilesTokenizer,
    device: torch.device,
    max_examples: int,
    max_len: int,
    steps: int,
    block_length: int,
    temperature: float,
    remasking: str,
    ir_as_prompt: bool,
    cfg_scale: float,
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()

    inference = DiffusionInference(model, tokenizer, device=device, ir_as_prompt=ir_as_prompt)
    n = min(max_examples, len(dataset))
    predictions: List[str] = []
    targets: List[str] = []
    for idx in range(n):
        row = dataset[idx]
        if len(row) == 4:
            target_tokens, ir_data, nmr_tokens, _ = row
        else:
            target_tokens, ir_data, nmr_tokens = row
        pred = inference.decode(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            max_len=max_len,
            steps=steps,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            cfg_scale=cfg_scale,
        )[0]
        predictions.append(pred)
        targets.append(_decode_target_smiles(tokenizer, target_tokens))

    metrics = _compute_decode_metrics(predictions, targets)
    if was_training:
        model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Lean LLADA-style diffusion pretraining (DDP-ready)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to resume")
    args = parser.parse_args()

    cfg = load_diffusion_config(args.config)
    training_cfg = cfg["training"]
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
                raise RuntimeError("MPS DDP is not supported in this trainer. Use CUDA + NCCL for multi-GPU.")
            device = torch.device("mps")
        else:
            raise ValueError(f"Unsupported device '{requested_device_name}'.")

        _set_perf_flags(training_cfg)
        base_seed = int(training_cfg.get("seed", 1337))
        _seed_everything(base_seed + dist_ctx.rank)

        tokenized_dir = Path(cfg["data"]["tokenized_dir"])
        if not tokenized_dir.exists():
            raise FileNotFoundError(f"Tokenized directory not found: {tokenized_dir}")

        if dist_ctx.is_main:
            print(
                f"[setup] device={device} tokenized_dir={tokenized_dir} "
                f"ddp={dist_ctx.enabled} world_size={dist_ctx.world_size}"
            )

        smiles_vocab_path = Path(__file__).with_name("vocab.txt")
        smiles_tokenizer = SmilesTokenizer(vocab_file=str(smiles_vocab_path))
        nmr_tokenizer = load_nmr_tokenizer(tokenized_dir)
        nmr_pad_token_id = int(nmr_tokenizer.get("<PAD>", 0))
        diffusion_cfg = cfg["diffusion"]
        architecture = str(diffusion_cfg.get("architecture", "cross_attention")).lower()
        block_size = int(diffusion_cfg.get("block_size", 8))
        mask_schedule = str(diffusion_cfg.get("mask_schedule", "uniform")).lower()
        if mask_schedule not in {"uniform", "token_priority", "ar_criticality"}:
            raise ValueError(f"Unsupported diffusion mask_schedule='{mask_schedule}'.")
        priority_token_ids = build_smiles_priority_token_ids(smiles_tokenizer) if mask_schedule == "token_priority" else None
        ar_criticality_cache = (
            load_ar_criticality_cache(str(diffusion_cfg["ar_criticality_cache_path"]))
            if mask_schedule == "ar_criticality"
            else None
        )
        train_loader, val_loader, test_loader = create_loaders(
            tokenized_dir=tokenized_dir,
            smiles_tokenizer=smiles_tokenizer,
            nmr_tokenizer=nmr_tokenizer,
            config=cfg,
            dist_ctx=dist_ctx,
            device=device,
            return_train_index=mask_schedule == "ar_criticality",
            return_eval_index=mask_schedule == "ar_criticality",
        )

        model = build_diffusion_model(cfg, smiles_tokenizer, nmr_tokenizer).to(device)
        parameter_count = int(sum(p.numel() for p in model.parameters()))
        run_started_at_utc = datetime.now(timezone.utc).isoformat()

        compile_enabled = bool(training_cfg.get("compile", False))
        if compile_enabled:
            if hasattr(torch, "compile"):
                model = torch.compile(
                    model,
                    mode=str(training_cfg.get("compile_mode", "max-autotune")),
                    dynamic=bool(training_cfg.get("compile_dynamic", False)),
                    fullgraph=bool(training_cfg.get("compile_fullgraph", False)),
                )
                if dist_ctx.is_main:
                    print("[setup] torch.compile enabled")
            elif dist_ctx.is_main:
                print("[setup] torch.compile requested but unavailable; continuing without compile.")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        )

        start_epoch = 0
        checkpoint_payload = None
        if args.checkpoint:
            checkpoint_payload = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint_payload["model_state_dict"])
            optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
            start_epoch = int(checkpoint_payload.get("epoch", 0)) + 1
            if dist_ctx.is_main:
                print(f"Loaded checkpoint {args.checkpoint} at epoch {start_epoch}")

        if dist_ctx.enabled:
            model = DDP(
                model,
                device_ids=[dist_ctx.local_rank] if device.type == "cuda" else None,
                output_device=dist_ctx.local_rank if device.type == "cuda" else None,
                find_unused_parameters=bool(training_cfg.get("ddp_find_unused_parameters", False)),
                gradient_as_bucket_view=bool(training_cfg.get("ddp_gradient_as_bucket_view", True)),
                static_graph=bool(training_cfg.get("ddp_static_graph", True)),
            )

        precision = str(training_cfg.get("precision", "fp32")).lower()
        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unsupported precision '{precision}'. Use one of: fp32, bf16, fp16.")
        use_grad_scaler = device.type == "cuda" and precision == "fp16"
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
        if scaler.is_enabled() and checkpoint_payload is not None:
            scaler_state = checkpoint_payload.get("scaler_state_dict")
            if scaler_state:
                scaler.load_state_dict(scaler_state)

        grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
        grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
        log_every = int(training_cfg.get("log_every_steps", 20))
        non_blocking = device.type == "cuda"

        num_epochs = int(training_cfg["num_epochs"])
        validate_every_epochs = max(1, int(training_cfg.get("validate_every_epochs", 1)))
        run_final_test = bool(training_cfg.get("run_final_test", True))
        decode_validate_examples = max(0, int(cfg["diffusion"].get("decode_validate_examples", 0)))
        pad_token_id = smiles_tokenizer.pad_token_id
        bos_token_id = smiles_tokenizer.cls_token_id
        mask_token_id = smiles_tokenizer.mask_token_id
        output_dir = Path(cfg["checkpoint"].get("output_dir", "checkpoints_diffusion"))
        save_every_epoch = bool(cfg["checkpoint"].get("save_every_epoch", True))
        save_every_n_epochs = max(1, int(cfg["checkpoint"].get("save_every_n_epochs", 1)))
        if dist_ctx.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_run_artifacts(
                output_dir=output_dir,
                cfg=cfg,
                tokenized_dir=tokenized_dir,
                smiles_vocab_path=smiles_vocab_path,
                run_started_at_utc=run_started_at_utc,
                world_size=dist_ctx.world_size,
                parameter_count=parameter_count,
                latest_metrics={},
            )
            print(f"[diffusion] architecture={cfg['diffusion']['architecture']}")
            print(f"[diffusion] mask_schedule={mask_schedule}")
        if dist_ctx.enabled:
            dist_ctx.enabled and torch.distributed.barrier()

        condition_dropout_prob = float(diffusion_cfg.get("condition_dropout_prob", 0.0))
        mask_prob_floor = float(diffusion_cfg.get("mask_prob_floor", 1e-3))
        mask_prob_ceiling = float(diffusion_cfg.get("mask_prob_ceiling", 1.0))
        priority_weight = float(diffusion_cfg.get("priority_weight", 2.0))
        complementary_masking = bool(diffusion_cfg.get("complementary_masking", False))
        last_epoch_metrics: Dict[str, Any] = {}
        for epoch in range(start_epoch, num_epochs):
            if dist_ctx.enabled and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss_sum = 0.0
            valid_tokens = 0.0
            epoch_start = time.perf_counter()
            micro_steps = 0
            optimizer_steps = 0

            for step, batch in enumerate(train_loader, start=1):
                target_tokens, ir_data, nmr_tokens, example_indices = unpack_batch(batch)
                target_tokens = target_tokens.to(device, non_blocking=non_blocking)
                nmr_tokens = nmr_tokens.to(device, non_blocking=non_blocking)
                if ir_data is not None:
                    ir_data = ir_data.to(device, non_blocking=non_blocking)
                position_scores = resolve_position_scores(
                    target_tokens=target_tokens,
                    mask_schedule=mask_schedule,
                    priority_token_ids=priority_token_ids,
                    ar_criticality_cache=ar_criticality_cache,
                    example_indices=example_indices,
                )

                micro_steps += 1

                if architecture == "block_diffusion":
                    cond_nmr_tokens, cond_ir_data, cond_nmr_padding_mask, _ = apply_condition_dropout(
                        nmr_tokens=nmr_tokens,
                        ir_data=ir_data,
                        condition_dropout_prob=condition_dropout_prob,
                        nmr_pad_token_id=nmr_pad_token_id,
                    )
                    block_losses = []
                    block_valid_tokens = 0.0
                    for block_start, block_end in iter_generated_blocks(target_tokens.size(1), block_size):
                        noisy_tokens, masked_indices, p_mask, valid_positions = prepare_block_diffusion_targets(
                            target_tokens=target_tokens,
                            block_start=block_start,
                            block_end=block_end,
                            mask_token_id=mask_token_id,
                            pad_token_id=pad_token_id,
                            bos_token_id=bos_token_id,
                            mask_prob_floor=mask_prob_floor,
                            mask_prob_ceiling=mask_prob_ceiling,
                            position_scores=position_scores,
                            priority_weight=priority_weight,
                        )
                        if not valid_positions.any():
                            continue
                        block_valid_tokens += float(valid_positions.sum().item())
                        with _autocast_context(device, precision):
                            logits = model(
                                nmr_tokens=cond_nmr_tokens,
                                ir_data=cond_ir_data,
                                target_seq=noisy_tokens,
                                target_padding_mask=noisy_tokens.eq(pad_token_id),
                                nmr_padding_mask=cond_nmr_padding_mask,
                                block_start=block_start,
                                block_end=block_end,
                            )
                            block_loss = compute_masked_diffusion_loss(
                                logits=logits,
                                target_tokens=target_tokens,
                                masked_indices=masked_indices,
                                p_mask=p_mask,
                                valid_positions=valid_positions,
                            )
                            block_losses.append(block_loss)
                    valid_tokens += block_valid_tokens
                    loss = torch.stack(block_losses).mean() if block_losses else torch.zeros((), device=device)
                    scaled_loss = loss / grad_accum_steps
                else:
                    noisy_tokens, masked_indices, p_mask, valid_positions = prepare_diffusion_targets(
                        target_tokens=target_tokens,
                        mask_token_id=mask_token_id,
                        pad_token_id=pad_token_id,
                        bos_token_id=bos_token_id,
                        mask_prob_floor=mask_prob_floor,
                        mask_prob_ceiling=mask_prob_ceiling,
                        position_scores=position_scores,
                        priority_weight=priority_weight,
                        complementary_masking=complementary_masking,
                    )
                    if complementary_masking:
                        target_tokens = torch.cat([target_tokens, target_tokens], dim=0)
                        nmr_tokens = torch.cat([nmr_tokens, nmr_tokens], dim=0)
                        if ir_data is not None:
                            ir_data = torch.cat([ir_data, ir_data], dim=0)
                    valid_tokens += float(valid_positions.sum().item())

                    with _autocast_context(device, precision):
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
                        scaled_loss = loss / grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                running_loss_sum += float(loss.item())
                should_step = (step % grad_accum_steps) == 0
                if should_step:
                    if grad_clip_norm > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

                if log_every > 0 and step % log_every == 0:
                    global_loss_sum, global_micro = _reduce_pair(running_loss_sum, float(micro_steps), device, dist_ctx)
                    global_tokens, _ = _reduce_pair(valid_tokens, 0.0, device, dist_ctx)
                    elapsed = max(time.perf_counter() - epoch_start, 1e-6)
                    if dist_ctx.is_main:
                        print(
                            f"[epoch {epoch+1}/{num_epochs}] step={step} "
                            f"train_loss={global_loss_sum/max(global_micro, 1.0):.4f} "
                            f"tok_s={global_tokens/elapsed:.1f} elapsed_s={elapsed:.1f} "
                            f"opt_steps={optimizer_steps}"
                        )

            if (micro_steps % grad_accum_steps) != 0:
                if grad_clip_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            global_loss_sum, global_micro = _reduce_pair(running_loss_sum, float(micro_steps), device, dist_ctx)
            global_tokens, _ = _reduce_pair(valid_tokens, 0.0, device, dist_ctx)
            train_loss = global_loss_sum / max(global_micro, 1.0)

            should_validate = ((epoch + 1) % validate_every_epochs) == 0 or (epoch + 1) == num_epochs
            val_loss = (
                evaluate_diffusion_loss(
                    model=model,
                    loader=val_loader,
                    pad_token_id=pad_token_id,
                    bos_token_id=bos_token_id,
                    mask_token_id=mask_token_id,
                    mask_prob_floor=mask_prob_floor,
                    mask_prob_ceiling=mask_prob_ceiling,
                    nmr_pad_token_id=nmr_pad_token_id,
                    device=device,
                    precision=precision,
                    dist_ctx=dist_ctx,
                    mask_schedule=mask_schedule,
                    priority_weight=priority_weight,
                    complementary_masking=complementary_masking,
                    priority_token_ids=priority_token_ids,
                    ar_criticality_cache=ar_criticality_cache,
                    architecture=architecture,
                    block_size=block_size,
                )
                if should_validate
                else float("nan")
            )
            epoch_elapsed = max(time.perf_counter() - epoch_start, 1e-6)
            if dist_ctx.is_main:
                if should_validate:
                    print(
                        f"[epoch {epoch+1}/{num_epochs}] train_loss={train_loss:.4f} "
                        f"val_loss={val_loss:.4f} tok_s={global_tokens/epoch_elapsed:.1f} "
                        f"elapsed_s={epoch_elapsed:.1f} opt_steps={optimizer_steps}"
                    )
                else:
                    print(
                        f"[epoch {epoch+1}/{num_epochs}] train_loss={train_loss:.4f} "
                        f"val_skipped tok_s={global_tokens/epoch_elapsed:.1f} "
                        f"elapsed_s={epoch_elapsed:.1f} opt_steps={optimizer_steps}"
                    )

            epoch_decode_metrics: Optional[Dict[str, Any]] = None
            if should_validate and decode_validate_examples > 0:
                if dist_ctx.is_main:
                    model_for_decode = model.module if isinstance(model, DDP) else model
                    epoch_decode_metrics = evaluate_decode_metrics(
                        model=model_for_decode,
                        dataset=val_loader.dataset,
                        tokenizer=smiles_tokenizer,
                        device=device,
                        max_examples=decode_validate_examples,
                        max_len=int(cfg["model"]["max_seq_length"]),
                        steps=int(diffusion_cfg["sampling_steps"]),
                        block_length=int(diffusion_cfg["sampling_block_length"]),
                        temperature=float(diffusion_cfg["sampling_temperature"]),
                        remasking=str(diffusion_cfg["sampling_remasking"]),
                        ir_as_prompt=bool(cfg["model"].get("ir_as_prompt", False)),
                        cfg_scale=float(diffusion_cfg.get("cfg_scale", 0.0)),
                    )
                    decode_valid = float(epoch_decode_metrics.get("decode_valid_smiles", float("nan")))
                    decode_exact = float(epoch_decode_metrics.get("decode_exact_match_all", float("nan")))
                    decode_tani = float(epoch_decode_metrics.get("decode_avg_tanimoto", float("nan")))
                    decode_ecfp6 = float(epoch_decode_metrics.get("decode_avg_ecfp6_iou", float("nan")))
                    decode_mcs = float(epoch_decode_metrics.get("decode_avg_mcs_over_target", float("nan")))
                    decode_examples = int(epoch_decode_metrics.get("decode_examples", 0))
                    decode_valid_str = f"{decode_valid:.4f}" if math.isfinite(decode_valid) else "nan"
                    decode_exact_str = f"{decode_exact:.4f}" if math.isfinite(decode_exact) else "nan"
                    decode_tani_str = f"{decode_tani:.4f}" if math.isfinite(decode_tani) else "nan"
                    decode_ecfp6_str = f"{decode_ecfp6:.4f}" if math.isfinite(decode_ecfp6) else "nan"
                    decode_mcs_str = f"{decode_mcs:.4f}" if math.isfinite(decode_mcs) else "nan"
                    metric_mode = "rdkit" if bool(epoch_decode_metrics.get("decode_rdkit", False)) else "string"
                    print(
                        f"[epoch {epoch+1}/{num_epochs}] decode_examples={decode_examples} "
                        f"valid_smiles={decode_valid_str} exact_match_all={decode_exact_str} "
                        f"avg_tanimoto={decode_tani_str} avg_ecfp6_iou={decode_ecfp6_str} "
                        f"avg_mcs_over_target={decode_mcs_str} mode={metric_mode}"
                    )
                if dist_ctx.enabled:
                    torch.distributed.barrier()

            latest_epoch_metrics: Dict[str, Any] = {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "tok_s": float(global_tokens / epoch_elapsed),
                "epoch_elapsed_s": float(epoch_elapsed),
                "optimizer_steps": int(optimizer_steps),
            }
            if epoch_decode_metrics is not None:
                latest_epoch_metrics.update(epoch_decode_metrics)
            last_epoch_metrics = latest_epoch_metrics

            should_save = save_every_epoch and (((epoch + 1) % save_every_n_epochs) == 0 or (epoch + 1) == num_epochs)
            if should_save and dist_ctx.is_main:
                model_to_save = model.module if isinstance(model, DDP) else model
                ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "decode_metrics": epoch_decode_metrics,
                        "config": cfg,
                        "world_size": dist_ctx.world_size,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")
                latest_with_checkpoint = dict(latest_epoch_metrics)
                latest_with_checkpoint["checkpoint"] = str(ckpt_path.name)
                _write_run_artifacts(
                    output_dir=output_dir,
                    cfg=cfg,
                    tokenized_dir=tokenized_dir,
                    smiles_vocab_path=smiles_vocab_path,
                    run_started_at_utc=run_started_at_utc,
                    world_size=dist_ctx.world_size,
                    parameter_count=parameter_count,
                    latest_metrics=latest_with_checkpoint,
                )

        if run_final_test:
            test_loss = evaluate_diffusion_loss(
                model=model,
                loader=test_loader,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                mask_token_id=mask_token_id,
                mask_prob_floor=mask_prob_floor,
                mask_prob_ceiling=mask_prob_ceiling,
                nmr_pad_token_id=nmr_pad_token_id,
                device=device,
                precision=precision,
                dist_ctx=dist_ctx,
                mask_schedule=mask_schedule,
                priority_weight=priority_weight,
                complementary_masking=complementary_masking,
                priority_token_ids=priority_token_ids,
                ar_criticality_cache=ar_criticality_cache,
            )
            if dist_ctx.is_main:
                print(f"[final] test_loss={test_loss:.4f}")
                final_metrics = dict(last_epoch_metrics)
                final_metrics["final_test_loss"] = float(test_loss)
                _write_run_artifacts(
                    output_dir=output_dir,
                    cfg=cfg,
                    tokenized_dir=tokenized_dir,
                    smiles_vocab_path=smiles_vocab_path,
                    run_started_at_utc=run_started_at_utc,
                    world_size=dist_ctx.world_size,
                    parameter_count=parameter_count,
                    latest_metrics=final_metrics,
                )
        elif dist_ctx.is_main:
            print("[final] test_skipped")

    finally:
        _cleanup_dist(dist_ctx)


if __name__ == "__main__":
    main()
