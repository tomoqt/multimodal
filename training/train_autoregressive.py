#!/usr/bin/env python3
"""
Lean autoregressive pretraining entrypoint with optional DDP.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.multimodal_prefix_to_smiles import MultiModalPrefixToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from inference.inference import DecodingStrategy, ModelInference
from training.core_dataset import SpectralSmilesDataset, collate_spectral_smiles
from training.core_train import compute_next_token_loss


@dataclass
class DistContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: Optional[str] = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _deep_update(base: Dict, patch: Dict) -> Dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def load_config(path: str = None) -> Dict:
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
            "architecture": "cross_attention",
        },
        "training": {
            "batch_size": 8,
            "test_batch_size": 8,
            "num_epochs": 1,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "drop_last": True,
            "log_every_steps": 20,
            "seed": 1337,
            "device": _default_device(),
            "precision": "bf16" if torch.cuda.is_available() else "fp32",
            "grad_accum_steps": 1,
            "grad_clip_norm": 0.0,
            "allow_tf32": True,
            "cudnn_benchmark": True,
            "matmul_precision": "high",
            "ddp_backend": "nccl",
            "ddp_timeout_minutes": 30,
            "ddp_find_unused_parameters": False,
            "ddp_static_graph": True,
            "ddp_gradient_as_bucket_view": True,
            "compile": False,
            "compile_mode": "max-autotune",
            "compile_dynamic": False,
            "compile_fullgraph": False,
            "validate_every_epochs": 1,
            "run_final_test": True,
            "decode_validate_examples": 0,
            "decode_validate_max_len": 0,
        },
        "data": {
            "tokenized_dir": "data/tokenized_baseline/data",
            "pretokenize": True,
            "preload_ir": False,
            "use_disk_cache": True,
            "write_disk_cache": True,
            "cache_dir": None,
            "ir_cache_mode": "none",
            "ir_cache_dtype": "float32",
        },
        "checkpoint": {
            "output_dir": "checkpoints",
            "save_every_epoch": True,
            "save_every_n_epochs": 1,
        },
    }
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_update(config, user_cfg)
    return config


def load_nmr_tokenizer(tokenized_dir: Path) -> Dict[str, int]:
    vocab_json = tokenized_dir.parent / "vocab.json"
    if not vocab_json.exists():
        raise FileNotFoundError(f"Missing NMR vocabulary at {vocab_json}")
    with vocab_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _set_perf_flags(training_cfg: Dict) -> None:
    allow_tf32 = bool(training_cfg.get("allow_tf32", True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(training_cfg.get("cudnn_benchmark", True))
    matmul_precision = str(training_cfg.get("matmul_precision", "high"))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_dist_context(training_cfg: Dict, requested_device: str) -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistContext(enabled=False)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if requested_device not in {"cuda", "cpu"}:
        raise RuntimeError(
            f"DDP requested via WORLD_SIZE={world_size}, but device '{requested_device}' is unsupported for DDP."
        )

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with CUDA requested but CUDA is unavailable.")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds visible CUDA device count={torch.cuda.device_count()}."
            )
        torch.cuda.set_device(local_rank)

    backend = str(training_cfg.get("ddp_backend", "nccl" if requested_device == "cuda" else "gloo"))
    timeout_minutes = float(training_cfg.get("ddp_timeout_minutes", 30))
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
    return DistContext(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank, backend=backend)


def _cleanup_dist(ctx: DistContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    p = precision.lower()
    if p == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if p == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _reduce_pair(total: float, count: float, device: torch.device, dist_ctx: DistContext) -> Tuple[float, float]:
    stats = torch.tensor([total, count], device=device, dtype=torch.float32)
    if dist_ctx.enabled:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), float(stats[1].item())


def _quick_file_fingerprint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    st = path.stat()
    out: Dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    try:
        h = hashlib.sha256()
        sample_bytes = 1024 * 1024
        with path.open("rb") as f:
            h.update(f.read(sample_bytes))
            if st.st_size > sample_bytes:
                f.seek(max(st.st_size - sample_bytes, 0))
                h.update(f.read(sample_bytes))
        out["quick_sha256"] = h.hexdigest()
    except Exception:
        pass
    return out


def _copy_if_exists(src: Path, dst: Path) -> Optional[str]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _write_run_artifacts(
    output_dir: Path,
    cfg: Dict,
    tokenized_dir: Path,
    smiles_vocab_path: Path,
    run_started_at_utc: str,
    world_size: int,
    parameter_count: int,
    latest_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = output_dir / "config.yaml"
    with config_snapshot.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    smiles_vocab_snapshot = output_dir / "smiles_vocab.txt"
    _copy_if_exists(smiles_vocab_path, smiles_vocab_snapshot)

    nmr_vocab_src = tokenized_dir.parent / "vocab.json"
    nmr_vocab_snapshot = output_dir / "nmr_vocab.json"
    _copy_if_exists(nmr_vocab_src, nmr_vocab_snapshot)

    dataset_fingerprint = {
        "tokenized_dir": str(tokenized_dir),
        "splits": {
            "train": {
                "src": _quick_file_fingerprint(tokenized_dir / "src-train.txt"),
                "tgt": _quick_file_fingerprint(tokenized_dir / "tgt-train.txt"),
                "ir": _quick_file_fingerprint(tokenized_dir / "ir-train.npy"),
            },
            "val": {
                "src": _quick_file_fingerprint(tokenized_dir / "src-val.txt"),
                "tgt": _quick_file_fingerprint(tokenized_dir / "tgt-val.txt"),
                "ir": _quick_file_fingerprint(tokenized_dir / "ir-val.npy"),
            },
            "test": {
                "src": _quick_file_fingerprint(tokenized_dir / "src-test.txt"),
                "tgt": _quick_file_fingerprint(tokenized_dir / "tgt-test.txt"),
                "ir": _quick_file_fingerprint(tokenized_dir / "ir-test.npy"),
            },
        },
        "nmr_vocab": _quick_file_fingerprint(nmr_vocab_src),
        "smiles_vocab": _quick_file_fingerprint(smiles_vocab_path),
    }
    dataset_fingerprint_path = output_dir / "dataset_fingerprint.json"
    with dataset_fingerprint_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_fingerprint, f, indent=2)

    manifest = {
        "run_started_at_utc": run_started_at_utc,
        "run_updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "world_size": world_size,
        "parameter_count": parameter_count,
        "tokenized_dir": str(tokenized_dir),
        "artifacts": {
            "config": str(config_snapshot.name),
            "dataset_fingerprint": str(dataset_fingerprint_path.name),
            "smiles_vocab": str(smiles_vocab_snapshot.name) if smiles_vocab_snapshot.exists() else None,
            "nmr_vocab": str(nmr_vocab_snapshot.name) if nmr_vocab_snapshot.exists() else None,
        },
        "latest_metrics": latest_metrics or {},
    }
    manifest_path = output_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _decode_target_smiles(tokenizer: SmilesTokenizer, target_tokens: torch.Tensor) -> str:
    ids = target_tokens.tolist()
    cleaned: List[int] = []
    for token_id in ids:
        if token_id == tokenizer.sep_token_id:
            break
        if token_id in (tokenizer.cls_token_id, tokenizer.pad_token_id):
            continue
        cleaned.append(token_id)
    return tokenizer.decode(cleaned).replace(" ", "").strip()


def _compute_decode_metrics(predictions: List[str], targets: List[str]) -> Dict[str, Any]:
    n = len(predictions)
    if n == 0:
        return {
            "decode_examples": 0,
            "decode_rdkit": False,
            "decode_valid_smiles": float("nan"),
            "decode_exact_match_all": float("nan"),
            "decode_exact_match_valid_pairs": float("nan"),
            "decode_avg_tanimoto": float("nan"),
            "decode_avg_ecfp6_iou": float("nan"),
            "decode_avg_mcs_over_target": float("nan"),
            "decode_valid_pairs": 0,
        }

    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem, rdFMCS

        RDLogger.DisableLog("rdApp.*")
        rdkit_available = True
    except Exception:
        rdkit_available = False

    if not rdkit_available:
        exact_all = float(np.mean([int(p == t) for p, t in zip(predictions, targets)]))
        return {
            "decode_examples": n,
            "decode_rdkit": False,
            "decode_valid_smiles": float("nan"),
            "decode_exact_match_all": exact_all,
            "decode_exact_match_valid_pairs": float("nan"),
            "decode_avg_tanimoto": float("nan"),
            "decode_avg_ecfp6_iou": float("nan"),
            "decode_avg_mcs_over_target": float("nan"),
            "decode_valid_pairs": 0,
        }

    valid_pred = 0
    exact_all = 0
    exact_valid = 0
    valid_pairs = 0
    tanimoto_sum = 0.0
    ecfp6_iou_sum = 0.0
    mcs_over_target_sum = 0.0

    for pred, target in zip(predictions, targets):
        pred_clean = pred.replace(" ", "")
        target_clean = target.replace(" ", "")
        mol_pred = Chem.MolFromSmiles(pred_clean)
        mol_target = Chem.MolFromSmiles(target_clean)

        is_valid_pred = mol_pred is not None
        if is_valid_pred:
            valid_pred += 1

        if mol_pred is None or mol_target is None:
            continue

        can_pred = Chem.MolToSmiles(mol_pred, canonical=True)
        can_target = Chem.MolToSmiles(mol_target, canonical=True)
        is_exact = int(can_pred == can_target)
        exact_all += is_exact
        exact_valid += is_exact
        valid_pairs += 1

        fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2)
        fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, 2)
        tanimoto_sum += float(DataStructs.TanimotoSimilarity(fp_pred, fp_target))

        fp3_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, radius=3, nBits=1024)
        fp3_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, radius=3, nBits=1024)
        intersection = sum((fp3_pred & fp3_target))
        union = sum((fp3_pred | fp3_target))
        ecfp6_iou_sum += float(intersection / union) if union > 0 else 0.0

        mcs = rdFMCS.FindMCS([mol_pred, mol_target])
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
        if mcs_mol is not None:
            target_atoms = mol_target.GetNumAtoms()
            mcs_over_target_sum += float(mcs_mol.GetNumAtoms() / target_atoms) if target_atoms > 0 else 0.0

    return {
        "decode_examples": n,
        "decode_rdkit": True,
        "decode_valid_smiles": float(valid_pred / max(n, 1)),
        "decode_exact_match_all": float(exact_all / max(n, 1)),
        "decode_exact_match_valid_pairs": float(exact_valid / max(valid_pairs, 1)),
        "decode_avg_tanimoto": float(tanimoto_sum / max(valid_pairs, 1)),
        "decode_avg_ecfp6_iou": float(ecfp6_iou_sum / max(valid_pairs, 1)),
        "decode_avg_mcs_over_target": float(mcs_over_target_sum / max(valid_pairs, 1)),
        "decode_valid_pairs": int(valid_pairs),
    }


@torch.no_grad()
def evaluate_decode_metrics(
    model: torch.nn.Module,
    dataset: SpectralSmilesDataset,
    tokenizer: SmilesTokenizer,
    device: torch.device,
    max_examples: int,
    max_len: int,
    ir_as_prompt: bool,
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()

    inference = ModelInference(model, tokenizer, device=device, ir_as_prompt=ir_as_prompt)
    n = min(max_examples, len(dataset))
    predictions: List[str] = []
    targets: List[str] = []
    non_blocking = device.type == "cuda"

    for idx in range(n):
        target_tokens, ir_data, nmr_tokens = dataset[idx]
        if ir_data is not None:
            ir_data = ir_data.to(device, non_blocking=non_blocking)
        if nmr_tokens is not None:
            nmr_tokens = nmr_tokens.to(device, non_blocking=non_blocking)

        pred = inference.decode(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            strategy=DecodingStrategy.GREEDY,
            max_len=max_len,
        )[0]
        tgt = _decode_target_smiles(tokenizer, target_tokens)
        predictions.append(pred)
        targets.append(tgt)

    metrics = _compute_decode_metrics(predictions, targets)
    if was_training:
        model.train()
    return metrics


def create_loaders(
    tokenized_dir: Path,
    smiles_tokenizer: SmilesTokenizer,
    nmr_tokenizer: Dict[str, int],
    config: Dict,
    dist_ctx: Optional[DistContext] = None,
    device: Optional[torch.device] = None,
    return_train_index: bool = False,
    return_eval_index: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    dist_ctx = dist_ctx or DistContext(enabled=False)
    device = device or torch.device("cpu")

    train_ds = SpectralSmilesDataset(
        data_dir=str(tokenized_dir),
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split="train",
        max_smiles_len=model_cfg["max_seq_length"],
        max_nmr_len=model_cfg["max_nmr_length"],
        pretokenize=data_cfg.get("pretokenize", True),
        preload_ir=data_cfg.get("preload_ir", False),
        use_disk_cache=data_cfg.get("use_disk_cache", True),
        write_disk_cache=data_cfg.get("write_disk_cache", True),
        cache_dir=data_cfg.get("cache_dir"),
        ir_cache_mode=data_cfg.get("ir_cache_mode", "none"),
        ir_cache_dtype=data_cfg.get("ir_cache_dtype", "float32"),
        return_index=return_train_index,
    )
    val_ds = SpectralSmilesDataset(
        data_dir=str(tokenized_dir),
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split="val",
        max_smiles_len=model_cfg["max_seq_length"],
        max_nmr_len=model_cfg["max_nmr_length"],
        pretokenize=data_cfg.get("pretokenize", True),
        preload_ir=data_cfg.get("preload_ir", False),
        use_disk_cache=data_cfg.get("use_disk_cache", True),
        write_disk_cache=data_cfg.get("write_disk_cache", True),
        cache_dir=data_cfg.get("cache_dir"),
        ir_cache_mode=data_cfg.get("ir_cache_mode", "none"),
        ir_cache_dtype=data_cfg.get("ir_cache_dtype", "float32"),
        return_index=return_eval_index,
    )
    test_ds = SpectralSmilesDataset(
        data_dir=str(tokenized_dir),
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        split="test",
        max_smiles_len=model_cfg["max_seq_length"],
        max_nmr_len=model_cfg["max_nmr_length"],
        pretokenize=data_cfg.get("pretokenize", True),
        preload_ir=data_cfg.get("preload_ir", False),
        use_disk_cache=data_cfg.get("use_disk_cache", True),
        write_disk_cache=data_cfg.get("write_disk_cache", True),
        cache_dir=data_cfg.get("cache_dir"),
        ir_cache_mode=data_cfg.get("ir_cache_mode", "none"),
        ir_cache_dtype=data_cfg.get("ir_cache_dtype", "float32"),
        return_index=return_eval_index,
    )

    if dist_ctx.is_main and data_cfg.get("use_disk_cache", True):
        print(
            "[data] token-cache "
            f"train={'hit' if train_ds.token_cache_hit else 'miss'} "
            f"val={'hit' if val_ds.token_cache_hit else 'miss'} "
            f"test={'hit' if test_ds.token_cache_hit else 'miss'}"
        )
    if dist_ctx.is_main and data_cfg.get("ir_cache_mode", "none") == "pt":
        print(
            "[data] ir-cache "
            f"train={'hit' if train_ds.ir_cache_hit else 'miss'} "
            f"val={'hit' if val_ds.ir_cache_hit else 'miss'} "
            f"test={'hit' if test_ds.ir_cache_hit else 'miss'} "
            f"dtype={data_cfg.get('ir_cache_dtype', 'float32')}"
        )

    collate = lambda b: collate_spectral_smiles(  # noqa: E731
        b,
        smiles_pad_token_id=smiles_tokenizer.pad_token_id,
        nmr_pad_token_id=nmr_tokenizer["<PAD>"],
    )

    train_sampler = None
    val_sampler = None
    test_sampler = None
    if dist_ctx.enabled:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=bool(train_cfg.get("drop_last", True)),
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=False,
            drop_last=False,
        )

    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(train_cfg.get("persistent_workers", num_workers > 0)) and num_workers > 0
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))

    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collate,
    }
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=bool(train_cfg.get("drop_last", True)),
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("test_batch_size", train_cfg["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.get("test_batch_size", train_cfg["batch_size"]),
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    pad_token_id: int,
    nmr_pad_token_id: int,
    device: torch.device,
    precision: str,
    dist_ctx: DistContext,
) -> float:
    model.eval()
    total = 0.0
    count = 0.0
    non_blocking = device.type == "cuda"

    for target_tokens, ir_data, nmr_tokens in loader:
        target_tokens = target_tokens.to(device, non_blocking=non_blocking)
        nmr_tokens = nmr_tokens.to(device, non_blocking=non_blocking)
        if ir_data is not None:
            ir_data = ir_data.to(device, non_blocking=non_blocking)

        with _autocast_context(device, precision):
            decoder_input = target_tokens[:, :-1]
            logits = model(
                nmr_tokens=nmr_tokens,
                ir_data=ir_data,
                target_seq=decoder_input,
                target_padding_mask=(decoder_input == pad_token_id),
                nmr_padding_mask=(nmr_tokens == nmr_pad_token_id),
            )
            loss = compute_next_token_loss(logits, target_tokens, pad_token_id)
        total += float(loss.item())
        count += 1.0

    global_total, global_count = _reduce_pair(total, count, device, dist_ctx)
    return global_total / max(global_count, 1.0)


def _build_model(cfg: Dict, smiles_tokenizer: SmilesTokenizer, nmr_tokenizer: Dict[str, int], device: torch.device):
    architecture = str(cfg["model"].get("architecture", "cross_attention")).lower()
    if architecture == "cross_attention":
        model_cls = MultiModalToSMILESModel
    elif architecture == "prefix":
        model_cls = MultiModalPrefixToSMILESModel
    else:
        raise ValueError(f"Unsupported AR architecture: {architecture}")

    model = model_cls(
        smiles_vocab_size=len(smiles_tokenizer),
        nmr_vocab_size=max(nmr_tokenizer.values()) + 1,
        max_seq_length=cfg["model"]["max_seq_length"],
        max_nmr_length=cfg["model"]["max_nmr_length"],
        max_memory_length=cfg["model"]["max_memory_length"],
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        verbose=False,
        use_stablemax=cfg["model"].get("use_stablemax", False),
        ir_encoder_type=cfg["model"].get("ir_encoder_type", "regular"),
        ir_as_prompt=cfg["model"].get("ir_as_prompt", False),
        use_rmsnorm=cfg["model"].get("use_rmsnorm", False),
    ).to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Lean autoregressive pretraining (DDP-ready)")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
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
        train_loader, val_loader, test_loader = create_loaders(
            tokenized_dir,
            smiles_tokenizer,
            nmr_tokenizer,
            cfg,
            dist_ctx=dist_ctx,
            device=device,
        )

        model = _build_model(cfg, smiles_tokenizer, nmr_tokenizer, device)
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
                print("[setup] torch.compile requested but unavailable on this PyTorch build; continuing without compile.")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg.get("weight_decay", 0.01),
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
        decode_validate_examples = max(0, int(training_cfg.get("decode_validate_examples", 0)))
        decode_validate_max_len_cfg = int(training_cfg.get("decode_validate_max_len", 0))
        pad_token_id = smiles_tokenizer.pad_token_id
        output_dir = Path(cfg["checkpoint"].get("output_dir", "checkpoints"))
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
        if dist_ctx.enabled:
            dist.barrier()

        last_epoch_metrics: Dict[str, Any] = {}
        for epoch in range(start_epoch, num_epochs):
            if dist_ctx.enabled and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss_sum = 0.0
            nonpad_tokens = 0.0
            epoch_start = time.perf_counter()
            micro_steps = 0
            optimizer_steps = 0

            for step, batch in enumerate(train_loader, start=1):
                target_tokens, ir_data, nmr_tokens = batch
                target_tokens = target_tokens.to(device, non_blocking=non_blocking)
                nmr_tokens = nmr_tokens.to(device, non_blocking=non_blocking)
                if ir_data is not None:
                    ir_data = ir_data.to(device, non_blocking=non_blocking)

                nonpad_tokens += float((target_tokens[:, 1:] != pad_token_id).sum().item())
                micro_steps += 1

                with _autocast_context(device, precision):
                    decoder_input = target_tokens[:, :-1]
                    logits = model(
                        nmr_tokens=nmr_tokens,
                        ir_data=ir_data,
                        target_seq=decoder_input,
                        target_padding_mask=(decoder_input == pad_token_id),
                        nmr_padding_mask=(nmr_tokens == nmr_pad_token_id),
                    )
                    loss = compute_next_token_loss(logits, target_tokens, pad_token_id)
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
                    global_tokens, _ = _reduce_pair(nonpad_tokens, 0.0, device, dist_ctx)
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
            global_tokens, _ = _reduce_pair(nonpad_tokens, 0.0, device, dist_ctx)
            train_loss = global_loss_sum / max(global_micro, 1.0)

            should_validate = ((epoch + 1) % validate_every_epochs) == 0 or (epoch + 1) == num_epochs
            val_loss = (
                evaluate_loss(
                    model,
                    val_loader,
                    pad_token_id,
                    nmr_pad_token_id,
                    device,
                    precision,
                    dist_ctx,
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
                    decode_max_len = (
                        decode_validate_max_len_cfg
                        if decode_validate_max_len_cfg > 0
                        else int(cfg["model"]["max_seq_length"])
                    )
                    epoch_decode_metrics = evaluate_decode_metrics(
                        model=model_for_decode,
                        dataset=val_loader.dataset,
                        tokenizer=smiles_tokenizer,
                        device=device,
                        max_examples=decode_validate_examples,
                        max_len=decode_max_len,
                        ir_as_prompt=bool(cfg["model"].get("ir_as_prompt", False)),
                    )
                    decode_valid = float(epoch_decode_metrics.get("decode_valid_smiles", float("nan")))
                    decode_exact = float(epoch_decode_metrics.get("decode_exact_match_all", float("nan")))
                    decode_tani = float(epoch_decode_metrics.get("decode_avg_tanimoto", float("nan")))
                    decode_ecfp6 = float(epoch_decode_metrics.get("decode_avg_ecfp6_iou", float("nan")))
                    decode_mcs = float(epoch_decode_metrics.get("decode_avg_mcs_over_target", float("nan")))
                    decode_examples = int(epoch_decode_metrics.get("decode_examples", 0))
                    decode_valid_str = f"{decode_valid:.4f}" if np.isfinite(decode_valid) else "nan"
                    decode_exact_str = f"{decode_exact:.4f}" if np.isfinite(decode_exact) else "nan"
                    decode_tani_str = f"{decode_tani:.4f}" if np.isfinite(decode_tani) else "nan"
                    decode_ecfp6_str = f"{decode_ecfp6:.4f}" if np.isfinite(decode_ecfp6) else "nan"
                    decode_mcs_str = f"{decode_mcs:.4f}" if np.isfinite(decode_mcs) else "nan"
                    metric_mode = "rdkit" if bool(epoch_decode_metrics.get("decode_rdkit", False)) else "string"
                    print(
                        f"[epoch {epoch+1}/{num_epochs}] decode_examples={decode_examples} "
                        f"valid_smiles={decode_valid_str} exact_match_all={decode_exact_str} "
                        f"avg_tanimoto={decode_tani_str} avg_ecfp6_iou={decode_ecfp6_str} "
                        f"avg_mcs_over_target={decode_mcs_str} mode={metric_mode}"
                    )
                if dist_ctx.enabled:
                    dist.barrier()

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
            test_loss = evaluate_loss(
                model,
                test_loader,
                pad_token_id,
                nmr_pad_token_id,
                device,
                precision,
                dist_ctx,
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
