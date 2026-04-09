from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from training.diffusion_core import build_priority_position_scores, build_smiles_priority_token_ids


def unpack_batch(batch):
    if len(batch) == 4:
        target_tokens, ir_data, nmr_tokens, example_indices = batch
        return target_tokens, ir_data, nmr_tokens, example_indices
    target_tokens, ir_data, nmr_tokens = batch
    return target_tokens, ir_data, nmr_tokens, None


def load_ar_criticality_cache(path: str) -> Dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"AR criticality cache not found: {cache_path}")
    blob = torch.load(cache_path, map_location="cpu")
    if not isinstance(blob, dict) or "scores" not in blob:
        raise ValueError(f"Invalid AR criticality cache payload at {cache_path}")
    scores = blob["scores"]
    if not isinstance(scores, torch.Tensor) or scores.dim() != 2:
        raise ValueError(f"AR criticality cache at {cache_path} must contain a 2D 'scores' tensor.")
    blob["scores"] = scores.to(dtype=torch.float32, device="cpu").contiguous()
    return blob


def resolve_position_scores(
    target_tokens: torch.Tensor,
    mask_schedule: str,
    priority_token_ids: Optional[Sequence[int]] = None,
    ar_criticality_cache: Optional[Dict[str, Any]] = None,
    example_indices: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    schedule = str(mask_schedule).lower()
    if schedule == "uniform":
        return None
    if schedule == "token_priority":
        return build_priority_position_scores(target_tokens, priority_token_ids or [])
    if schedule == "ar_criticality":
        if ar_criticality_cache is None:
            raise ValueError("AR criticality schedule requires a loaded cache.")
        if example_indices is None:
            raise ValueError("AR criticality schedule requires dataset example indices in the batch.")
        scores = ar_criticality_cache["scores"]
        max_index = int(example_indices.max().item()) if example_indices.numel() else -1
        if max_index >= scores.size(0):
            raise IndexError(
                f"Batch example index {max_index} exceeds AR criticality cache size {scores.size(0)}."
            )
        out = scores[example_indices.cpu()]
        if out.shape != target_tokens.shape:
            raise ValueError(
                f"AR criticality scores shape {tuple(out.shape)} does not match targets {tuple(target_tokens.shape)}."
            )
        return out
    raise ValueError(f"Unsupported mask_schedule='{mask_schedule}'.")
