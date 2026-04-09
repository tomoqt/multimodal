import re
from typing import Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F


_SMILES_PRIORITY_SYMBOLS = {"(", ")", "=", "#", "-", "+", "\\", "/", ":", "~", "@", ".", "*", "$"}
_SMILES_AROMATIC_ATOMS = {"b", "c", "n", "o", "s", "p"}


def build_smiles_priority_token_ids(tokenizer) -> Set[int]:
    """
    Heuristic chemistry-aware priority tokens for the simple scheduler baseline.
    """
    special_ids = {
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
        getattr(tokenizer, "unk_token_id", None),
    }
    priority_ids: Set[int] = set()
    for token_id, token in tokenizer.ids_to_tokens.items():
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        if token in _SMILES_PRIORITY_SYMBOLS:
            priority_ids.add(token_id)
            continue
        if token in _SMILES_AROMATIC_ATOMS:
            priority_ids.add(token_id)
            continue
        if token.isdigit() or re.fullmatch(r"%\d{2}", token):
            priority_ids.add(token_id)
            continue
        if token.startswith("[") and token.endswith("]"):
            priority_ids.add(token_id)
    return priority_ids


def build_priority_position_scores(target_tokens: torch.Tensor, priority_token_ids: Sequence[int]) -> torch.Tensor:
    if target_tokens.dim() != 2:
        raise ValueError(f"Expected target_tokens with shape (B, T), got {tuple(target_tokens.shape)}")
    if not priority_token_ids:
        return torch.zeros_like(target_tokens, dtype=torch.float32)
    priority_tensor = torch.as_tensor(list(priority_token_ids), device=target_tokens.device, dtype=target_tokens.dtype)
    return torch.isin(target_tokens, priority_tensor).to(dtype=torch.float32)


def apply_condition_dropout(
    nmr_tokens: Optional[torch.Tensor],
    ir_data: Optional[torch.Tensor],
    condition_dropout_prob: float = 0.0,
    nmr_pad_token_id: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool]:
    """
    Batch-level conditioning dropout for classifier-free guidance training.
    """
    if nmr_tokens is None and ir_data is None:
        return None, None, None, True

    nmr_padding_mask = nmr_tokens.eq(nmr_pad_token_id) if nmr_tokens is not None and nmr_pad_token_id is not None else None
    if condition_dropout_prob <= 0.0:
        return nmr_tokens, ir_data, nmr_padding_mask, False

    ref_tensor = nmr_tokens if nmr_tokens is not None else ir_data
    if torch.rand((), device=ref_tensor.device).item() < condition_dropout_prob:
        return None, None, None, True
    return nmr_tokens, ir_data, nmr_padding_mask, False


def _compute_position_mask_probs(
    target_tokens: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    mask_prob_floor: float,
    mask_prob_ceiling: float,
    position_scores: Optional[torch.Tensor],
    priority_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_tokens.dim() != 2:
        raise ValueError(f"Expected target_tokens with shape (B, T), got {tuple(target_tokens.shape)}")
    if mask_prob_ceiling < mask_prob_floor:
        raise ValueError("mask_prob_ceiling must be >= mask_prob_floor.")

    valid_positions = target_tokens.ne(pad_token_id) & target_tokens.ne(bos_token_id)
    batch_size, seq_len = target_tokens.shape
    t = torch.rand(batch_size, device=target_tokens.device)
    sigma = ((mask_prob_ceiling - mask_prob_floor) * t + mask_prob_floor).unsqueeze(1)
    p_mask = sigma.expand(batch_size, seq_len).clone()

    if position_scores is None or priority_weight <= 1.0:
        return p_mask, valid_positions

    if position_scores.shape != target_tokens.shape:
        raise ValueError(
            f"position_scores shape {tuple(position_scores.shape)} does not match targets {tuple(target_tokens.shape)}."
        )

    scores = position_scores.to(device=target_tokens.device, dtype=torch.float32).clamp_min(0.0)
    scores = torch.where(valid_positions, scores, torch.zeros_like(scores))
    valid_count = valid_positions.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=torch.float32)
    row_max = scores.max(dim=1, keepdim=True).values
    normalized = torch.where(row_max > 0.0, scores / row_max.clamp_min(1e-6), scores)
    weights = 1.0 + (priority_weight - 1.0) * normalized
    mean_weight = (weights * valid_positions.to(dtype=weights.dtype)).sum(dim=1, keepdim=True) / valid_count
    p_mask = (p_mask * weights) / mean_weight.clamp_min(1e-6)
    return p_mask.clamp_(0.0, 1.0), valid_positions


def sample_noisy_targets(
    target_tokens: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    mask_prob_floor: float = 1e-3,
    mask_prob_ceiling: float = 1.0,
    position_scores: Optional[torch.Tensor] = None,
    priority_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the LLADA-style forward process to a batch of target token IDs.

    `position_scores` optionally biases corruption toward scheduler-critical positions
    while preserving the row-wise expected masking rate approximately.
    """
    p_mask, valid_positions = _compute_position_mask_probs(
        target_tokens=target_tokens,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        mask_prob_floor=mask_prob_floor,
        mask_prob_ceiling=mask_prob_ceiling,
        position_scores=position_scores,
        priority_weight=priority_weight,
    )

    masked_indices = (torch.rand_like(p_mask) < p_mask) & valid_positions
    for row in range(masked_indices.size(0)):
        if masked_indices[row].any() or not valid_positions[row].any():
            continue
        valid_idx = torch.nonzero(valid_positions[row], as_tuple=False).squeeze(-1)
        pick = valid_idx[torch.randint(valid_idx.numel(), (1,), device=target_tokens.device)]
        masked_indices[row, pick] = True

    noisy_tokens = torch.where(
        masked_indices,
        torch.full_like(target_tokens, mask_token_id),
        target_tokens,
    )
    return noisy_tokens, masked_indices, p_mask, valid_positions


def iter_generated_blocks(max_seq_length: int, block_size: int):
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    start = 1
    while start < max_seq_length:
        end = min(start + block_size, max_seq_length)
        yield start, end
        start = end


def prepare_block_diffusion_targets(
    target_tokens: torch.Tensor,
    block_start: int,
    block_end: int,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    mask_prob_floor: float = 1e-3,
    mask_prob_ceiling: float = 1.0,
    position_scores: Optional[torch.Tensor] = None,
    priority_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (0 <= block_start < block_end <= target_tokens.size(1)):
        raise ValueError(
            f"Invalid block range [{block_start}, {block_end}) for target shape {tuple(target_tokens.shape)}."
        )

    current_block = target_tokens[:, block_start:block_end]
    current_scores = None if position_scores is None else position_scores[:, block_start:block_end]
    noisy_block, masked_block, p_mask_block, valid_block = sample_noisy_targets(
        target_tokens=current_block,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        mask_prob_floor=mask_prob_floor,
        mask_prob_ceiling=mask_prob_ceiling,
        position_scores=current_scores,
        priority_weight=priority_weight,
    )

    noisy_tokens = target_tokens.clone()
    noisy_tokens[:, block_start:block_end] = noisy_block
    if block_end < noisy_tokens.size(1):
        noisy_tokens[:, block_end:] = pad_token_id

    masked_indices = torch.zeros_like(target_tokens, dtype=torch.bool)
    masked_indices[:, block_start:block_end] = masked_block

    p_mask = torch.ones_like(target_tokens, dtype=torch.float32)
    p_mask[:, block_start:block_end] = p_mask_block

    valid_positions = torch.zeros_like(target_tokens, dtype=torch.bool)
    valid_positions[:, block_start:block_end] = valid_block
    return noisy_tokens, masked_indices, p_mask, valid_positions


def prepare_diffusion_targets(
    target_tokens: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    mask_prob_floor: float = 1e-3,
    mask_prob_ceiling: float = 1.0,
    position_scores: Optional[torch.Tensor] = None,
    priority_weight: float = 1.0,
    complementary_masking: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy_tokens, masked_indices, p_mask, valid_positions = sample_noisy_targets(
        target_tokens=target_tokens,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        mask_prob_floor=mask_prob_floor,
        mask_prob_ceiling=mask_prob_ceiling,
        position_scores=position_scores,
        priority_weight=priority_weight,
    )
    if not complementary_masking:
        return noisy_tokens, masked_indices, p_mask, valid_positions

    complementary_mask = valid_positions & ~masked_indices
    complementary_noisy_tokens = torch.where(
        complementary_mask,
        torch.full_like(target_tokens, mask_token_id),
        target_tokens,
    )
    complementary_p_mask = (1.0 - p_mask).clamp_min(1e-6)
    return (
        torch.cat([noisy_tokens, complementary_noisy_tokens], dim=0),
        torch.cat([masked_indices, complementary_mask], dim=0),
        torch.cat([p_mask, complementary_p_mask], dim=0),
        torch.cat([valid_positions, valid_positions], dim=0),
    )


def compute_masked_diffusion_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    masked_indices: torch.Tensor,
    p_mask: torch.Tensor,
    valid_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.shape[:2] != target_tokens.shape:
        raise ValueError(
            f"Logits shape {tuple(logits.shape)} is incompatible with targets shape {tuple(target_tokens.shape)}."
        )

    if not masked_indices.any():
        return logits.sum() * 0.0

    token_loss = F.cross_entropy(
        logits[masked_indices],
        target_tokens[masked_indices],
        reduction="none",
    )
    weighted_loss = token_loss / p_mask[masked_indices].clamp_min(1e-6)
    if valid_positions is None:
        normalizer = torch.tensor(float(target_tokens.numel()), device=logits.device, dtype=logits.dtype)
    else:
        normalizer = valid_positions.sum().clamp_min(1).to(device=logits.device, dtype=logits.dtype)
    return weighted_loss.sum() / normalizer


def diffusion_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    device: torch.device,
    mask_prob_floor: float = 1e-3,
    mask_prob_ceiling: float = 1.0,
    nmr_pad_token_id: Optional[int] = None,
    condition_dropout_prob: float = 0.0,
    position_scores: Optional[torch.Tensor] = None,
    priority_weight: float = 1.0,
    complementary_masking: bool = False,
) -> float:
    """
    Execute one diffusion pretraining step for the mask-predictor baseline.
    """
    if len(batch) == 4:
        target_tokens, ir_data, nmr_tokens, _ = batch
    else:
        target_tokens, ir_data, nmr_tokens = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)

    target_tokens = target_tokens.to(device)
    nmr_tokens = nmr_tokens.to(device)
    if ir_data is not None:
        ir_data = ir_data.to(device)
    if position_scores is not None:
        position_scores = position_scores.to(device)

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
        if nmr_tokens is not None:
            nmr_tokens = torch.cat([nmr_tokens, nmr_tokens], dim=0)
        if ir_data is not None:
            ir_data = torch.cat([ir_data, ir_data], dim=0)

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
    loss.backward()
    optimizer.step()
    return float(loss.item())


def block_diffusion_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    mask_token_id: int,
    pad_token_id: int,
    bos_token_id: int,
    device: torch.device,
    block_size: int,
    mask_prob_floor: float = 1e-3,
    mask_prob_ceiling: float = 1.0,
    nmr_pad_token_id: Optional[int] = None,
    condition_dropout_prob: float = 0.0,
    position_scores: Optional[torch.Tensor] = None,
    priority_weight: float = 1.0,
) -> float:
    if len(batch) == 4:
        target_tokens, ir_data, nmr_tokens, _ = batch
    else:
        target_tokens, ir_data, nmr_tokens = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)

    target_tokens = target_tokens.to(device)
    nmr_tokens = nmr_tokens.to(device)
    if ir_data is not None:
        ir_data = ir_data.to(device)
    if position_scores is not None:
        position_scores = position_scores.to(device)

    cond_nmr_tokens, cond_ir_data, cond_nmr_padding_mask, _ = apply_condition_dropout(
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        condition_dropout_prob=condition_dropout_prob,
        nmr_pad_token_id=nmr_pad_token_id,
    )

    total_loss = None
    active_blocks = 0
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
        total_loss = block_loss if total_loss is None else (total_loss + block_loss)
        active_blocks += 1

    if total_loss is None or active_blocks == 0:
        loss = torch.zeros((), device=device, requires_grad=True)
    else:
        loss = total_loss / active_blocks
    loss.backward()
    optimizer.step()
    return float(loss.item())
