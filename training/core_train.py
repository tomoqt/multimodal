from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_next_token_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Compute autoregressive next-token loss.

    ``logits`` is expected to correspond to ``target_tokens[:, :-1]``.
    """
    expected_steps = target_tokens.size(1) - 1
    if logits.size(1) != expected_steps:
        raise ValueError(
            f"Logit sequence length ({logits.size(1)}) does not match expected "
            f"target length ({expected_steps})."
        )
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_tokens[:, 1:].reshape(-1),
        ignore_index=pad_token_id,
    )


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> float:
    """
    Execute one minimal pretraining step for the core architecture.
    """
    target_tokens, ir_data, nmr_tokens = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)

    target_tokens = target_tokens.to(device)
    nmr_tokens = nmr_tokens.to(device)
    if ir_data is not None:
        ir_data = ir_data.to(device)

    logits = model(
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        target_seq=target_tokens[:, :-1],
    )
    loss = compute_next_token_loss(logits, target_tokens, pad_token_id)
    loss.backward()
    optimizer.step()
    return float(loss.item())
