from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prefix_diffusion_transformer import PrefixConditionedDiffusionTransformer
from .spectral_encoder import MultimodalSpectralEncoder


class MultiModalPrefixDiffusionModel(nn.Module):
    """
    Prefix-conditioned diffusion model.

    IR embeddings and NMR token embeddings are injected directly into the same
    transformer input as the masked SMILES sequence. This removes the separate
    cross-attention memory path used by the older diffusion baseline.
    """

    def __init__(
        self,
        smiles_vocab_size: int,
        nmr_vocab_size: int,
        max_seq_length: int = 512,
        max_nmr_length: int = 128,
        max_memory_length: int = 128,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        verbose: bool = False,
        ir_encoder_type: str = "regular",
        ir_as_prompt: bool = False,
        ir_vocab_size: int = None,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.ir_as_prompt = ir_as_prompt
        self.max_memory_length = max_memory_length
        self.embed_dim = embed_dim

        self.encoder = MultimodalSpectralEncoder(
            embed_dim=embed_dim,
            verbose=verbose,
            encoder_type=ir_encoder_type,
            ir_as_prompt=ir_as_prompt,
        )

        if self.ir_as_prompt:
            if ir_vocab_size is None:
                raise ValueError("ir_vocab_size must be provided when ir_as_prompt is True.")
            self.ir_embed = nn.Embedding(ir_vocab_size, embed_dim)
        else:
            self.ir_embed = None

        self.decoder = PrefixConditionedDiffusionTransformer(
            smiles_vocab_size=smiles_vocab_size,
            nmr_vocab_size=nmr_vocab_size,
            max_seq_length=max_seq_length,
            max_nmr_length=max_nmr_length,
            max_memory_length=max_memory_length,
            memory_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            verbose=verbose,
        )

    def _build_ir_prefix(
        self,
        ir_data: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.ir_as_prompt:
            if ir_data is None:
                return None, None
            if ir_data.dim() != 2:
                raise ValueError(f"Expected tokenized IR prompt with shape (B, L), got {tuple(ir_data.shape)}")
            ir_tokens = ir_data[:, : self.max_memory_length]
            return self.ir_embed(ir_tokens), None

        if ir_data is None:
            return None, None

        ir_prefix = self.encoder(None, ir_data, None)
        if ir_prefix is None:
            return None, None
        if ir_prefix.size(1) > self.max_memory_length:
            ir_prefix = F.adaptive_avg_pool1d(ir_prefix.transpose(1, 2), self.max_memory_length).transpose(1, 2)
        return ir_prefix, None

    def forward(
        self,
        nmr_tokens: torch.Tensor = None,
        ir_data: torch.Tensor = None,
        target_seq: torch.Tensor = None,
        target_padding_mask: torch.Tensor = None,
        nmr_padding_mask: torch.Tensor = None,
        memory_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        del memory_padding_mask
        if target_seq is None:
            raise ValueError("target_seq is required for forward pass.")

        ir_prefix, ir_padding_mask = self._build_ir_prefix(ir_data)
        return self.decoder(
            tgt=target_seq,
            ir_prefix=ir_prefix,
            nmr_tokens=nmr_tokens,
            tgt_padding_mask=target_padding_mask,
            ir_padding_mask=ir_padding_mask,
            nmr_padding_mask=nmr_padding_mask,
        )
