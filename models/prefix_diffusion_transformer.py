from typing import Optional

import torch
import torch.nn as nn


class PrefixConditionedDiffusionTransformer(nn.Module):
    """
    Non-causal mask predictor over a single flat sequence:
    [IR prefix][NMR prefix][masked SMILES].

    This is a stricter multimodal adaptation of the LLaDA-style setup than the
    cross-attention baseline because all modalities are visible inside the same
    transformer stack rather than through a separate memory interface.
    """

    def __init__(
        self,
        smiles_vocab_size: int,
        nmr_vocab_size: int,
        max_seq_length: int = 512,
        max_nmr_length: int = 128,
        max_memory_length: int = 128,
        memory_dim: int = 768,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.max_seq_length = max_seq_length
        self.max_nmr_length = max_nmr_length
        self.max_memory_length = max_memory_length
        self.embed_dim = embed_dim

        self.smiles_embed = nn.Embedding(smiles_vocab_size, embed_dim)
        self.nmr_embed = nn.Embedding(nmr_vocab_size, embed_dim)
        total_capacity = max(max_memory_length + max_nmr_length + max_seq_length + 4, 2048)
        self.sequence_pos_embed = nn.Embedding(total_capacity, embed_dim)
        self.segment_embed = nn.Embedding(3, embed_dim)

        self.ir_proj = nn.Identity()
        if memory_dim != embed_dim:
            self.ir_proj = nn.Linear(memory_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, smiles_vocab_size)

    def _zeros_mask(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    def forward(
        self,
        tgt: torch.Tensor,
        ir_prefix: Optional[torch.Tensor] = None,
        nmr_tokens: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        ir_padding_mask: Optional[torch.Tensor] = None,
        nmr_padding_mask: Optional[torch.Tensor] = None,
        **_: dict,
    ) -> torch.Tensor:
        if tgt is None:
            raise ValueError("tgt cannot be None")

        tgt = tgt[:, : self.max_seq_length]
        batch_size, tgt_len = tgt.shape
        if tgt_len == 0:
            raise ValueError("Target sequence length must be > 0.")

        if tgt_padding_mask is not None:
            tgt_padding_mask = tgt_padding_mask[:, :tgt_len].to(device=tgt.device, dtype=torch.bool)
        else:
            tgt_padding_mask = self._zeros_mask(batch_size, tgt_len, tgt.device)

        pieces = []
        masks = []
        segment_ids = []

        if ir_prefix is not None and ir_prefix.size(1) > 0:
            ir_prefix = ir_prefix[:, : self.max_memory_length]
            ir_len = ir_prefix.size(1)
            ir_prefix = self.ir_proj(ir_prefix)
            pieces.append(ir_prefix)
            if ir_padding_mask is None:
                masks.append(self._zeros_mask(batch_size, ir_len, tgt.device))
            else:
                masks.append(ir_padding_mask[:, :ir_len].to(device=tgt.device, dtype=torch.bool))
            segment_ids.append(torch.zeros(batch_size, ir_len, dtype=torch.long, device=tgt.device))

        if nmr_tokens is not None:
            nmr_tokens = nmr_tokens[:, : self.max_nmr_length]
            nmr_len = nmr_tokens.size(1)
            pieces.append(self.nmr_embed(nmr_tokens))
            if nmr_padding_mask is None:
                masks.append(self._zeros_mask(batch_size, nmr_len, tgt.device))
            else:
                masks.append(nmr_padding_mask[:, :nmr_len].to(device=tgt.device, dtype=torch.bool))
            segment_ids.append(torch.ones(batch_size, nmr_len, dtype=torch.long, device=tgt.device))

        tgt_embed = self.smiles_embed(tgt)
        pieces.append(tgt_embed)
        masks.append(tgt_padding_mask)
        segment_ids.append(torch.full((batch_size, tgt_len), 2, dtype=torch.long, device=tgt.device))

        sequence = torch.cat(pieces, dim=1)
        combined_padding_mask = torch.cat(masks, dim=1)
        combined_segments = torch.cat(segment_ids, dim=1)

        total_len = sequence.size(1)
        if total_len > self.sequence_pos_embed.num_embeddings:
            raise ValueError(
                f"Combined IR+NMR+target length {total_len} exceeds configured "
                f"capacity {self.sequence_pos_embed.num_embeddings}."
            )

        pos = torch.arange(total_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        sequence = sequence + self.sequence_pos_embed(pos) + self.segment_embed(combined_segments)

        hidden = self.encoder(sequence, src_key_padding_mask=combined_padding_mask)
        target_hidden = hidden[:, total_len - tgt_len :, :]
        return self.out(self.final_norm(target_hidden))
