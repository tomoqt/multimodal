from typing import Optional, Tuple

import torch
import torch.nn as nn


class UnifiedDiffusionTransformer(nn.Module):
    """
    Paper-aligned conditional mask predictor.

    Prompt tokens and masked target tokens are concatenated into a single
    non-causal sequence and processed by one Transformer stack.
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
        ir_as_prompt: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.max_seq_length = max_seq_length
        self.max_nmr_length = max_nmr_length
        self.max_memory_length = max_memory_length
        self.embed_dim = embed_dim
        self.ir_as_prompt = ir_as_prompt

        self.smiles_embed = nn.Embedding(smiles_vocab_size, embed_dim)
        self.nmr_embed = nn.Embedding(nmr_vocab_size, embed_dim)
        total_capacity = max(max_memory_length + max_nmr_length + max_seq_length + 2, 2048)
        self.sequence_pos_embed = nn.Embedding(total_capacity, embed_dim)
        self.segment_embed = nn.Embedding(2, embed_dim)

        self.memory_proj = nn.Identity()
        if not ir_as_prompt and memory_dim != embed_dim:
            self.memory_proj = nn.Linear(memory_dim, embed_dim)

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

    def _build_prompt(
        self,
        memory: torch.Tensor,
        nmr_tokens: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        nmr_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if memory is None:
            raise ValueError("memory cannot be None")

        if not self.ir_as_prompt:
            memory = self.memory_proj(memory)

        if nmr_tokens is None:
            prompt = memory
        else:
            nmr_tokens = nmr_tokens[:, : self.max_nmr_length]
            prompt = torch.cat([memory, self.nmr_embed(nmr_tokens)], dim=1)

        prompt_padding_mask: Optional[torch.Tensor] = None
        if memory_padding_mask is not None or nmr_padding_mask is not None:
            batch_size = memory.size(0)
            mem_len = memory.size(1)

            if memory_padding_mask is None:
                memory_padding_mask = torch.zeros(batch_size, mem_len, dtype=torch.bool, device=memory.device)
            else:
                memory_padding_mask = memory_padding_mask[:, :mem_len].to(device=memory.device, dtype=torch.bool)

            if nmr_tokens is None:
                prompt_padding_mask = memory_padding_mask
            else:
                nmr_len = nmr_tokens.size(1)
                if nmr_padding_mask is None:
                    nmr_padding_mask = torch.zeros(batch_size, nmr_len, dtype=torch.bool, device=memory.device)
                else:
                    nmr_padding_mask = nmr_padding_mask[:, :nmr_len].to(device=memory.device, dtype=torch.bool)
                prompt_padding_mask = torch.cat([memory_padding_mask, nmr_padding_mask], dim=1)

        return prompt, prompt_padding_mask

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        nmr_tokens: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
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

        prompt, prompt_padding_mask = self._build_prompt(
            memory=memory,
            nmr_tokens=nmr_tokens,
            memory_padding_mask=memory_padding_mask,
            nmr_padding_mask=nmr_padding_mask,
        )
        prompt_len = prompt.size(1)
        total_len = prompt_len + tgt_len
        if total_len > self.sequence_pos_embed.num_embeddings:
            raise ValueError(
                f"Combined prompt+target length {total_len} exceeds configured "
                f"capacity {self.sequence_pos_embed.num_embeddings}."
            )

        tgt_embed = self.smiles_embed(tgt)
        sequence = torch.cat([prompt, tgt_embed], dim=1)
        pos = torch.arange(total_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        sequence = sequence + self.sequence_pos_embed(pos)

        segment_ids = torch.cat(
            [
                torch.zeros(batch_size, prompt_len, dtype=torch.long, device=tgt.device),
                torch.ones(batch_size, tgt_len, dtype=torch.long, device=tgt.device),
            ],
            dim=1,
        )
        sequence = sequence + self.segment_embed(segment_ids)

        combined_padding_mask: Optional[torch.Tensor] = None
        if prompt_padding_mask is not None or tgt_padding_mask is not None:
            if prompt_padding_mask is None:
                prompt_padding_mask = torch.zeros(batch_size, prompt_len, dtype=torch.bool, device=tgt.device)
            if tgt_padding_mask is None:
                tgt_padding_mask = torch.zeros(batch_size, tgt_len, dtype=torch.bool, device=tgt.device)
            combined_padding_mask = torch.cat([prompt_padding_mask, tgt_padding_mask], dim=1)

        hidden = self.encoder(sequence, src_key_padding_mask=combined_padding_mask)
        target_hidden = hidden[:, prompt_len:, :]
        return self.out(self.final_norm(target_hidden))

