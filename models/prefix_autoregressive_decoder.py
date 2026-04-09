from typing import Optional, Tuple

import torch
import torch.nn as nn


class PrefixSMILESDecoder(nn.Module):
    """
    Prefix-conditioned causal transformer over a single sequence:
    [IR prefix][NMR prefix][SMILES target].

    Prompt tokens are fully visible to each other, while target tokens can attend
    to the entire prompt and only their left context.
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
        total_capacity = max(max_memory_length + max_nmr_length + max_seq_length + 4, 2048)
        self.sequence_pos_embed = nn.Embedding(total_capacity, embed_dim)
        self.segment_embed = nn.Embedding(3, embed_dim)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if memory is None:
            raise ValueError("memory cannot be None")

        if not self.ir_as_prompt:
            memory = self.memory_proj(memory)

        pieces = [memory]
        masks = []
        segment_ids = [torch.zeros(memory.size(0), memory.size(1), dtype=torch.long, device=memory.device)]

        if memory_padding_mask is None:
            masks.append(torch.zeros(memory.size(0), memory.size(1), dtype=torch.bool, device=memory.device))
        else:
            masks.append(memory_padding_mask[:, : memory.size(1)].to(device=memory.device, dtype=torch.bool))

        if nmr_tokens is not None:
            nmr_tokens = nmr_tokens[:, : self.max_nmr_length]
            nmr_embed = self.nmr_embed(nmr_tokens)
            pieces.append(nmr_embed)
            segment_ids.append(torch.ones(nmr_embed.size(0), nmr_embed.size(1), dtype=torch.long, device=memory.device))
            if nmr_padding_mask is None:
                masks.append(torch.zeros(nmr_embed.size(0), nmr_embed.size(1), dtype=torch.bool, device=memory.device))
            else:
                masks.append(nmr_padding_mask[:, : nmr_embed.size(1)].to(device=memory.device, dtype=torch.bool))

        prompt = torch.cat(pieces, dim=1)
        prompt_padding_mask = torch.cat(masks, dim=1)
        prompt_segments = torch.cat(segment_ids, dim=1)
        return prompt, prompt_padding_mask, prompt_segments

    def _build_prefix_causal_mask(self, prompt_len: int, tgt_len: int, device: torch.device) -> torch.Tensor:
        total_len = prompt_len + tgt_len
        mask = torch.ones(total_len, total_len, dtype=torch.bool, device=device)

        if prompt_len > 0:
            mask[:prompt_len, :prompt_len] = False
        if tgt_len > 0:
            mask[prompt_len:, :prompt_len] = False
            target_causal = torch.triu(
                torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask[prompt_len:, prompt_len:] = target_causal
        return mask

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        nmr_tokens: torch.Tensor = None,
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
        else:
            tgt_padding_mask = torch.zeros(batch_size, tgt_len, dtype=torch.bool, device=tgt.device)

        prompt, prompt_padding_mask, prompt_segments = self._build_prompt(
            memory=memory,
            nmr_tokens=nmr_tokens,
            memory_padding_mask=memory_padding_mask,
            nmr_padding_mask=nmr_padding_mask,
        )

        tgt_embed = self.smiles_embed(tgt)
        tgt_segments = torch.full((batch_size, tgt_len), 2, dtype=torch.long, device=tgt.device)

        combined = torch.cat([prompt, tgt_embed], dim=1)
        combined_padding_mask = torch.cat([prompt_padding_mask, tgt_padding_mask], dim=1)
        combined_segments = torch.cat([prompt_segments, tgt_segments], dim=1)

        total_len = combined.size(1)
        if total_len > self.sequence_pos_embed.num_embeddings:
            raise ValueError(
                f"Combined prompt+target length {total_len} exceeds configured "
                f"capacity {self.sequence_pos_embed.num_embeddings}."
            )

        pos = torch.arange(total_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        combined = combined + self.sequence_pos_embed(pos) + self.segment_embed(combined_segments)
        attn_mask = self._build_prefix_causal_mask(prompt.size(1), tgt_len, tgt.device)

        hidden = self.encoder(combined, mask=attn_mask, src_key_padding_mask=combined_padding_mask)
        target_hidden = hidden[:, prompt.size(1) :, :]
        return self.out(self.final_norm(target_hidden))
