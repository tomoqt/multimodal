import torch
import torch.nn as nn
from typing import Optional, Tuple


class SMILESDecoder(nn.Module):
    """
    Lean decoder-only transformer with optional IR/NMR prompt memory.
    Prompt memory is built by concatenating encoded IR memory and embedded NMR tokens.
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
        use_stablemax: bool = False,
        ir_as_prompt: bool = False,
        use_rmsnorm: bool = False,
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
        self.target_pos_embed = nn.Embedding(max_seq_length, embed_dim)
        prompt_capacity = max(max_memory_length + max_nmr_length + 2, 2048)
        self.prompt_pos_embed = nn.Embedding(prompt_capacity, embed_dim)

        self.memory_proj = nn.Identity()
        if not ir_as_prompt and memory_dim != embed_dim:
            self.memory_proj = nn.Linear(memory_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, smiles_vocab_size)

    def _build_prompt(
        self,
        memory: torch.Tensor,
        nmr_tokens: torch.Tensor = None,
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
            nmr_embed = self.nmr_embed(nmr_tokens)
            prompt = torch.cat([memory, nmr_embed], dim=1)

        prompt_padding_mask: Optional[torch.Tensor] = None
        if memory_padding_mask is not None or nmr_padding_mask is not None:
            batch_size = memory.size(0)
            mem_len = memory.size(1)

            if memory_padding_mask is None:
                memory_padding_mask = torch.zeros(
                    batch_size,
                    mem_len,
                    dtype=torch.bool,
                    device=memory.device,
                )
            else:
                memory_padding_mask = memory_padding_mask[:, :mem_len].to(device=memory.device, dtype=torch.bool)

            if nmr_tokens is None:
                prompt_padding_mask = memory_padding_mask
            else:
                nmr_len = nmr_tokens.size(1)
                if nmr_padding_mask is None:
                    nmr_padding_mask = torch.zeros(
                        batch_size,
                        nmr_len,
                        dtype=torch.bool,
                        device=memory.device,
                    )
                else:
                    nmr_padding_mask = nmr_padding_mask[:, :nmr_len].to(device=memory.device, dtype=torch.bool)
                prompt_padding_mask = torch.cat([memory_padding_mask, nmr_padding_mask], dim=1)

        prompt_len = prompt.size(1)
        if prompt_len > self.prompt_pos_embed.num_embeddings:
            raise ValueError(
                f"Prompt length {prompt_len} exceeds configured prompt positional "
                f"capacity {self.prompt_pos_embed.num_embeddings}."
            )
        prompt_pos = torch.arange(prompt_len, device=prompt.device).unsqueeze(0)
        return prompt + self.prompt_pos_embed(prompt_pos), prompt_padding_mask

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
        """
        Args:
            tgt: Target token IDs, shape (B, T)
            memory: Encoded IR prompt memory, shape (B, M, D)
            nmr_tokens: Optional NMR token IDs, shape (B, N)
        """
        if tgt is None:
            raise ValueError("tgt cannot be None")

        tgt = tgt[:, : self.max_seq_length]
        batch_size, tgt_len = tgt.shape
        if tgt_len == 0:
            raise ValueError("Target sequence length must be > 0.")

        if tgt_padding_mask is not None:
            tgt_padding_mask = tgt_padding_mask[:, :tgt_len].to(device=tgt.device, dtype=torch.bool)

        tgt_positions = torch.arange(tgt_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        tgt_embed = self.smiles_embed(tgt) + self.target_pos_embed(tgt_positions)
        prompt, prompt_padding_mask = self._build_prompt(
            memory=memory,
            nmr_tokens=nmr_tokens,
            memory_padding_mask=memory_padding_mask,
            nmr_padding_mask=nmr_padding_mask,
        )

        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.decoder(
            tgt=tgt_embed,
            memory=prompt,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=prompt_padding_mask,
        )
        logits = self.out(self.final_norm(hidden))
        return logits
