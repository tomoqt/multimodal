import torch
import torch.nn as nn

from .spectral_encoder import MultimodalSpectralEncoder
from .transformer_decoder import SMILESDecoder


class MultiModalToSMILESModel(nn.Module):
    """
    Lean end-to-end model:
    - Optional IR encoder (ConvNeXt or regular 1D CNN)
    - Decoder-only transformer over SMILES with IR/NMR prompt memory
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
        use_stablemax: bool = False,
        ir_encoder_type: str = "regular",
        ir_as_prompt: bool = False,
        ir_vocab_size: int = None,
        use_rmsnorm: bool = False,
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

        self.decoder = SMILESDecoder(
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
            use_stablemax=use_stablemax,
            ir_as_prompt=ir_as_prompt,
            use_rmsnorm=use_rmsnorm,
        )

    def _build_memory(
        self,
        nmr_tokens: torch.Tensor = None,
        ir_data: torch.Tensor = None,
        target_seq: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.ir_as_prompt:
            if ir_data is not None:
                if ir_data.dim() != 2:
                    raise ValueError(f"Expected tokenized IR prompt with shape (B, L), got {tuple(ir_data.shape)}")
                return self.ir_embed(ir_data)

            if target_seq is not None:
                batch_size, device = target_seq.size(0), target_seq.device
            elif nmr_tokens is not None:
                batch_size, device = nmr_tokens.size(0), nmr_tokens.device
            else:
                batch_size, device = 1, torch.device("cpu")
            return torch.zeros(batch_size, 1, self.embed_dim, device=device)

        memory = None
        if ir_data is not None:
            memory = self.encoder(None, ir_data, None)

        if memory is not None:
            return memory

        if target_seq is not None:
            batch_size, device = target_seq.size(0), target_seq.device
        elif nmr_tokens is not None:
            batch_size, device = nmr_tokens.size(0), nmr_tokens.device
        else:
            batch_size, device = 1, torch.device("cpu")
        return torch.zeros(batch_size, self.max_memory_length, self.embed_dim, device=device)

    def forward(
        self,
        nmr_tokens: torch.Tensor = None,
        ir_data: torch.Tensor = None,
        target_seq: torch.Tensor = None,
        target_mask: torch.Tensor = None,
        target_padding_mask: torch.Tensor = None,
        nmr_padding_mask: torch.Tensor = None,
        memory_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        del target_mask
        if target_seq is None:
            raise ValueError("target_seq is required for forward pass.")

        memory = self._build_memory(nmr_tokens=nmr_tokens, ir_data=ir_data, target_seq=target_seq)
        return self.decoder(
            tgt=target_seq,
            memory=memory,
            nmr_tokens=nmr_tokens,
            tgt_padding_mask=target_padding_mask,
            nmr_padding_mask=nmr_padding_mask,
            memory_padding_mask=memory_padding_mask,
        )
