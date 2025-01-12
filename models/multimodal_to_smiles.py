import torch as th
import torch.nn as nn
from typing import Any

from .spectral_encoder import MultimodalSpectralEncoder
from .transformer_decoder import SMILESDecoder


class MultiModalToSMILESModel(nn.Module):
    """
    A high-level model that:
      1) Encodes IR / H-NMR / C-NMR data via MultimodalSpectralEncoder (concatenation).
      2) Decodes tokens with SMILESDecoder using a prompt-based approach.
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
        verbose: bool = False
    ):
        """
        Args:
            smiles_vocab_size: Number of tokens in the SMILES vocabulary.
            nmr_vocab_size:   Number of tokens in the NMR vocabulary.
            max_seq_length:   Max tokens for decoding.
            max_nmr_length:   Max tokens for NMR decoding.
            max_memory_length: Max tokens for memory/IR embeddings.
            embed_dim:  Hidden dimension for the encoder & (matching) decoder memory.
            num_heads:  Number of attention heads in the decoder.
            num_layers: Number of decoder layers.
            dropout:    Dropout probability in the decoder.
            verbose:    If True, print debugging shapes in forward pass.
        """
        super().__init__()
        self.verbose = verbose

        # The spectral encoder always concatenates IR/H-NMR/C-NMR => final dim = embed_dim
        self.encoder = MultimodalSpectralEncoder(
            embed_dim=embed_dim,
            verbose=verbose
        )

        # The decoder expects memory_dim == encoder's output dim.
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
            verbose=verbose
        )

    def forward(
        self,
        nmr_tokens: th.Tensor | None,
        ir_data: th.Tensor | None,
        target_seq: th.Tensor | None = None,
        target_mask: th.Tensor | None = None
    ):
        """
        Args:
            nmr_tokens: Token IDs for NMR data, shape (B, L).
            ir_data:    IR data, shape (B, L).
            target_seq: Token IDs for SMILES, shape (B, T).
            target_mask: Optional causal mask for the target sequence.

        Returns:
            logits: (B, T, vocab_size), the decoder output for each token.
        """
        if self.verbose:
            print("\n=== Starting Forward Pass ===")

            # Debug prints for spectral data
            def shape_str(x):
                if x is None:
                    return "None"
                elif isinstance(x, th.Tensor):
                    return str(x.shape)
                return "Unknown"

            print(f"NMR Tokens: {shape_str(nmr_tokens)}")
            print(f"IR Data:    {shape_str(ir_data)}")

        # 1) Encode IR inputs -> (B, seq_len, embed_dim)
        memory = self.encoder(None, ir_data, None)

        if self.verbose:
            print("\n=== Starting Decoding ===")
            print(f"Encoder Output (memory) shape: {memory.shape}")

        # 2) Decode to SMILES: target_seq => shape (B, T)
        logits = self.decoder(target_seq, memory, nmr_tokens)

        if self.verbose:
            print("\n=== Forward Pass Complete ===")

        return logits 