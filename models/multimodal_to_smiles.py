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
        vocab_size: int,
        max_seq_length: int = 512,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        verbose: bool = False
    ):
        """
        Args:
            vocab_size:  Number of tokens in the SMILES vocabulary.
            max_seq_length: Max tokens for decoding.
            embed_dim:  Hidden dimension for the encoder & (matching) decoder memory.
            num_heads:  Number of attention heads in the decoder.
            num_layers: Number of decoder layers.
            dropout:    Dropout probability in the decoder. (The encoder uses a separate config.)
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
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            memory_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            verbose=verbose
        )

    def forward(
        self,
        nmr_data: tuple[th.Tensor, th.Tensor] | th.Tensor | None,
        ir_data: tuple[th.Tensor, th.Tensor] | th.Tensor | None,
        c_nmr_data: tuple[th.Tensor, th.Tensor] | th.Tensor | None,
        target_seq: th.Tensor | None = None,
        target_mask: th.Tensor | None = None
    ):
        """
        Args:
            nmr_data:  Typically a tuple (data, domain) with shape (B, L) each, or None.
            ir_data:   Similarly a tuple (data, domain), or None.
            c_nmr_data: Tuple (data, domain), or None.
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
                elif isinstance(x, (tuple, list)) and isinstance(x[0], th.Tensor):
                    return f"{x[0].shape} + domain {x[1].shape if x[1] is not None else 'None'}"
                elif isinstance(x, th.Tensor):
                    return str(x.shape)
                return "Unknown"

            print(f"NMR Data:   {shape_str(nmr_data)}")
            print(f"IR Data:    {shape_str(ir_data)}")
            print(f"C-NMR Data: {shape_str(c_nmr_data)}")

        # 1) Encode spectral inputs -> (B, seq_len, embed_dim)
        memory = self.encoder(nmr_data, ir_data, c_nmr_data)

        if self.verbose:
            print("\n=== Starting Decoding ===")
            print(f"Encoder Output (memory) shape: {memory.shape}")

        # 2) Decode to SMILES: target_seq => shape (B, T)
        logits = self.decoder(target_seq, memory)

        if self.verbose:
            print("\n=== Forward Pass Complete ===")

        return logits 