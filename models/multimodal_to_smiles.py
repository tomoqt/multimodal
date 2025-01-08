import torch as th
import torch.nn as nn
from typing import Any

from .spectral_encoder import MultimodalSpectralEncoder
from .transformer_decoder import SMILESDecoder

class MultiModalToSMILESModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 512,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        resample_size: int = 1000,
        use_concat: bool = True,
        verbose: bool = False,
        domain_ranges: list | None = None,
        cls_dim: int = 512
    ):
        super().__init__()
        
        self.use_concat = use_concat
        self.verbose = verbose
        self.cls_dim = cls_dim
        
        # Spectral encoder with verbose off
        memory_dim = 2046 if use_concat else embed_dim
        self.encoder = MultimodalSpectralEncoder(
            # encode to 2046 but later downproject memory in transformer
            embed_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            resample_size=resample_size,
            use_concat=use_concat,
            verbose=verbose,
            domain_ranges=domain_ranges
        )
        
        # Calculate decoder input dimension
        decoder_dim = embed_dim * 1 if use_concat else embed_dim
        
        # SMILES decoder with verbose off
        self.decoder = SMILESDecoder(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            memory_dim=memory_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            verbose=verbose
        )

        # extra projections for cls
        self.to_cls_enc = th.nn.Linear(memory_dim, cls_dim)
        self.to_cls_dec = th.nn.Linear(embed_dim, cls_dim)

    def forward(self, nmr_data: tuple | th.Tensor | None, ir_data: tuple | th.Tensor | None, c_nmr_data: tuple | th.Tensor | None, target_seq: Any | None = None, target_mask: th.Tensor | None = None):
        """ Returns the cls (dense encoding for molecule) and logits to sample smiles from
        * (cls_enc, cls_dec) = 2x (B, D)
        * logits: (B, L, D)
        """
        if self.verbose:
            print("\n=== Starting Forward Pass ===")
            print("\nSpectroscopic data shapes inside forward:")
            if nmr_data is not None:
                if isinstance(nmr_data, tuple):
                    print(f"NMR data: {nmr_data[0].shape}")
                else:
                    print(f"NMR: {nmr_data.shape}")
            if ir_data is not None:
                if isinstance(ir_data, tuple):
                    print(f"IR data: {ir_data[0].shape}")
                else:
                    print(f"IR: {ir_data.shape}")
            if c_nmr_data is not None:
                if isinstance(c_nmr_data, tuple):
                    print(f"C-NMR data: {c_nmr_data[0].shape}")
                else:
                    print(f"C-NMR: {c_nmr_data.shape}")
            
        # Encode spectral inputs
        memory = self.encoder(nmr_data, ir_data, c_nmr_data)
        
        if self.verbose:
            print("\n=== Starting Decoding ===")

        # Decode to SMILES
        cls, logits = self.decoder(target_seq, memory, target_mask)
        
        if self.verbose:
            print("\n=== Forward Pass Complete ===")

        # [CLS] by mean pooling: (B, D)
        cls_enc = self.to_cls_enc(memory).mean(dim=-2)
        cls_dec = self.to_cls_dec(cls)
            
        return (cls_enc, cls_dec), logits