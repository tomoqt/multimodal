import torch
import torch.nn as nn

from .spectral_encoder import MultimodalSpectralEncoder
from .transformer_decoder import SMILESDecoder

class MultiModalToSMILESModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_length=512,
        embed_dim=768,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        resample_size=1000,
    ):
        super().__init__()
        
        # Spectral encoder
        self.encoder = MultimodalSpectralEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            resample_size=resample_size
        )
        
        # SMILES decoder
        self.decoder = SMILESDecoder(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, nmr_data, ir_data, hsqc_data, target_seq=None, target_mask=None):
        # Encode spectral inputs
        memory = self.encoder(nmr_data, ir_data, hsqc_data)
        
        # Decode to SMILES
        logits = self.decoder(target_seq, memory, target_mask)
        
        return logits 