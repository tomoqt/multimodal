import torch
import torch.nn as nn

from .convnext import ConvNeXt1D
from .preprocessing import PerSampleInterpolator, SpectralPreprocessor

class MultimodalSpectralEncoder(nn.Module):
    """
    An encoder that:
      - Takes IR, H-NMR, C-NMR data + domain
      - Interpolates each to a uniform size with `SpectralPreprocessor`
      - Feeds them into 1D ConvNeXt backbones
      - Concatenates the outputs
    """
    def __init__(
        self,
        embed_dim=768,
        resample_size=1000,
        verbose=True,
        domain_ranges=None
    ):
        super().__init__()
        self.verbose = verbose

        if embed_dim % 3 != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by 3 (number of modalities) "
                f"to ensure equal dimension distribution across modalities."
            )

        # Unpack domain ranges if provided
        if domain_ranges:
            ir_range, h_nmr_range, c_nmr_range, _, _ = domain_ranges
        else:
            ir_range = [400, 4000]  # Default IR window
            h_nmr_range = [0, 12]   # Default H-NMR window
            c_nmr_range = [0, 200]  # Default C-NMR window

        # Create interpolators for each modality
        ir_interpolator = PerSampleInterpolator(
            target_size=resample_size,
            method='linear',
            do_normalize=True
        )
        hnmr_interpolator = PerSampleInterpolator(
            target_size=resample_size,
            method='linear',
            do_normalize=True
        )
        cnmr_interpolator = PerSampleInterpolator(
            target_size=resample_size,
            method='linear',
            do_normalize=True
        )

        # Create preprocessor
        self.spectral_preprocessor = SpectralPreprocessor(
            ir_interpolator=ir_interpolator,
            hnmr_interpolator=hnmr_interpolator,
            cnmr_interpolator=cnmr_interpolator
        )

        # Each backbone outputs embed_dim//3
        backbone_dim = embed_dim // 3

        base_config = {
            'depths': [3, 3, 6, 3],
            'dims': [64, 128, 256, backbone_dim],
            'drop_path_rate': 0.1,
            'layer_scale_init_value': 1e-6,
            'regression': True,
            'regression_dim': backbone_dim
        }

        self.nmr_backbone = ConvNeXt1D(in_chans=1, **base_config)
        self.ir_backbone = ConvNeXt1D(in_chans=1, **base_config)
        self.c_nmr_backbone = ConvNeXt1D(in_chans=1, **base_config)

    def forward(self, nmr_data, ir_data, c_nmr_data):
        """
        Each input is a tuple (data, domain) where:
            data: shape (B, L) 
            domain: shape (B, L)
        """
        if self.verbose:
            print("\nEncoder Processing:")
            print("Processing input data...")

        # Unpack data and domains
        h_nmr_data, h_nmr_domain = nmr_data if isinstance(nmr_data, tuple) else (nmr_data, None)
        ir_data, ir_domain = ir_data if isinstance(ir_data, tuple) else (ir_data, None)
        c_nmr_data, c_nmr_domain = c_nmr_data if isinstance(c_nmr_data, tuple) else (c_nmr_data, None)

        # Fix: Pass only the data tensors to preprocessor
        x_ir, x_hnmr, x_cnmr = self.spectral_preprocessor(
            ir_data,
            h_nmr_data, 
            c_nmr_data
        )

        if self.verbose:
            print("\nPreprocessed shapes:")
            if x_hnmr is not None:
                print(f"H-NMR: {x_hnmr.shape}")
            if x_ir is not None:
                print(f"IR: {x_ir.shape}")
            if x_cnmr is not None:
                print(f"C-NMR: {x_cnmr.shape}")

        # Forward each through backbone
        emb_nmr = self.nmr_backbone(x_hnmr, keep_sequence=True) if x_hnmr is not None else None
        emb_ir = self.ir_backbone(x_ir, keep_sequence=True) if x_ir is not None else None
        emb_c_nmr = self.c_nmr_backbone(x_cnmr, keep_sequence=True) if x_cnmr is not None else None

        if self.verbose:
            print("\nBackbone outputs:")
            if emb_nmr is not None:
                print(f"NMR embedding: {emb_nmr.shape}")
            if emb_ir is not None:
                print(f"IR embedding: {emb_ir.shape}")
            if emb_c_nmr is not None:
                print(f"C-NMR embedding: {emb_c_nmr.shape}")

        # Concatenate along embedding dimension
        out = torch.cat([emb_nmr, emb_ir, emb_c_nmr], dim=-1)  # [B, seq_len, embed_dim]
        if self.verbose:
            print(f"\nFinal concatenated output: {out.shape}")
        return out