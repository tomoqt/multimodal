import torch
import torch.nn as nn

from .convnext import ConvNeXt1D
from .preprocessing import PerSampleInterpolator, SpectralPreprocessor

class MultimodalSpectralEncoder(nn.Module):
    """
    A simplified encoder that only processes IR data through a ConvNeXt backbone.
    NMR data is handled as tokens in the decoder.
    """
    def __init__(
        self,
        embed_dim=768,
        verbose=True,
    ):
        super().__init__()
        self.verbose = verbose

        base_config = {
            'depths': [3, 3, 6, 3],
            'dims': [64, 128, 256, embed_dim],
            'drop_path_rate': 0.1,
            'layer_scale_init_value': 1e-6,
            'regression': True,
            'regression_dim': embed_dim
        }

        self.ir_backbone = ConvNeXt1D(in_chans=1, **base_config)

    def forward(self, nmr_data, ir_data, c_nmr_data):
        """
        Process only IR data, NMR data is handled by decoder as tokens
        """
        if self.verbose:
            print("\nEncoder Processing:")
            print("Processing IR data...")

        # Process IR through backbone
        if ir_data is not None:
            # Add channel dimension if needed
            if ir_data.dim() == 2:
                ir_data = ir_data.unsqueeze(1)  # [B, 1, L]
            emb_ir = self.ir_backbone(ir_data, keep_sequence=True)
        else:
            emb_ir = None

        if self.verbose and emb_ir is not None:
            print(f"IR embedding: {emb_ir.shape}")

        return emb_ir