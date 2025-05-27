import torch
import torch.nn as nn

from .convnext import ConvNeXt1D
from .preprocessing import PerSampleInterpolator, SpectralPreprocessor

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.ln = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        # Transpose for LayerNorm: from (B, C, L) to (B, L, C)
        out = out.transpose(1, 2)
        out = self.ln(out)
        out = out.transpose(1, 2)
        out = self.relu(out)
        return out + self.residual(x)

class Regular1DCNNEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_blocks=12, initial_channels=128, num_preconvs=0):
        super().__init__()
        # Optional non-strided convolutional layers before downsampling
        if num_preconvs > 0:
            preconv_layers = []
            preconv_layers.append(nn.Sequential(nn.Conv1d(1, initial_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)))
            for _ in range(num_preconvs - 1):
                preconv_layers.append(nn.Sequential(nn.Conv1d(initial_channels, initial_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)))
            self.pre_convs = nn.Sequential(*preconv_layers)
            conv1_in_channels = initial_channels
        else:
            self.pre_convs = None
            conv1_in_channels = 1

        # Downsampling: first layer downsamples by factor 4, then additional layers with stride 2 (overall factor 32).
        self.conv1 = nn.Conv1d(conv1_in_channels, initial_channels, kernel_size=4, stride=4)  # factor 4
        self.conv2 = nn.Conv1d(initial_channels, initial_channels * 4, kernel_size=3, stride=2, padding=1)  # factor 2
        self.conv3 = nn.Conv1d(initial_channels * 4, initial_channels * 4, kernel_size=3, stride=2, padding=1)  # factor 2
        self.conv4 = nn.Conv1d(initial_channels * 4, embed_dim, kernel_size=3, stride=2, padding=1)  # factor 2
        self.relu = nn.ReLU(inplace=True)
        self.final_ln = nn.LayerNorm(embed_dim)

    def forward(self, x, keep_sequence=True):
        # x: (B, L) or (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.pre_convs is not None:
            x = self.pre_convs(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)  
        if keep_sequence:
            x = x.transpose(1, 2)  # (B, L, embed_dim)
            x = self.final_ln(x)
        else:
            x = x.mean(dim=-1)
            x = self.final_ln(x)
        return x

class MultimodalSpectralEncoder(nn.Module):
    """
    A simplified encoder that only processes IR data using either a convnext or a regular CNN encoder.
    NMR data is handled as tokens in the decoder.
    When 'ir_as_prompt' is True, IR data is used as prompt tokens and not processed by an encoder.
    """
    def __init__(
        self,
        embed_dim=768,
        verbose=True,
        encoder_type="convnext",
        ir_as_prompt: bool = False
    ):
        super().__init__()
        self.verbose = verbose
        self.ir_as_prompt = ir_as_prompt
        self.encoder_type = encoder_type

        # Only create encoder if not using IR as prompt
        if not self.ir_as_prompt:
            if encoder_type == "convnext":
                base_config = {
                    'depths': [3, 3, 6, 3],
                    'dims': [64, 128, 256, embed_dim],
                    'drop_path_rate': 0.1,
                    'layer_scale_init_value': 1e-6,
                    'regression': True,
                    'regression_dim': embed_dim
                }
                try:
                    from .convnext import ConvNeXt1D
                    self.ir_encoder = ConvNeXt1D(in_chans=1, **base_config)
                except ImportError:
                    print("[Warning] ConvNeXt1D not available, falling back to regular encoder")
                    self.ir_encoder = Regular1DCNNEncoder(embed_dim=embed_dim, num_blocks=3, initial_channels=384)
            elif encoder_type == "regular":
                self.ir_encoder = Regular1DCNNEncoder(embed_dim=embed_dim, num_blocks=3, initial_channels=384)
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, nmr_data, ir_data, c_nmr_data):
        """
        Process only IR data. If 'ir_as_prompt' is True, bypass encoding and return None.
        Otherwise, process IR data through the backbone encoder.
        """
        if self.ir_as_prompt:
            if self.verbose:
                print("[Spectral Encoder] IR as prompt enabled; skipping IR encoding.")
            return None

        if self.verbose:
            print("\nEncoder Processing:")
            print(f"Processing IR data with {self.encoder_type} encoder...")

        # Process IR through backbone
        if ir_data is not None:
            # Add channel dimension if needed
            if ir_data.dim() == 2:
                ir_data = ir_data.unsqueeze(1)  # [B, 1, L]
            emb_ir = self.ir_encoder(ir_data, keep_sequence=True)
        else:
            emb_ir = None

        if self.verbose and emb_ir is not None:
            print(f"IR embedding: {emb_ir.shape}")

        return emb_ir 