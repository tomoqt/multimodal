import torch
import torch.nn as nn
import numpy as np

from .convnext import ConvNeXt1D, ConvNeXt2D
from .preprocessing import GlobalWindowResampler, GlobalWindowResampler2D, Normalizer
from .higher_order_crossattention import HigherOrderMultiInputCrossAttention

class SpectralPreprocessor:
    def __init__(self, resample_size=1000):
        # NMR window typically 0-12 ppm
        self.nmr_processor = GlobalWindowResampler(
            target_size=resample_size,
            window=[0, 12]
        )
        
        # IR window typically 400-4000 cm^-1
        self.ir_processor = GlobalWindowResampler(
            target_size=resample_size,
            window=[400, 4000]
        )
        
        # HSQC needs 2D resampling with appropriate windows
        self.hsqc_processor = GlobalWindowResampler2D(
            target_size=(resample_size, resample_size),
            window_h=[0, 12],    # Proton dimension typically 0-12 ppm
            window_c=[0, 200],   # Carbon dimension typically 0-200 ppm
            method='linear'      # Use linear interpolation for stability
        )
        
        self.normalizer = Normalizer()
        self.resample_size = resample_size

    def __call__(self, nmr_data, ir_data, hsqc_data):
        device = None
        
        # First determine the device from any valid input tensor
        if nmr_data is not None and isinstance(nmr_data, (tuple, list)) and len(nmr_data) == 2:
            x_nmr = nmr_data[0]
            if isinstance(x_nmr, torch.Tensor):
                device = x_nmr.device
        if device is None and ir_data is not None and isinstance(ir_data, (tuple, list)) and len(ir_data) == 2:
            x_ir = ir_data[0]
            if isinstance(x_ir, torch.Tensor):
                device = x_ir.device
        if device is None and hsqc_data is not None and isinstance(hsqc_data, (tuple, list)) and len(hsqc_data) == 3:
            x_hsqc = hsqc_data[0]
            if isinstance(x_hsqc, torch.Tensor):
                device = x_hsqc.device
        
        # Process NMR
        if nmr_data is not None and isinstance(nmr_data, (tuple, list)) and len(nmr_data) == 2:
            x_nmr, domain_nmr = nmr_data
            if isinstance(x_nmr, torch.Tensor):
                x_nmr = x_nmr.cpu().numpy()
                domain_nmr = domain_nmr.cpu().numpy() if isinstance(domain_nmr, torch.Tensor) else domain_nmr
            x_nmr = self.nmr_processor(x_nmr, domain_nmr)
            x_nmr = self.normalizer(x_nmr)
            x_nmr = torch.from_numpy(x_nmr).float().unsqueeze(0)  # Add channel dimension
        else:
            x_nmr = torch.zeros(1, self.resample_size)

        # Process IR
        if ir_data is not None and isinstance(ir_data, (tuple, list)) and len(ir_data) == 2:
            x_ir, domain_ir = ir_data
            if isinstance(x_ir, torch.Tensor):
                x_ir = x_ir.cpu().numpy()
                domain_ir = domain_ir.cpu().numpy() if isinstance(domain_ir, torch.Tensor) else domain_ir
            x_ir = self.ir_processor(x_ir, domain_ir)
            x_ir = self.normalizer(x_ir)
            x_ir = torch.from_numpy(x_ir).float().unsqueeze(0)
        else:
            x_ir = torch.zeros(1, self.resample_size)

        # Process HSQC (now as 2D)
        if hsqc_data is not None and isinstance(hsqc_data, (tuple, list)) and len(hsqc_data) == 3:
            x_hsqc, domain_h, domain_c = hsqc_data
            if isinstance(x_hsqc, torch.Tensor):
                x_hsqc = x_hsqc.cpu().numpy()
                domain_h = domain_h.cpu().numpy() if isinstance(domain_h, torch.Tensor) else domain_h
                domain_c = domain_c.cpu().numpy() if isinstance(domain_c, torch.Tensor) else domain_c
            x_hsqc = self.hsqc_processor(x_hsqc, domain_h, domain_c)
            x_hsqc = self.normalizer(x_hsqc)
            x_hsqc = torch.from_numpy(x_hsqc).float().unsqueeze(0)  # Add channel dimension
        else:
            x_hsqc = torch.zeros(1, self.resample_size, self.resample_size)

        # Move to device
        if device is not None:
            x_nmr = x_nmr.to(device)
            x_ir = x_ir.to(device)
            x_hsqc = x_hsqc.to(device)
        
        return x_nmr, x_ir, x_hsqc

class MultimodalSpectralEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.1, resample_size=1000):
        super().__init__()
        self.preprocessor = SpectralPreprocessor(resample_size=resample_size)
        
        # Base ConvNeXt config
        base_config = {
            'depths': [3, 3, 6, 3],
            'dims': [64, 128, 256, embed_dim],
            'drop_path_rate': 0.1,
            'layer_scale_init_value': 1e-6,
            'regression': True,
            'regression_dim': embed_dim
        }
        
        # Create 1D backbones for NMR and IR
        self.nmr_backbone = ConvNeXt1D(in_chans=1, **base_config)
        self.ir_backbone = ConvNeXt1D(in_chans=1, **base_config)
        
        # Create 2D backbone for HSQC
        self.hsqc_backbone = ConvNeXt2D(in_chans=1, **base_config)
        
        # Ensure all backbones are on the same device as the parent module
        device = next(self.parameters()).device
        self.nmr_backbone = self.nmr_backbone.to(device)
        self.ir_backbone = self.ir_backbone.to(device)
        self.hsqc_backbone = self.hsqc_backbone.to(device)
        
        # Add higher-order cross attention
        cross_attn_config = type('Config', (), {
            'n_head': num_heads,
            'n_embd': embed_dim,
            'order': 3,  # We have 3 modalities: NMR, IR, HSQC
            'dropout': dropout,
            'bias': True
        })
        self.cross_attention = HigherOrderMultiInputCrossAttention(cross_attn_config)
        
        # Add final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, nmr_data, ir_data, hsqc_data):
        # Get the device of the model
        device = next(self.parameters()).device
        
        # Preprocess the input data
        x_nmr, x_ir, x_hsqc = self.preprocessor(nmr_data, ir_data, hsqc_data)
        
        # Ensure inputs are on the correct device
        x_nmr = x_nmr.to(device)
        x_ir = x_ir.to(device)
        x_hsqc = x_hsqc.to(device)
        
        # Pass through backbones
        emb_nmr = self.nmr_backbone(x_nmr)
        emb_ir = self.ir_backbone(x_ir)
        emb_hsqc = self.hsqc_backbone(x_hsqc)
        
        # Ensure embeddings have sequence dimension if they don't already
        if emb_nmr.dim() == 2:
            emb_nmr = emb_nmr.unsqueeze(1)
            emb_ir = emb_ir.unsqueeze(1)
            emb_hsqc = emb_hsqc.unsqueeze(1)
            
        # Apply higher-order cross attention
        # The first input will determine the output sequence length
        fused = self.cross_attention(emb_nmr, emb_ir, emb_hsqc)
        
        # Apply final normalization
        fused = self.final_norm(fused)
        
        # Average over sequence dimension if present
        if fused.dim() == 3:
            fused = fused.mean(dim=1)
            
        return fused