import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
import json
import os

from .convnext import ConvNeXt1D, ConvNeXt2D
from .higher_order_crossattention import HigherOrderMultiInputCrossAttention

class SpectralPreprocessor:
    def __init__(self, resample_size=1000, process_nmr=True, process_ir=False, process_c_nmr=True,
                 nmr_window=None, ir_window=None, c_nmr_window=None):
        self.process_nmr = process_nmr
        self.process_ir = process_ir
        self.process_c_nmr = process_c_nmr
        
        # Load spectrum dimensions from JSON
        dimensions_path = os.path.join('data_extraction', 'multimodal_spectroscopic_dataset', 
                                     'meta_data', 'spectrum_dimensions.json')
        with open(dimensions_path, 'r') as f:
            spectrum_dims = json.load(f)
            
        # Store the domains
        self.nmr_domain = np.array(spectrum_dims['h_nmr_spectra']['dimensions'])
        self.ir_domain = np.array(spectrum_dims['ir_spectra']['dimensions'])
        self.c_nmr_domain = np.array(spectrum_dims['c_nmr_spectra']['dimensions'])
        
        # Use provided windows or defaults based on domain ranges
        self.nmr_window = nmr_window or [min(self.nmr_domain), max(self.nmr_domain)]
        self.ir_window = ir_window or [min(self.ir_domain), max(self.ir_domain)]
        self.c_nmr_window = c_nmr_window or [min(self.c_nmr_domain), max(self.c_nmr_domain)]

    def __call__(self, nmr_data, ir_data, c_nmr_data):
        # Just convert to tensor and add channel dimension if needed
        if self.process_nmr and nmr_data is not None:
            if not isinstance(nmr_data, torch.Tensor):
                nmr_data = torch.from_numpy(nmr_data).float()
            if nmr_data.dim() == 2:
                nmr_data = nmr_data.unsqueeze(1)

        if self.process_ir and ir_data is not None:
            # Handle case where ir_data might be a tuple or other type
            if isinstance(ir_data, (tuple, list)):
                # If it's a sequence type, take the first element
                ir_data = ir_data[0]
            
            if not isinstance(ir_data, torch.Tensor):
                # Convert to numpy if it isn't already
                if not isinstance(ir_data, np.ndarray):
                    ir_data = np.array(ir_data)
                ir_data = torch.from_numpy(ir_data).float()
            
            if ir_data.dim() == 2:
                ir_data = ir_data.unsqueeze(1)

        if self.process_c_nmr and c_nmr_data is not None:
            if not isinstance(c_nmr_data, torch.Tensor):
                c_nmr_data = torch.from_numpy(c_nmr_data).float()
            if c_nmr_data.dim() == 2:
                c_nmr_data = c_nmr_data.unsqueeze(1)

        return nmr_data, ir_data, c_nmr_data

class MultimodalSpectralEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1, resample_size=1000, 
                 use_concat=True, verbose=True, domain_ranges=None,
                 use_nmr=True, use_ir=True, use_c_nmr=True):  # Add flags for each modality
        super().__init__()
        
        self.use_nmr = use_nmr
        self.use_ir = use_ir
        self.use_c_nmr = use_c_nmr
        
        # Count active modalities
        active_modalities = sum([use_nmr, use_ir, use_c_nmr])
        if active_modalities == 0:
            raise ValueError("At least one modality must be enabled")
            
        # Only check divisibility if using concatenation and multiple modalities
        if use_concat and active_modalities > 1 and embed_dim % active_modalities != 0:
            raise ValueError(
                f"When using concatenation (use_concat=True), embed_dim ({embed_dim}) "
                f"must be divisible by number of active modalities ({active_modalities}) "
                f"to ensure equal dimension distribution."
            )
            
        self.verbose = verbose
        
        # Unpack domain ranges if provided
        if domain_ranges:
            ir_range, h_nmr_range, c_nmr_range, _, _ = domain_ranges
        else:
            ir_range = None
            h_nmr_range = None
            c_nmr_range = None
        
        self.preprocessor = SpectralPreprocessor(
            resample_size=resample_size,
            process_nmr=use_nmr,
            process_ir=use_ir,
            process_c_nmr=use_c_nmr,
            nmr_window=h_nmr_range,
            ir_window=ir_range,
            c_nmr_window=c_nmr_range
        )
        
        # Calculate individual backbone output dimensions
        backbone_dim = embed_dim // active_modalities if (use_concat and active_modalities > 1) else embed_dim
        
        # Calculate the final sequence length after all downsampling
        final_seq_len = resample_size // 32
        
        # Base ConvNeXt config with appropriate final dimension
        base_config = {
            'depths': [3, 3, 6, 3],
            'dims': [64, 128, 256, backbone_dim],
            'drop_path_rate': 0.1,
            'layer_scale_init_value': 1e-6,
            'regression': True,
            'regression_dim': backbone_dim
        }
        
        if verbose:
            print(f"Active modalities: {active_modalities}")
            print(f"Input sequence length: {resample_size}")
            print(f"Final sequence length: {final_seq_len}")
            print(f"Backbone output dimension: {backbone_dim}")
            print(f"Using concatenation: {use_concat}")
        
        # Create backbones only for active modalities
        self.nmr_backbone = ConvNeXt1D(in_chans=1, **base_config) if use_nmr else None
        self.ir_backbone = ConvNeXt1D(in_chans=1, **base_config) if use_ir else None
        self.c_nmr_backbone = ConvNeXt1D(in_chans=1, **base_config) if use_c_nmr else None
        
        # Move active backbones to correct device
        device = next(self.parameters()).device
        if self.nmr_backbone:
            self.nmr_backbone = self.nmr_backbone.to(device)
        if self.ir_backbone:
            self.ir_backbone = self.ir_backbone.to(device)
        if self.c_nmr_backbone:
            self.c_nmr_backbone = self.c_nmr_backbone.to(device)
        
        self.use_concat = use_concat
        
        # Only create cross attention components if not using concatenation and multiple modalities
        if not use_concat and active_modalities > 1:
            cross_attn_config = type('Config', (), {
                'n_head': num_heads,
                'n_embd': embed_dim,
                'order': active_modalities,  # Adjust order based on active modalities
                'dropout': dropout,
                'bias': True
            })
            self.cross_attention = HigherOrderMultiInputCrossAttention(cross_attn_config)
            self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, nmr_data, ir_data, c_nmr_data):
        if self.verbose:
            print("\nEncoder Processing:")
            print("Processing input data...")
        
        # Preprocess the input data
        x_nmr, x_ir, x_c_nmr = self.preprocessor(nmr_data, ir_data, c_nmr_data)
        
        # Process active modalities
        embeddings = []
        
        if self.use_nmr and x_nmr is not None:
            if x_nmr.dim() == 2:
                x_nmr = x_nmr.unsqueeze(1)
            elif x_nmr.dim() == 3 and x_nmr.size(1) > x_nmr.size(2):
                x_nmr = x_nmr.transpose(1, 2)
            emb_nmr = self.nmr_backbone(x_nmr, keep_sequence=True)
            embeddings.append(emb_nmr)
            
        if self.use_ir and x_ir is not None:
            if x_ir.dim() == 2:
                x_ir = x_ir.unsqueeze(1)
            elif x_ir.dim() == 3 and x_ir.size(1) > x_ir.size(2):
                x_ir = x_ir.transpose(1, 2)
            emb_ir = self.ir_backbone(x_ir, keep_sequence=True)
            embeddings.append(emb_ir)
            
        if self.use_c_nmr and x_c_nmr is not None:
            if x_c_nmr.dim() == 2:
                x_c_nmr = x_c_nmr.unsqueeze(1)
            elif x_c_nmr.dim() == 3 and x_c_nmr.size(1) > x_c_nmr.size(2):
                x_c_nmr = x_c_nmr.transpose(1, 2)
            emb_c_nmr = self.c_nmr_backbone(x_c_nmr, keep_sequence=True)
            embeddings.append(emb_c_nmr)
            
        if self.verbose:
            print(f"\nBackbone outputs:")
            for i, emb in enumerate(embeddings):
                print(f"Modality {i} embedding: {emb.shape}")
            
        # Handle single modality case
        if len(embeddings) == 1:
            return embeddings[0]
            
        # Handle multiple modalities
        if self.use_concat:
            # Verify all sequences have same length
            seq_len = embeddings[0].size(1)
            assert all(emb.size(1) == seq_len for emb in embeddings), "Sequence lengths must match"
            
            # Concatenate along embedding dimension
            result = torch.cat(embeddings, dim=-1)
            
            if self.verbose:
                print(f"\nFinal concatenated output: {result.shape}")
            return result
        else:
            # Apply higher-order cross attention
            fused = self.cross_attention(*embeddings)
            fused = self.final_norm(fused)
            
            if self.verbose:
                print(f"\nFinal fused output: {fused.shape}")
            return fused