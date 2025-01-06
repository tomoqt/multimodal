import torch
import torch.nn as nn
import numpy as np

from .convnext import ConvNeXt1D, ConvNeXt2D
from .preprocessing import GlobalWindowResampler, GlobalWindowResampler2D, Normalizer
from .higher_order_crossattention import HigherOrderMultiInputCrossAttention

class SpectralPreprocessor:
    def __init__(self, resample_size=1000, process_nmr=True, process_ir=True, process_c_nmr=True,
                 nmr_window=None, ir_window=None, c_nmr_window=None):
        self.process_nmr = process_nmr
        self.process_ir = process_ir
        self.process_c_nmr = process_c_nmr
        
        # Use provided windows or defaults
        nmr_window = nmr_window or [0, 12]  # Default NMR window
        ir_window = ir_window or [400, 4000]  # Default IR window
        c_nmr_window = c_nmr_window or [0, 200]  # Default C-NMR window
        
        # NMR resampler with provided window
        self.nmr_processor = GlobalWindowResampler(
            target_size=resample_size,
            window=nmr_window
        ) if process_nmr else None
        
        # IR resampler with provided window
        self.ir_processor = GlobalWindowResampler(
            target_size=resample_size,
            window=ir_window
        ) if process_ir else None
        
        # C-NMR resampler with provided window
        self.c_nmr_processor = GlobalWindowResampler(
            target_size=resample_size,
            window=c_nmr_window
        ) if process_c_nmr else None
        
        self.normalizer = Normalizer()
        self.resample_size = resample_size

    def __call__(self, nmr_data, ir_data, c_nmr_data):
        # Process NMR - expects [batch, length] -> outputs [batch, 1, length]
        if self.process_nmr and nmr_data is not None:
            x_nmr, domain_nmr = nmr_data
            if isinstance(x_nmr, torch.Tensor):
                x_nmr = x_nmr.cpu().numpy()
                domain_nmr = domain_nmr.cpu().numpy() if isinstance(domain_nmr, torch.Tensor) else domain_nmr
                
            # Handle batched data
            processed_nmr = []
            for i in range(len(x_nmr)):
                x_nmr_i = self.nmr_processor(x_nmr[i], domain_nmr)
                x_nmr_i = self.normalizer(x_nmr_i)
                processed_nmr.append(x_nmr_i)
            x_nmr = np.stack(processed_nmr)
            x_nmr = torch.from_numpy(x_nmr).float()
            x_nmr = x_nmr.unsqueeze(1)  # Add channel dimension
        else:
            x_nmr = nmr_data[0] if isinstance(nmr_data, tuple) else nmr_data

        # Process IR - expects [batch, length] -> outputs [batch, 1, length]
        if self.process_ir and ir_data is not None:
            x_ir, domain_ir = ir_data
            if isinstance(x_ir, torch.Tensor):
                x_ir = x_ir.cpu().numpy()
                domain_ir = domain_ir.cpu().numpy() if isinstance(domain_ir, torch.Tensor) else domain_ir
                
            # Handle batched data
            processed_ir = []
            for i in range(len(x_ir)):
                x_ir_i = self.ir_processor(x_ir[i], domain_ir)
                x_ir_i = self.normalizer(x_ir_i)
                processed_ir.append(x_ir_i)
            x_ir = np.stack(processed_ir)
            x_ir = torch.from_numpy(x_ir).float()
            x_ir = x_ir.unsqueeze(1)  # Add channel dimension
        else:
            x_ir = ir_data[0] if isinstance(ir_data, tuple) else ir_data

        # Process C-NMR - expects [batch, length] -> outputs [batch, 1, length]
        if self.process_c_nmr and c_nmr_data is not None:
            x_c_nmr, domain_c_nmr = c_nmr_data
            if isinstance(x_c_nmr, torch.Tensor):
                x_c_nmr = x_c_nmr.cpu().numpy()
                domain_c_nmr = domain_c_nmr.cpu().numpy() if isinstance(domain_c_nmr, torch.Tensor) else domain_c_nmr
                
            # Handle batched data
            processed_c_nmr = []
            for i in range(len(x_c_nmr)):
                x_c_nmr_i = self.c_nmr_processor(x_c_nmr[i], domain_c_nmr)
                x_c_nmr_i = self.normalizer(x_c_nmr_i)
                processed_c_nmr.append(x_c_nmr_i)
            x_c_nmr = np.stack(processed_c_nmr)
            x_c_nmr = torch.from_numpy(x_c_nmr).float()
            x_c_nmr = x_c_nmr.unsqueeze(1)  # Add channel dimension
        else:
            x_c_nmr = c_nmr_data[0] if isinstance(c_nmr_data, tuple) else c_nmr_data

        # Move to device if inputs are on device
        if isinstance(nmr_data[0], torch.Tensor):
            device = nmr_data[0].device
            x_nmr = x_nmr.to(device)
            x_ir = x_ir.to(device)
            x_c_nmr = x_c_nmr.to(device)
        
        return x_nmr, x_ir, x_c_nmr

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