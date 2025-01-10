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
            # Pass through raw data when not processing
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
            # Pass through raw data when not processing
            x_c_nmr = c_nmr_data[0] if isinstance(c_nmr_data, tuple) else c_nmr_data

        # Move to device if inputs are on device
        device = None
        if ir_data is not None and isinstance(ir_data, tuple) and isinstance(ir_data[0], torch.Tensor):
            device = ir_data[0].device
        
        if device is not None:
            if x_ir is not None and isinstance(x_ir, torch.Tensor):
                x_ir = x_ir.to(device)
            if x_nmr is not None and isinstance(x_nmr, torch.Tensor):
                x_nmr = x_nmr.to(device)
            if x_c_nmr is not None and isinstance(x_c_nmr, torch.Tensor):
                x_c_nmr = x_c_nmr.to(device)
        
        return x_nmr, x_ir, x_c_nmr

class MultimodalSpectralEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1, resample_size=1000, 
                 use_concat=True, verbose=True, domain_ranges=None, use_mlp_for_nmr=True):
        super().__init__()
        
        self.verbose = verbose
        self.use_concat = use_concat
        self.use_mlp_for_nmr = use_mlp_for_nmr
        
        # Only check divisibility by 3 if using concatenation and not using MLP
        if use_concat and not use_mlp_for_nmr and embed_dim % 3 != 0:
            raise ValueError(
                f"When using concatenation (use_concat=True), embed_dim ({embed_dim}) "
                f"must be divisible by 3 (number of modalities) to ensure equal "
                f"dimension distribution across modalities."
            )
            
        # Unpack domain ranges if provided
        if domain_ranges:
            ir_range, h_nmr_range, c_nmr_range, _, _ = domain_ranges
        else:
            ir_range = None
            h_nmr_range = None
            c_nmr_range = None
        
        # Create preprocessor only for IR if using MLP for NMR
        self.preprocessor = SpectralPreprocessor(
            resample_size=resample_size,
            process_nmr=not use_mlp_for_nmr,
            process_ir=True,
            process_c_nmr=not use_mlp_for_nmr,
            nmr_window=h_nmr_range,
            ir_window=ir_range,
            c_nmr_window=c_nmr_range
        )
        
        # Calculate backbone dimensions
        if use_mlp_for_nmr:
            backbone_dim = embed_dim  # IR backbone uses full dimension
            # Create MLPs for NMRs
            self.h_nmr_mlp = nn.Sequential(
                nn.Linear(10000, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.c_nmr_mlp = nn.Sequential(
                nn.Linear(10000, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            n_modalities = 3
            backbone_dim = embed_dim // n_modalities if use_concat else embed_dim
        
        # Calculate the final sequence length after all downsampling
        final_seq_len = resample_size // 32
        
        if verbose:
            print(f"Input sequence length: {resample_size}")
            print(f"Final sequence length: {final_seq_len}")
            print(f"Backbone output dimension: {backbone_dim}")
            print(f"Using concatenation: {use_concat}")
            print(f"Using MLP for NMR: {use_mlp_for_nmr}")
        
        # Base ConvNeXt config
        base_config = {
            'depths': [3, 3, 6, 3],
            'dims': [64, 128, 256, backbone_dim],
            'drop_path_rate': 0.1,
            'layer_scale_init_value': 1e-6,
            'regression': True,
            'regression_dim': backbone_dim
        }
        
        # Create backbones
        if not use_mlp_for_nmr:
            self.nmr_backbone = ConvNeXt1D(in_chans=1, **base_config)
            self.c_nmr_backbone = ConvNeXt1D(in_chans=1, **base_config)
        self.ir_backbone = ConvNeXt1D(in_chans=1, **base_config)
        
        # Only create cross attention components if not using concatenation
        if not use_concat and not use_mlp_for_nmr:
            cross_attn_config = type('Config', (), {
                'n_head': num_heads,
                'n_embd': embed_dim,
                'order': 3,
                'dropout': dropout,
                'bias': True
            })
            self.cross_attention = HigherOrderMultiInputCrossAttention(cross_attn_config)
            self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, nmr_data, ir_data, c_nmr_data):
        if self.verbose:
            print("\nEncoder Processing:")
            print("Processing input data...")
        
        if self.use_mlp_for_nmr:
            # Get raw NMR data
            h_nmr_data = nmr_data[0] if isinstance(nmr_data, tuple) else nmr_data
            c_nmr_data = c_nmr_data[0] if isinstance(c_nmr_data, tuple) else c_nmr_data
            
            # Process only IR through preprocessor
            _, x_ir, _ = self.preprocessor(None, ir_data, None)
        else:
            # Process all data through preprocessor
            x_nmr, x_ir, x_c_nmr = self.preprocessor(nmr_data, ir_data, c_nmr_data)
        
        # Process through backbones/MLPs
        if self.use_mlp_for_nmr:
            # Process NMRs through MLPs
            emb_nmr = self.h_nmr_mlp(h_nmr_data)  # [B, embed_dim]
            emb_nmr = emb_nmr.unsqueeze(1)  # [B, 1, embed_dim]
            
            emb_c_nmr = self.c_nmr_mlp(c_nmr_data)  # [B, embed_dim]
            emb_c_nmr = emb_c_nmr.unsqueeze(1)  # [B, 1, embed_dim]
            
            # Process IR through backbone
            emb_ir = self.ir_backbone(x_ir, keep_sequence=True)  # [B, seq_len, embed_dim]
            
            # Concatenate along sequence dimension
            result = torch.cat([emb_ir, emb_nmr, emb_c_nmr], dim=1)  # [B, seq_len+2, embed_dim]
            
            if self.verbose:
                print(f"\nFinal output (MLP mode): {result.shape}")
            return result
            
        else:
            # Original processing logic
            emb_nmr = self.nmr_backbone(x_nmr, keep_sequence=True)
            emb_ir = self.ir_backbone(x_ir, keep_sequence=True)
            emb_c_nmr = self.c_nmr_backbone(x_c_nmr, keep_sequence=True)
            
            if self.use_concat:
                result = torch.cat([emb_nmr, emb_ir, emb_c_nmr], dim=-1)
                if self.verbose:
                    print(f"\nFinal concatenated output: {result.shape}")
                return result
            else:
                fused = self.cross_attention(emb_nmr, emb_ir, emb_c_nmr)
                fused = self.final_norm(fused)
                if self.verbose:
                    print(f"\nFinal fused output: {fused.shape}")
                return fused