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
                 use_concat=True, verbose=True, domain_ranges=None):
        super().__init__()
        
        # Add assertion check for embedding dimension
        if use_concat and embed_dim % 3 != 0:
            raise ValueError(
                f"When using concatenation (use_concat=True), embed_dim ({embed_dim}) "
                f"must be divisible by 3 (number of modalities) to ensure equal "
                f"dimension distribution across modalities."
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
            process_nmr=True,
            process_ir=True,
            process_c_nmr=True,
            nmr_window=h_nmr_range,
            ir_window=ir_range,
            c_nmr_window=c_nmr_range
        )
        
        # Calculate individual backbone output dimensions
        n_modalities = 3
        backbone_dim = embed_dim // n_modalities
        
        # Calculate the final sequence length after all downsampling
        # resample_size -> /4 (stem) -> /2 -> /2 -> /2 (three downsampling layers)
        final_seq_len = resample_size // 32
        
        # Base ConvNeXt config with reduced final dimension
        base_config = {
            'depths': [3, 3, 6, 3],
            'dims': [64, 128, 256, backbone_dim],  # Final dim is embed_dim//3 per modality
            'drop_path_rate': 0.1,
            'layer_scale_init_value': 1e-6,
            'regression': True,
            'regression_dim': backbone_dim  # Each backbone outputs embed_dim//3
        }
        
        if verbose:
            print(f"Input sequence length: {resample_size}")
            print(f"Final sequence length: {final_seq_len}")
            print(f"Backbone output dimension: {backbone_dim}")
        
        # Create 1D backbones for all spectra
        self.nmr_backbone = ConvNeXt1D(in_chans=1, **base_config)
        self.ir_backbone = ConvNeXt1D(in_chans=1, **base_config)
        self.c_nmr_backbone = ConvNeXt1D(in_chans=1, **base_config)
        
        # Ensure all backbones are on the same device as the parent module
        device = next(self.parameters()).device
        self.nmr_backbone = self.nmr_backbone.to(device)
        self.ir_backbone = self.ir_backbone.to(device)
        self.c_nmr_backbone = self.c_nmr_backbone.to(device)
        
        self.use_concat = use_concat
        
        # Only create cross attention components if not using concatenation
        if not use_concat:
            # Add higher-order cross attention
            cross_attn_config = type('Config', (), {
                'n_head': num_heads,
                'n_embd': embed_dim,
                'order': 3,  # We still have 3 modalities: H-NMR, IR, C-NMR
                'dropout': dropout,
                'bias': True
            })
            self.cross_attention = HigherOrderMultiInputCrossAttention(cross_attn_config)
            
            # Add final layer norm
            self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, nmr_data, ir_data, c_nmr_data):
        if self.verbose:
            print("\nEncoder Processing:")
            print("Processing input data...")
            
        # Get the device of the model
        device = next(self.parameters()).device
        
        # Preprocess the input data
        x_nmr, x_ir, x_c_nmr = self.preprocessor(nmr_data, ir_data, c_nmr_data)
        
        if self.verbose:
            print(f"Preprocessed shapes:")
            print(f"NMR: {x_nmr.shape}")
            print(f"IR: {x_ir.shape}")
            print(f"C-NMR: {x_c_nmr.shape}")
        
        # Reshape inputs to [batch, channels, sequence]
        if isinstance(x_nmr, tuple):
            x_nmr = x_nmr[0]
        if isinstance(x_ir, tuple):
            x_ir = x_ir[0]
        if isinstance(x_c_nmr, tuple):
            x_c_nmr = x_c_nmr[0]
        
        # Add channel dimension and transpose if needed
        if x_nmr.dim() == 2:
            x_nmr = x_nmr.unsqueeze(1)  # [batch, 1, sequence]
        elif x_nmr.dim() == 3 and x_nmr.size(1) > x_nmr.size(2):  # if [batch, sequence, 1]
            x_nmr = x_nmr.transpose(1, 2)  # [batch, 1, sequence]
        
        if x_ir.dim() == 2:
            x_ir = x_ir.unsqueeze(1)
        elif x_ir.dim() == 3 and x_ir.size(1) > x_ir.size(2):
            x_ir = x_ir.transpose(1, 2)
        
        if x_c_nmr.dim() == 2:
            x_c_nmr = x_c_nmr.unsqueeze(1)
        elif x_c_nmr.dim() == 3 and x_c_nmr.size(1) > x_c_nmr.size(2):
            x_c_nmr = x_c_nmr.transpose(1, 2)
        
        if self.verbose:
            print(f"Reshaped input shapes:")
            print(f"NMR: {x_nmr.shape}")
            print(f"IR: {x_ir.shape}")
            print(f"C-NMR: {x_c_nmr.shape}")
        
        # Pass through backbones
        emb_nmr = self.nmr_backbone(x_nmr, keep_sequence=True)    # [B, seq_len, embed_dim//3]
        emb_ir = self.ir_backbone(x_ir, keep_sequence=True)       # [B, seq_len, embed_dim//3]
        emb_c_nmr = self.c_nmr_backbone(x_c_nmr, keep_sequence=True)  # [B, seq_len, embed_dim//3]
        
        if self.verbose:
            print(f"\nBackbone outputs:")
            print(f"NMR embedding: {emb_nmr.shape}")
            print(f"IR embedding: {emb_ir.shape}")
            print(f"C-NMR embedding: {emb_c_nmr.shape}")
            
        if self.use_concat:
            # All sequences should have same length after backbone processing
            assert emb_nmr.size(1) == emb_ir.size(1) == emb_c_nmr.size(1), "Sequence lengths must match"
            
            # Concatenate along embedding dimension
            result = torch.cat([emb_nmr, emb_ir, emb_c_nmr], dim=-1)  # [B, seq_len, embed_dim]
            
            if self.verbose:
                print(f"\nFinal concatenated output: {result.shape}")
            return result
        else:
            # Apply higher-order cross attention
            fused = self.cross_attention(emb_nmr, emb_ir, emb_c_nmr)
            
            # Apply final normalization
            fused = self.final_norm(fused)
            
            if self.verbose:
                print(f"\nFinal fused output: {fused.shape}")
            return fused