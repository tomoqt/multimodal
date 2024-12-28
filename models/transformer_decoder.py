import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_pos_emb

class DecoderLayer(nn.Module):
    def __init__(self, d_model, memory_dim, nhead, dim_feedforward=2048, dropout=0.1, use_rope=False, rope_freqs=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_rope = use_rope
        self.rope_freqs = rope_freqs
        
        # Self attention
        self.q1 = nn.Linear(d_model, d_model)
        self.k1 = nn.Linear(d_model, d_model)
        self.v1 = nn.Linear(d_model, d_model)
        
        # Cross attention - note k2 and v2 use memory_dim
        self.q2 = nn.Linear(d_model, d_model)
        self.k2 = nn.Linear(memory_dim, d_model)
        self.v2 = nn.Linear(memory_dim, d_model)
        
        # Output projections
        self.out1 = nn.Linear(d_model, d_model)
        self.out2 = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _multihead_attn(self, q, k, v, mask=None):
        B = q.size(0)
        T = q.size(1)
        S = k.size(1)  # Source sequence length
        H = self.nhead
        
        # Project and split heads
        q = q.view(B, T, H, self.head_dim).transpose(1, 2)  # B H T d
        k = k.view(B, S, H, self.head_dim).transpose(1, 2)  # B H S d
        v = v.view(B, S, H, self.head_dim).transpose(1, 2)  # B H S d
        
        # Apply RoPE if enabled
        if self.use_rope:
            rope_len = max(T, S)
            if self.rope_freqs is not None:
                freqs = self.rope_freqs.get_embed(rope_len, q.device)
                freqs = freqs.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
                q_rot, k_rot = apply_rotary_pos_emb(q, k, freqs[..., :self.head_dim])
                q, k = q_rot, k_rot
        
        # Scaled dot product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B H T S
        
        if mask is not None:
            # Properly reshape mask for broadcasting
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, S]
            scores = scores.masked_fill(mask, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # B H T d
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return out
        
    def forward(self, x, memory, tgt_mask=None):
        # Self attention
        q1 = self.q1(x)
        k1 = self.k1(x)
        v1 = self.v1(x)
        out_self = self._multihead_attn(q1, k1, v1, tgt_mask)
        out_self = self.dropout(self.out1(out_self))
        x = self.norm1(x + out_self)
        
        # Cross attention
        q2 = self.q2(x)
        k2 = self.k2(memory)
        v2 = self.v2(memory)
        out_cross = self._multihead_attn(q2, k2, v2)
        out_cross = self.dropout(self.out2(out_cross))
        x = self.norm2(x + out_cross)
        
        # FFN
        out_ffn = self.dropout(self.ffn(x))
        x = self.norm3(x + out_ffn)
        
        return x

class SMILESDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_length=512,
        memory_dim=768,
        embed_dim=768,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        dim_feedforward=2048,
        verbose=True
    ):
        super().__init__()
        
        # Store configuration
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.verbose = verbose
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        
        # Add input projection for memory if dimensions don't match
        self.memory_proj = nn.Linear(memory_dim, memory_dim)
        
        # Create decoder layers with updated memory dimension
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=embed_dim,
                memory_dim=memory_dim,  # Pass memory dimension
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_rope=False
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tgt, memory, tgt_mask=None):
        if self.verbose:
            print(f"\nDecoder Input Shapes:")
            print(f"Target sequence: {tgt.shape}")
            print(f"Memory: {memory.shape}")
            if tgt_mask is not None:
                print(f"Target mask: {tgt_mask.shape}")
        
        x = self.embed(tgt)
        if self.verbose:
            print(f"After embedding: {x.shape}")
            
        x = x + self.pos_embed[:, :x.size(1)]
        
        # Project memory to correct dimensions
        memory = self.memory_proj(memory)
        if self.verbose:
            print(f"After memory projection: {memory.shape}")
        
        # Ensure memory has sequence dimension
        if memory.dim() == 2:
            memory = memory.unsqueeze(1)
        
        # Expand memory batch dimension if needed
        if memory.size(0) == 1 and x.size(0) > 1:
            memory = memory.expand(x.size(0), -1, -1)
        # Expand memory sequence dimension if needed 
        elif memory.size(1) == 1:
            memory = memory.expand(-1, x.size(1), -1)
            
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
            
        if self.verbose:
            print(f"Final output shape: {x.shape}")
            
        return self.out(x)