import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from .rope import apply_rotary_pos_emb

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, use_rope=False, rope_freqs=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.use_rope = use_rope
        self.rope_freqs = rope_freqs

    def forward(self, query, key_value):
        residual = query
        B, T_q, C = query.size()
        B, T_kv, C = key_value.size()
        H = self.num_heads
        head_dim = C // H

        q = self.linear_q(query).view(B, T_q, H, head_dim)
        k = self.linear_k(key_value).view(B, T_kv, H, head_dim)
        v = self.linear_v(key_value).view(B, T_kv, H, head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.use_rope and self.rope_freqs is not None:
            rope_len = max(T_q, T_kv)
            freqs = self.rope_freqs.get_embed(rope_len, q.device)
            freqs = freqs.unsqueeze(0).unsqueeze(0)
            q_rot, k_rot = apply_rotary_pos_emb(q, k, freqs[..., :head_dim])
            q, k = q_rot, k_rot

        attn_output = scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T_q, C)
        out = self.out_proj(attn_output)
        out = residual + self.dropout(out)
        out = self.norm(out)
        return out 