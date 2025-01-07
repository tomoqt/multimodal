import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rms_norm(x, eps=1e-5):
    """
    GPT-style RMSNorm
    x: [batch, seq, hidden_dim]
    """
    norm_x = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + eps)
    return norm_x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)

    def forward(self, q, k):
        """Apply rotary embeddings to q and k"""
        b, h, seq_len, d = q.shape
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

        return q_rot, k_rot

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, use_rotary=True, max_seq_len=2048):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.use_rotary = use_rotary

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        if use_rotary:
            self.rotary = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(self, x, attn_mask=None):
        B, T, _ = x.shape
        H = self.n_head

        q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)

        if self.use_rotary:
            q, k = self.rotary(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)

class CrossAttention(nn.Module):
    def __init__(self, d_model, memory_dim, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.memory_dim = memory_dim
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(memory_dim, d_model, bias=False)
        self.v_proj = nn.Linear(memory_dim, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, memory):
        B, T, _ = x.shape
        S = memory.size(1)
        H = self.n_head

        q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, S, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, S, H, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)

class MLP(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        inner_dim = expansion * d_model
        self.fc1 = nn.Linear(d_model, inner_dim, bias=False)
        self.fc2 = nn.Linear(inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x).square()  # ReLU^2
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, memory_dim, n_head, use_rotary=True, dropout=0.1, expansion=4, max_seq_len=2048):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_head, use_rotary=use_rotary, max_seq_len=max_seq_len)
        self.cross_attn = CrossAttention(d_model, memory_dim, n_head)
        self.mlp = MLP(d_model, expansion=expansion, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = lambda x: rms_norm(x)
        self.norm2 = lambda x: rms_norm(x)
        self.norm3 = lambda x: rms_norm(x)

    def forward(self, x, memory, self_mask=None):
        # Pre-norm architecture
        h = self.norm1(x)
        h = self.self_attn(h, attn_mask=self_mask)
        x = x + self.dropout(h)

        h = self.norm2(x)
        h = self.cross_attn(h, memory)
        x = x + self.dropout(h)

        h = self.norm3(x)
        h = self.mlp(h)
        x = x + self.dropout(h)

        return x

class EnhancedDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, memory_dim, n_head, n_layers, dropout=0.1, 
                 max_seq_length=2048, use_rotary=True):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                memory_dim=memory_dim,
                n_head=n_head,
                use_rotary=use_rotary,
                dropout=dropout,
                max_seq_len=max_seq_length
            )
            for _ in range(n_layers)
        ])
        
        self.final_norm = lambda x: rms_norm(x)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, memory, mask=None):
        x = self.token_embedding(x)
        
        for layer in self.layers:
            x = layer(x, memory, self_mask=mask)
            
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits 