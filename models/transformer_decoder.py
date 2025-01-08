import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(th.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (th.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = th.arange(seq_len, device=x.device)
            freqs = th.einsum("i,j->ij", t, self.inv_freq)
            emb = th.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return th.cat((-x2, x1), -1)


# from: https://github.com/BlinkDL/SmallInitEmb/blob/main/model.py
# @th.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[..., : q.shape[-2], :], sin[..., : q.shape[-2], :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class DecoderPromptLayer(nn.Module):
    def __init__(self, d_model: int, memory_dim: int, nhead: int, d_ffn: int=2048, dropout=0.1, use_rope: bool = False):
        super().__init__()
        """ Same application as `DecoderLayer` but removes the x-attn part and only usesself-attn
        The `memory` is injected as the prompt, and allowed to fully interact. 
        The decoding is autoregressive. 
        """
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.memory_dim = memory_dim
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_rope = use_rope
        assert use_rope, "Other posemb than rope not supported atm"
        if use_rope:
            self.rotary_ndims = int(self.head_dim * 0.5)
            self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        # Self attention
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        # FFN
        self.ffn_w1 = nn.Linear(d_model, d_ffn)
        self.ffn_w2 = nn.Linear(d_ffn, d_model)

        # Layer norms
        self.attn_norm = nn.LayerNorm(d_model)
        self.mlp_norm = nn.LayerNorm(d_model)

        self.attn_dropout = nn.Dropout(0.0)
        self.mlp_dropout = nn.Dropout(0.0)
        self.intra_mlp_dropout = nn.Dropout(0.0)


    def forward(self, x: th.Tensor, mask: th.Tensor | None = None):
        """ x: (B, T, D) and mask: (B, T, T) """
        B, T, D = x.shape
        H = self.nhead
        DH = self.head_dim

        xattn = self.attn_norm(x)

        # Self attention
        q = self.q(xattn).view(B, T, H, DH).transpose(1, 2)
        k = self.k(xattn).view(B, T, H, DH).transpose(1, 2)
        v = self.v(xattn).view(B, T, H, DH).transpose(1, 2)

        if self.use_rope:
            q, query_pass = q[..., : self.rotary_ndims], q[..., self.rotary_ndims :]
            k, key_pass = k[..., : self.rotary_ndims], k[..., self.rotary_ndims :]
            cos, sin = self.rotary_emb(q, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)  # rotary encoding
            q = th.cat((q, query_pass), dim=-1)
            k = th.cat((k, key_pass), dim=-1)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        xattn = self.attn_dropout(self.out(attn))
        x = xattn + x

        # FFN
        xffn = self.mlp_norm(x)
        xffn = F.relu(self.ffn_w1(xffn)).square()
        xffn = self.mlp_dropout(self.ffn_w2(xffn))
        x = x + xffn

        return x

class SMILESDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 512,
        memory_dim: int = 768,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: int = 0.1,
        verbose: bool = True
    ):
        super().__init__()
        
        # Store configuration
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.verbose = verbose
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Add input projection for memory if dimensions don't match
        # original:
        # self.memory_proj = nn.Linear(memory_dim, memory_dim)
        self.memory_proj = nn.Linear(memory_dim, embed_dim) if memory_dim != embed_dim else nn.Identity()
        
        # Create decoder layers with updated memory dimension
        self.layers = nn.ModuleList([
            # DecoderLayer(
            DecoderPromptLayer(
                d_model=embed_dim,
                memory_dim=memory_dim,
                nhead=num_heads,
                d_ffn=embed_dim * 4,
                dropout=dropout,
                use_rope=True
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tgt: th.Tensor, memory: th.Tensor):
        """ inputs:
            tgt: target sequence tensor, shape (B, T)
            memory: memory tensor, shape (B, D) or (B, S, D)
        """
        B, T, = tgt.shape
        if self.verbose:
            print(f"\nDecoder Input Shapes:")
            print(f"Target sequence: {tgt.shape}")
            print(f"Memory: {memory.shape}")

        # Check sequence length
        if tgt.size(1) > self.max_seq_length + 256:
            raise ValueError(f"Input sequence length {tgt.size(1)} exceeds maximum length {self.max_seq_length}")
        
        # (B, T) -> (B, T, D) token embeddings
        x = self.embed(tgt)
        if self.verbose:
            print(f"After embedding: {x.shape}")

        # Project memory to correct dimensions
        memory = self.memory_proj(memory)
        if self.verbose:
            print(f"After memory projection: {memory.shape}")
        
        # Ensure memory has sequence dimension
        if memory.dim() == 2:
            print(f"memory.shape: {memory.shape}")
            memory = memory.unsqueeze(1)

        # Expand memory batch dimension if needed
        # if memory.size(0) == 1 and x.size(0) > 1:
        #    memory = memory.expand(x.size(0), -1, -1)

        # mix `memory` and `x` as prompt + answer, and add causal mask
        M = memory.shape[-2]
        x = th.cat([memory, x], dim=1)

        mask = th.ones(B, T+M, T+M, device=x.device).tril().bool()
        # memory can attend to all of itself. Unlike causal decoding
        mask[:, :M, :M] = True
        mask = mask[:, None].repeat(1, self.num_heads, 1, 1)


        for layer in self.layers:
            x = layer(x, mask)
            
        if self.verbose:
            print(f"Final output shape: {x.shape}")

        # remove memory prompt from output tokens
        out = self.out(x[:, M:])
        return out