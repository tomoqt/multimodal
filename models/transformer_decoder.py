import torch as th
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
        scores = th.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B H T S
        
        if mask is not None:
            # Properly reshape mask for broadcasting
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, S]
            scores = scores.masked_fill(mask, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        out = th.matmul(attn, v)  # B H T d
        
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


    def forward(self, x: th.Tensor, memory: th.Tensor, tgt_mask: th.Tensor | None = None):
        """ x: Target sequence tensor (B, T, D)
            memory: Memory tensor (B, M, D)
            tgt_mask: Optional attention mask
        """
        B, T, D = x.shape
        M = memory.shape[1]  # Memory sequence length
        H = self.nhead
        DH = self.head_dim

        # Concatenate memory and target sequence
        x_full = th.cat([memory, x], dim=1)
        
        xattn = self.attn_norm(x_full)

        # Self attention
        q = self.q(xattn).view(B, M+T, H, DH).transpose(1, 2)
        k = self.k(xattn).view(B, M+T, H, DH).transpose(1, 2)
        v = self.v(xattn).view(B, M+T, H, DH).transpose(1, 2)

        if self.use_rope:
            q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
            k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
            cos, sin = self.rotary_emb(q, seq_len=M+T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q = th.cat((q, query_pass), dim=-1)
            k = th.cat((k, key_pass), dim=-1)

        # Create attention mask that allows:
        # 1. Memory tokens to attend to all memory tokens
        # 2. Target tokens to attend causally to themselves and all memory tokens
        if tgt_mask is None:
            mask = th.zeros(B, H, M+T, M+T, device=x_full.device).bool()
            # Memory can attend to itself fully
            mask[:, :, :M, :M] = True
            # Target sequence can attend to memory and causally to itself
            mask[:, :, M:, :M] = True  # Can attend to all memory
            causal_mask = th.triu(th.ones(T, T, device=x_full.device), diagonal=1).bool()
            mask[:, :, M:, M:] = ~causal_mask  # Causal mask for target sequence

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        
        attn = attn.transpose(1, 2).contiguous().view(B, M+T, -1)
        xattn = self.attn_dropout(self.out(attn))
        x_full = xattn + x_full

        # FFN
        xffn = self.mlp_norm(x_full)
        xffn = F.relu(self.ffn_w1(xffn)).square()
        xffn = self.mlp_dropout(self.ffn_w2(xffn))
        x_full = x_full + xffn

        # Return only the target sequence part
        return x_full[:, M:]

class DecoderPromptLayerWithNMR(nn.Module):
    def __init__(self, d_model: int, memory_dim: int, nhead: int, d_ffn: int=2048, dropout=0.1, use_rope: bool = False):
        super().__init__()
        """ Similar to DecoderPromptLayer but handles both memory (IR) and NMR data as prompts.
        The memory (IR) and NMR are injected as prompts, with NMR tokens able to attend to each other.
        The decoding remains autoregressive for the target sequence.
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

    def forward(self, x: th.Tensor, memory: th.Tensor, nmr: th.Tensor, mask: th.Tensor | None = None):
        """ 
        Args:
            x: Target sequence tensor (B, T, D)
            memory: IR embedding tensor (B, M, D)
            nmr: NMR tokens tensor (B, N, D)
            mask: Optional attention mask
        """
        B, T, D = x.shape
        M = memory.shape[1]  # Memory (IR) sequence length
        N = nmr.shape[1]     # NMR sequence length
        H = self.nhead
        DH = self.head_dim

        # Concatenate memory (IR), NMR, and target sequence
        # Order: [memory, nmr, target]
        x_full = th.cat([memory, nmr, x], dim=1)
        
        xattn = self.attn_norm(x_full)

        # Self attention
        q = self.q(xattn).view(B, M+N+T, H, DH).transpose(1, 2)
        k = self.k(xattn).view(B, M+N+T, H, DH).transpose(1, 2)
        v = self.v(xattn).view(B, M+N+T, H, DH).transpose(1, 2)

        if self.use_rope:
            q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
            k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
            cos, sin = self.rotary_emb(q, seq_len=M+N+T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q = th.cat((q, query_pass), dim=-1)
            k = th.cat((k, key_pass), dim=-1)

        # Create attention mask that allows:
        # 1. Memory (IR) tokens to attend to themselves and NMR
        # 2. NMR tokens to attend to themselves and memory
        # 3. Target tokens to attend causally to all previous tokens
        if mask is None:
            mask = th.zeros(B, H, M+N+T, M+N+T, device=x_full.device).bool()
            # Memory (IR) and NMR can attend to each other and themselves
            mask[:, :, :M+N, :M+N] = True
            # Target sequence gets causal mask for itself and can attend to memory and NMR
            mask[:, :, M+N:, :M+N] = True  # Can attend to memory and NMR
            causal_mask = th.triu(th.ones(T, T, device=x_full.device), diagonal=1).bool()
            mask[:, :, M+N:, M+N:] = ~causal_mask  # Causal mask for target sequence

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        
        attn = attn.transpose(1, 2).contiguous().view(B, M+N+T, -1)
        xattn = self.attn_dropout(self.out(attn))
        x_full = xattn + x_full

        # FFN
        xffn = self.mlp_norm(x_full)
        xffn = F.relu(self.ffn_w1(xffn)).square()
        xffn = self.mlp_dropout(self.ffn_w2(xffn))
        x_full = x_full + xffn

        # Return only the target sequence part
        return x_full[:, M+N:]

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
        verbose: bool = True,
        decoder_type: str = "prompt_nmr"  # Add decoder type option
    ):
        super().__init__()
        
        # Store configuration
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.verbose = verbose
        self.max_seq_length = max_seq_length
        self.decoder_type = decoder_type
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Add input projection for memory if dimensions don't match
        self.memory_proj = nn.Linear(memory_dim, embed_dim) if memory_dim != embed_dim else nn.Identity()
        
        # Create decoder layers based on type
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if decoder_type == "vanilla":
                layer = DecoderLayer(
                    d_model=embed_dim,
                    memory_dim=memory_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=dropout,
                    use_rope=True
                )
            elif decoder_type == "prompt":
                layer = DecoderPromptLayer(
                    d_model=embed_dim,
                    memory_dim=memory_dim,
                    nhead=num_heads,
                    d_ffn=embed_dim * 4,
                    dropout=dropout,
                    use_rope=True
                )
            elif decoder_type == "prompt_nmr":
                layer = DecoderPromptLayerWithNMR(
                    d_model=embed_dim,
                    memory_dim=memory_dim,
                    nhead=num_heads,
                    d_ffn=embed_dim * 4,
                    dropout=dropout,
                    use_rope=True
                )
            else:
                raise ValueError(f"Unknown decoder type: {decoder_type}")
            self.layers.append(layer)
        
        # Output projection
        self.out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tgt: th.Tensor, memory: th.Tensor, nmr: th.Tensor | None = None, tgt_mask: th.Tensor | None = None):
        if self.verbose:
            print(f"\nDecoder Input Shapes:")
            print(f"Target sequence: {tgt.shape}")
            print(f"Memory: {memory.shape}")
            if nmr is not None:
                print(f"NMR: {nmr.shape}")
            if tgt_mask is not None:
                print(f"Target mask: {tgt_mask.shape}")

        # Check sequence length
        if tgt.size(1) > self.max_seq_length + 256:
            raise ValueError(f"Input sequence length {tgt.size(1)} exceeds maximum length {self.max_seq_length}")
        
        # (B, T) -> (B, T, D) token embeddings
        x = self.embed(tgt)
        
        # Project memory to correct dimensions
        memory = self.memory_proj(memory)
        
        # Process through layers
        for layer in self.layers:
            if self.decoder_type == "prompt_nmr":
                x = layer(x, memory, nmr, tgt_mask)
            else:
                x = layer(x, memory, tgt_mask)
            
        if self.verbose:
            print(f"Final output shape: {x.shape}")
            
        out = self.out(x)
        return out