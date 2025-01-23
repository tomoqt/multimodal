import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

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

def stable_s(x, clamp_val=20.0, epsilon=1e-9):
    """
    Piecewise transform:
        s(x) = 1 / (1 - x + epsilon)   if x < 0
               x + 1                  if x >= 0
    
    with added clamping and replacements to avoid NaNs.
    """
    # 1) Replace +/- inf with 0.0 (or some sentinel value)
    x = th.where(th.isinf(x), th.tensor(0.0, device=x.device, dtype=x.dtype), x)

    # 2) Clamp to avoid huge magnitudes: [-20, 20] is somewhat arbitrary, 
    #    but typically prevents float over-/under-flow in exponent-like transforms
    x = x.clamp(-clamp_val, clamp_val)

    # 3) Apply your piecewise function
    s_x = th.where(
        x < 0,
        1.0 / (1.0 - x + epsilon),  # denominator won't blow up unless x ~ 1
        x + 1.0
    )
    return s_x


def stablemax(x, dim=-1, clamp_val=20.0, epsilon=1e-9):
    """
    'Stablemax' using the piecewise transform above, plus a safe denominator.
    """
    # 1) Transform
    s_x = stable_s(x, clamp_val=clamp_val, epsilon=epsilon)

    # 2) Sum along dim
    denom = s_x.sum(dim=dim, keepdim=True)

    # 3) Avoid zero denominator -> clamp_min
    denom = denom.clamp_min(epsilon)

    # 4) Divide
    return s_x / denom

class DecoderPromptLayer(nn.Module):
    def __init__(self, d_model: int, memory_dim: int, nhead: int, d_ffn: int=2048, dropout=0.1, use_rope: bool = False, use_stablemax: bool = False):
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
        self.use_stablemax = use_stablemax
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

        # Modified attention computation to use stablemax if enabled
        if self.use_stablemax:
            # Compute attention scores
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(DH)

            
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))

            
            # Use stablemax directly
            attn_weights = stablemax(scores, dim=-1)

            attn = attn_weights @ v
        else:
            # Use standard scaled dot-product attention
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
        smiles_vocab_size: int,
        nmr_vocab_size: int,  # Separate vocab size for NMR tokens
        max_seq_length: int = 512,
        max_nmr_length: int = 128,
        max_memory_length: int = 128,
        memory_dim: int = 768,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: int = 0.1,
        verbose: bool = True,
        use_stablemax: bool = False  # Add this parameter
    ):
        super().__init__()
        
        # Store configuration
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.verbose = verbose
        self.max_seq_length = max_seq_length
        self.max_nmr_length = max_nmr_length
        self.max_memory_length = max_memory_length
        
        # Separate embeddings with different vocabulary sizes
        self.smiles_embed = nn.Embedding(smiles_vocab_size, embed_dim)
        self.nmr_embed = nn.Embedding(nmr_vocab_size, embed_dim)
        
        # Add input projection for memory if dimensions don't match
        self.memory_proj = nn.Linear(memory_dim, embed_dim) if memory_dim != embed_dim else nn.Identity()
        
        # Create decoder layers with stablemax option
        self.layers = nn.ModuleList([
            DecoderPromptLayer(
                d_model=embed_dim,
                memory_dim=memory_dim,
                nhead=num_heads,
                d_ffn=embed_dim * 4,
                dropout=dropout,
                use_rope=True,
                use_stablemax=use_stablemax  # Pass the stablemax option
            ) for _ in range(num_layers)
        ])
        
        # Add final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Output projection to SMILES vocabulary
        self.out = nn.Linear(embed_dim, smiles_vocab_size)
        
    def forward(self, tgt: th.Tensor, memory: th.Tensor, nmr_tokens: th.Tensor = None):
        """ inputs:
            tgt: target sequence tensor, shape (B, T)
            memory: memory tensor from IR encoder, shape (B, S, D)
            nmr_tokens: tokenized NMR data, shape (B, N)
        """
        B, T = tgt.shape
        if self.verbose:
            print(f"\nDecoder Input Shapes:")
            print(f"Target sequence: {tgt.shape}")
            print(f"Memory: {memory.shape}")
            if nmr_tokens is not None:
                print(f"NMR tokens: {nmr_tokens.shape}")

        # Check and enforce sequence length limits
        if memory.size(1) > self.max_memory_length:
            if self.verbose:
                print(f"Warning: Truncating memory from {memory.size(1)} to {self.max_memory_length}")
            memory = memory[:, :self.max_memory_length]

        if nmr_tokens is not None and nmr_tokens.size(1) > self.max_nmr_length:
            if self.verbose:
                print(f"Warning: Truncating NMR tokens from {nmr_tokens.size(1)} to {self.max_nmr_length}")
            nmr_tokens = nmr_tokens[:, :self.max_nmr_length]

        # Calculate total prompt length (memory + NMR)
        total_prompt_length = memory.size(1) + (nmr_tokens.size(1) if nmr_tokens is not None else 0)
        
        # Check if target sequence + prompt length exceeds maximum
        if T + total_prompt_length > self.max_seq_length + 256:
            raise ValueError(
                f"Combined sequence length ({T} + {total_prompt_length} = {T + total_prompt_length}) "
                f"exceeds maximum length ({self.max_seq_length + 256})"
            )

        # Embed target sequence using SMILES embeddings
        x = self.smiles_embed(tgt)
        
        # Project memory to full embedding dimension
        memory = self.memory_proj(memory)
        
        # Embed NMR tokens if provided and handle concatenation
        if nmr_tokens is not None:
            # Add dimension checks
            if self.verbose:
                print(f"Memory shape after projection: {memory.shape}")
                print(f"NMR tokens device: {nmr_tokens.device}, Memory device: {memory.device}")
                print(f"NMR tokens dtype: {nmr_tokens.dtype}, Memory dtype: {memory.dtype}")
            
            # Ensure NMR tokens are on the same device as memory
            nmr_tokens = nmr_tokens.to(memory.device)
            # Use NMR-specific embeddings
            nmr_embeddings = self.nmr_embed(nmr_tokens)  # (B, N, D)
            
            if self.verbose:
                print(f"NMR embeddings shape: {nmr_embeddings.shape}")
            
            # Check that dimensions match before concatenation
            assert memory.size(0) == nmr_embeddings.size(0), "Batch sizes don't match"
            assert memory.size(2) == nmr_embeddings.size(2), "Embedding dimensions don't match"
            
            # Concatenate memory (IR) and NMR embeddings
            prompt = th.cat([memory, nmr_embeddings], dim=1)  # (B, S+N, D)
        else:
            prompt = memory

        if self.verbose:
            print(f"Prompt shape after concatenation: {prompt.shape}")
            print(f"Target embeddings shape: {x.shape}")

        # Concatenate prompt and target embeddings
        M = prompt.shape[1]  # Total prompt length (IR + NMR)
        x = th.cat([prompt, x], dim=1)  # (B, S+N+T, D)

        # Create mask: prompt tokens can attend to each other, target is causal
        mask = th.ones(B, M+T, M+T, device=x.device).tril().bool()
        # Allow full attention within prompt
        mask[:, :M, :M] = True
        mask = mask[:, None].repeat(1, self.num_heads, 1, 1)

        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Only project the target sequence portion to vocabulary
        x_target = x[:, M:]  # Extract only the target sequence part
        x_target = self.final_norm(x_target)  # Apply final layer normalization
        out = self.out(x_target)  # (B, T, vocab_size)
        
        if self.verbose:
            print(f"Target sequence output shape: {out.shape}")

        return out