import torch
import math

class RoPEFreqs:
    def __init__(self, dim, base=10000):
        """
        Initialize RoPE frequency calculator.
        Args:
            dim: Feature dimension
            base: Base for exponential increase in frequencies
        """
        self.dim = dim
        self.base = base
        
    def get_embed(self, seq_len, device):
        """
        Get RoPE embeddings for given sequence length.
        Args:
            seq_len: Length of sequence
            device: Torch device
        Returns:
            Tensor of shape [seq_len, dim] containing frequency embeddings
        """
        dims = torch.arange(0, self.dim, 2, device=device).float()
        freqs = torch.pow(self.base, -dims / self.dim)
        pos = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(pos, freqs)
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        freqs = torch.stack([freqs_cos, freqs_sin], dim=-1)
        freqs = freqs.view(seq_len, self.dim)
        return freqs

def apply_rotary_pos_emb(q, k, freqs):
    """
    Apply rotary positional embeddings to queries and keys.
    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim]
        k: Key tensor of shape [batch, heads, seq_len, head_dim]
        freqs: Frequency tensor from RoPEFreqs of shape [seq_len, head_dim]
    Returns:
        Tuple of rotated (q, k)
    """
    # Extract cos and sin components
    cos = freqs[..., 0::2]  # [seq_len, head_dim/2]
    sin = freqs[..., 1::2]  # [seq_len, head_dim/2]
    
    # Reshape q and k to split last dimension
    q_split = q.view(*q.shape[:-1], -1, 2)  # [..., head_dim/2, 2]
    k_split = k.view(*k.shape[:-1], -1, 2)  # [..., head_dim/2, 2]
    
    # Apply rotation using complex multiplication
    def rotate(x, cos, sin):
        x1, x2 = x[..., 0], x[..., 1]
        return torch.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
    
    # Rotate q and k
    q_rot = rotate(q_split, cos, sin)
    k_rot = rotate(k_split, cos, sin)
    
    # Reshape back
    q_rot = q_rot.view(*q.shape)
    k_rot = k_rot.view(*k.shape)
    
    return q_rot, k_rot