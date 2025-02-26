import torch as th
import torch.nn as nn
from typing import Any

from .spectral_encoder import MultimodalSpectralEncoder
from .transformer_decoder import SMILESDecoder


class MultiModalToSMILESModel(nn.Module):
    """
    A high-level model that:
      1) Encodes IR / H-NMR / C-NMR data via MultimodalSpectralEncoder (concatenation),
         or uses IR as prompt tokens if enabled.
      2) Decodes tokens with SMILESDecoder using a prompt-based approach.
    """

    def __init__(
        self,
        smiles_vocab_size: int,
        nmr_vocab_size: int,
        max_seq_length: int = 512,
        max_nmr_length: int = 128,
        max_memory_length: int = 128,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        verbose: bool = False,
        use_stablemax: bool = False,
        ir_encoder_type: str = "regular",
        ir_as_prompt: bool = False,
        ir_vocab_size: int = None
    ):
        """
        Args:
            smiles_vocab_size: Number of tokens in the SMILES vocabulary.
            nmr_vocab_size:   Number of tokens in the NMR vocabulary.
            max_seq_length:   Max tokens for decoding.
            max_nmr_length:   Max tokens for NMR decoding.
            max_memory_length: Max tokens for memory/IR embeddings.
            embed_dim:  Hidden dimension for the encoder & (matching) decoder memory.
            num_heads:  Number of attention heads in the decoder.
            num_layers: Number of decoder layers.
            dropout:    Dropout probability in the decoder.
            verbose:    If True, print debugging shapes in forward pass.
            use_stablemax: If True, use stablemax instead of softmax in the decoder.
            ir_encoder_type: Type of IR encoder to use.
            ir_as_prompt: If True, use IR as prompt tokens.
            ir_vocab_size: Number of tokens in the IR vocabulary if IR is used as prompt.
        """
        super().__init__()
        self.verbose = verbose
        self.ir_as_prompt = ir_as_prompt
        self.max_memory_length = max_memory_length

        # Initialize spectral encoder; pass the flag so that it bypasses encoding if IR is prompt
        self.encoder = MultimodalSpectralEncoder(
            embed_dim=embed_dim,
            verbose=verbose,
            encoder_type=ir_encoder_type,
            ir_as_prompt=ir_as_prompt
        )

        # If IR is used as prompt, create an embedding layer for IR tokens
        if self.ir_as_prompt:
            if ir_vocab_size is None:
                raise ValueError("ir_vocab_size must be provided when ir_as_prompt is True.")
            self.ir_embed = nn.Embedding(ir_vocab_size, embed_dim)
        else:
            self.ir_embed = None

        # The decoder expects memory_dim == encoder's output dim; note that if IR is prompt, memory will come from IR embedding
        self.decoder = SMILESDecoder(
            smiles_vocab_size=smiles_vocab_size,
            nmr_vocab_size=nmr_vocab_size,
            max_seq_length=max_seq_length,
            max_nmr_length=max_nmr_length,
            max_memory_length=max_memory_length,
            memory_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            verbose=verbose,
            use_stablemax=use_stablemax,
            ir_as_prompt=ir_as_prompt
        )

    def forward(
        self,
        nmr_tokens: th.Tensor | None,
        ir_data: th.Tensor | None,
        target_seq: th.Tensor | None = None,
        target_mask: th.Tensor | None = None
    ):
        """
        Args:
            nmr_tokens: Token IDs for NMR data, shape (B, L).
            ir_data:    IR data, shape (B, L).
            target_seq: Token IDs for SMILES, shape (B, T).
            target_mask: Optional causal mask for the target sequence.

        Returns:
            logits: (B, T, vocab_size), the decoder output for each token.
        """
        if self.verbose:
            print("\n=== Starting Forward Pass ===")
            def shape_str(x):
                if x is None:
                    return "None"
                elif isinstance(x, th.Tensor):
                    return str(x.shape)
                return "Unknown"
            print(f"NMR Tokens: {shape_str(nmr_tokens)}")
            print(f"IR Data:    {shape_str(ir_data)}")
            print(f"Target sequence shape: {shape_str(target_seq)}")

        # Handle memory creation based on mode
        memory = None
        if self.ir_as_prompt:
            # If IR is used as prompt, embed IR tokens using the IR embedding layer
            if ir_data is not None:
                # Ensure IR data is exactly 400 tokens (the expected IR prompt length)
                if ir_data.size(1) != 401:
                    raise ValueError(f"IR prompt must be exactly 400 tokens, got {ir_data.size(1)}")
                memory = self.ir_embed(ir_data)  # ir_data: (B, L) token ids, becomes (B, L, embed_dim)
            else:
                batch_size = target_seq.size(0) if target_seq is not None else (nmr_tokens.size(0) if nmr_tokens is not None else 1)
                device = target_seq.device if target_seq is not None else th.device('cpu')
                memory = th.zeros(batch_size, 401, self.decoder.memory_dim, device=device)
        elif not self.ir_as_prompt:
            # Only use encoder if not in prompt mode
            if ir_data is not None:
                memory = self.encoder(None, ir_data, None)
            
            # Create zero memory if needed
            if memory is None:
                if target_seq is not None:
                    batch_size = target_seq.size(0)
                    device = target_seq.device
                elif nmr_tokens is not None:
                    batch_size = nmr_tokens.size(0)
                    device = nmr_tokens.device
                else:
                    batch_size = 1
                    device = th.device('cpu')
                memory = th.zeros(batch_size, self.max_memory_length, self.decoder.memory_dim, device=device)

        if self.verbose:
            print("\n=== Starting Decoding ===")
            print(f"Encoder Output (memory) shape: {memory.shape}")

        # 2) Decode to SMILES: target_seq => shape (B, T)
        logits = self.decoder(target_seq, memory, nmr_tokens)

        if self.verbose:
            print("\n=== Forward Pass Complete ===")

        return logits 