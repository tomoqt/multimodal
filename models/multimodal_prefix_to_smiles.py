from .multimodal_to_smiles import MultiModalToSMILESModel
from .prefix_autoregressive_decoder import PrefixSMILESDecoder


class MultiModalPrefixToSMILESModel(MultiModalToSMILESModel):
    """
    Autoregressive variant that injects IR memory and NMR tokens directly as a
    prefix inside the same causal transformer stack.
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
        ir_vocab_size: int = None,
        use_rmsnorm: bool = False,
    ) -> None:
        del use_stablemax
        del use_rmsnorm
        super().__init__(
            smiles_vocab_size=smiles_vocab_size,
            nmr_vocab_size=nmr_vocab_size,
            max_seq_length=max_seq_length,
            max_nmr_length=max_nmr_length,
            max_memory_length=max_memory_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            verbose=verbose,
            use_stablemax=False,
            ir_encoder_type=ir_encoder_type,
            ir_as_prompt=ir_as_prompt,
            ir_vocab_size=ir_vocab_size,
            use_rmsnorm=False,
        )
        self.decoder = PrefixSMILESDecoder(
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
            ir_as_prompt=ir_as_prompt,
        )
