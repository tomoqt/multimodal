from .multimodal_diffusion import MultiModalDiffusionModel


class MultiModalBlockDiffusionModel(MultiModalDiffusionModel):
    """
    BD3-style block diffusion model.

    Conditioning remains the same as the cross-attention baseline, but the
    denoiser is evaluated block-by-block with a block-causal target mask that
    exposes previous clean blocks and the current noisy block.
    """

    supports_block_diffusion = True
