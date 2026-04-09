import unittest

import torch

from models.multimodal_prefix_diffusion import MultiModalPrefixDiffusionModel


class PrefixDiffusionModelTests(unittest.TestCase):
    def test_forward_with_ir_encoder_prefix(self) -> None:
        model = MultiModalPrefixDiffusionModel(
            smiles_vocab_size=128,
            nmr_vocab_size=256,
            max_seq_length=32,
            max_nmr_length=32,
            max_memory_length=16,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            ir_encoder_type="regular",
            ir_as_prompt=False,
            verbose=False,
        )

        batch_size = 2
        nmr_tokens = torch.randint(0, 255, (batch_size, 24))
        ir_data = torch.randn(batch_size, 64)
        target_seq = torch.randint(0, 127, (batch_size, 20))

        logits = model(nmr_tokens=nmr_tokens, ir_data=ir_data, target_seq=target_seq)
        self.assertEqual(logits.shape, (batch_size, 20, 128))

    def test_forward_without_ir_still_runs(self) -> None:
        model = MultiModalPrefixDiffusionModel(
            smiles_vocab_size=96,
            nmr_vocab_size=128,
            max_seq_length=24,
            max_nmr_length=24,
            max_memory_length=8,
            embed_dim=48,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            ir_encoder_type="regular",
            ir_as_prompt=False,
            verbose=False,
        )

        batch_size = 3
        nmr_tokens = torch.randint(0, 127, (batch_size, 16))
        target_seq = torch.randint(0, 95, (batch_size, 12))

        logits = model(nmr_tokens=nmr_tokens, ir_data=None, target_seq=target_seq)
        self.assertEqual(logits.shape, (batch_size, 12, 96))


if __name__ == "__main__":
    unittest.main()
