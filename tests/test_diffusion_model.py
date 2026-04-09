import unittest

import torch

from models.multimodal_block_diffusion import MultiModalBlockDiffusionModel
from models.multimodal_diffusion import MultiModalDiffusionModel
from training.diffusion_core import sample_noisy_targets


class DiffusionModelTests(unittest.TestCase):
    def test_forward_with_ir_encoder_memory(self) -> None:
        model = MultiModalDiffusionModel(
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

    def test_noising_skips_pad_and_bos(self) -> None:
        target_tokens = torch.tensor(
            [
                [101, 10, 11, 102, 0, 0],
                [101, 12, 13, 14, 102, 0],
            ],
            dtype=torch.long,
        )
        noisy, masked, _, valid = sample_noisy_targets(
            target_tokens=target_tokens,
            mask_token_id=103,
            pad_token_id=0,
            bos_token_id=101,
            mask_prob_floor=1.0,
        )

        self.assertTrue(torch.equal(noisy[:, 0], target_tokens[:, 0]))
        self.assertTrue(torch.equal(noisy[target_tokens == 0], target_tokens[target_tokens == 0]))
        self.assertFalse(masked[:, 0].any())
        self.assertFalse(masked[target_tokens == 0].any())
        self.assertTrue(valid[0, 1])

    def test_block_diffusion_forward_with_block_mask(self) -> None:
        model = MultiModalBlockDiffusionModel(
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

        logits = model(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            target_seq=target_seq,
            target_padding_mask=target_seq.eq(0),
            block_start=5,
            block_end=9,
        )
        self.assertEqual(logits.shape, (batch_size, 20, 128))


if __name__ == "__main__":
    unittest.main()
