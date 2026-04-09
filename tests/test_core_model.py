import unittest

import torch

from models.multimodal_to_smiles import MultiModalToSMILESModel


class CoreModelTests(unittest.TestCase):
    def test_forward_with_ir_encoder_memory(self) -> None:
        model = MultiModalToSMILESModel(
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
        ir_data = torch.randn(batch_size, 128)
        target_seq = torch.randint(0, 127, (batch_size, 20))

        logits = model(nmr_tokens=nmr_tokens, ir_data=ir_data, target_seq=target_seq)
        self.assertEqual(logits.shape, (batch_size, 20, 128))

    def test_forward_with_ir_prompt_tokens(self) -> None:
        model = MultiModalToSMILESModel(
            smiles_vocab_size=96,
            nmr_vocab_size=128,
            max_seq_length=24,
            max_nmr_length=24,
            max_memory_length=8,
            embed_dim=48,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            ir_as_prompt=True,
            ir_vocab_size=300,
            verbose=False,
        )

        batch_size = 3
        nmr_tokens = torch.randint(0, 127, (batch_size, 16))
        ir_prompt_ids = torch.randint(0, 299, (batch_size, 20))
        target_seq = torch.randint(0, 95, (batch_size, 12))

        logits = model(nmr_tokens=nmr_tokens, ir_data=ir_prompt_ids, target_seq=target_seq)
        self.assertEqual(logits.shape, (batch_size, 12, 96))

    def test_forward_with_padding_masks(self) -> None:
        model = MultiModalToSMILESModel(
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
        nmr_tokens = torch.randint(1, 255, (batch_size, 24))
        nmr_tokens[:, -6:] = 0
        ir_data = torch.randn(batch_size, 128)
        target_seq = torch.randint(1, 127, (batch_size, 20))
        target_seq[:, -4:] = 0

        logits = model(
            nmr_tokens=nmr_tokens,
            ir_data=ir_data,
            target_seq=target_seq,
            target_padding_mask=(target_seq == 0),
            nmr_padding_mask=(nmr_tokens == 0),
        )
        self.assertEqual(logits.shape, (batch_size, 20, 128))


if __name__ == "__main__":
    unittest.main()
