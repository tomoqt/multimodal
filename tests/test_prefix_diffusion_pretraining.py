import unittest

import torch

from inference.diffusion_inference import DiffusionInference
from models.multimodal_prefix_diffusion import MultiModalPrefixDiffusionModel
from models.smiles_tokenizer import SmilesTokenizer
from training.diffusion_core import diffusion_train_step


def _encode_target(tokenizer: SmilesTokenizer, smiles: str, max_len: int) -> torch.Tensor:
    token_ids = tokenizer.encode(smiles, add_special_tokens=True, max_length=max_len, truncation=True)
    padded = token_ids + [tokenizer.pad_token_id] * (max_len - len(token_ids))
    return torch.tensor(padded[:max_len], dtype=torch.long)


class PrefixDiffusionPretrainingTests(unittest.TestCase):
    def test_single_train_step_runs_and_updates_weights(self) -> None:
        tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")
        model = MultiModalPrefixDiffusionModel(
            smiles_vocab_size=len(tokenizer),
            nmr_vocab_size=32,
            max_seq_length=16,
            max_nmr_length=8,
            max_memory_length=8,
            embed_dim=48,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            ir_as_prompt=False,
            verbose=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        target_batch = torch.stack([_encode_target(tokenizer, "CCO", 16) for _ in range(4)], dim=0)
        nmr_batch = torch.tensor(
            [
                [1, 2, 3, 4, 0, 0, 0, 0],
                [1, 2, 3, 4, 0, 0, 0, 0],
                [1, 2, 3, 4, 0, 0, 0, 0],
                [1, 2, 3, 4, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        batch = (target_batch, None, nmr_batch)

        before = model.decoder.out.weight.detach().clone()
        loss_value = diffusion_train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.cls_token_id,
            device=torch.device("cpu"),
            nmr_pad_token_id=0,
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss_value)))
        after = model.decoder.out.weight.detach()
        self.assertGreater(torch.norm(after - before).item(), 0.0)

    def test_iterative_decode_can_overfit_single_example(self) -> None:
        torch.manual_seed(0)
        tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")
        model = MultiModalPrefixDiffusionModel(
            smiles_vocab_size=len(tokenizer),
            nmr_vocab_size=32,
            max_seq_length=16,
            max_nmr_length=8,
            max_memory_length=8,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            ir_as_prompt=False,
            verbose=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        target = _encode_target(tokenizer, "CCO", 16)
        target_batch = torch.stack([target for _ in range(8)], dim=0)
        nmr_row = torch.tensor([1, 2, 3, 4, 5, 0, 0, 0], dtype=torch.long)
        nmr_batch = torch.stack([nmr_row for _ in range(8)], dim=0)
        batch = (target_batch, None, nmr_batch)

        for _ in range(120):
            diffusion_train_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.cls_token_id,
                device=torch.device("cpu"),
                mask_prob_floor=1e-3,
                nmr_pad_token_id=0,
            )

        inference = DiffusionInference(model, tokenizer, device=torch.device("cpu"))
        pred = inference.decode(
            nmr_tokens=nmr_row,
            ir_data=None,
            max_len=16,
            steps=15,
            block_length=15,
            temperature=0.0,
            remasking="low_confidence",
        )[0]
        self.assertEqual(pred, "CCO")


if __name__ == "__main__":
    unittest.main()
