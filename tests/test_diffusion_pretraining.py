import unittest

import torch
import torch.nn as nn

from inference.diffusion_inference import DiffusionInference
from models.multimodal_block_diffusion import MultiModalBlockDiffusionModel
from models.multimodal_diffusion import MultiModalDiffusionModel
from models.smiles_tokenizer import SmilesTokenizer
from training.core_dataset import collate_spectral_smiles
from training.diffusion_core import block_diffusion_train_step, diffusion_train_step
from training.diffusion_scheduler import resolve_position_scores


def _encode_target(tokenizer: SmilesTokenizer, smiles: str, max_len: int) -> torch.Tensor:
    token_ids = tokenizer.encode(smiles, add_special_tokens=True, max_length=max_len, truncation=True)
    padded = token_ids + [tokenizer.pad_token_id] * (max_len - len(token_ids))
    return torch.tensor(padded[:max_len], dtype=torch.long)


class DiffusionPretrainingTests(unittest.TestCase):
    def test_collate_preserves_example_indices(self) -> None:
        batch = [
            (
                torch.tensor([1, 2, 0], dtype=torch.long),
                None,
                torch.tensor([7, 0], dtype=torch.long),
                5,
            ),
            (
                torch.tensor([1, 3, 4], dtype=torch.long),
                None,
                torch.tensor([8, 9], dtype=torch.long),
                9,
            ),
        ]
        target_batch, ir_batch, nmr_batch, indices = collate_spectral_smiles(
            batch,
            smiles_pad_token_id=0,
            nmr_pad_token_id=0,
        )
        self.assertEqual(tuple(target_batch.shape), (2, 3))
        self.assertIsNone(ir_batch)
        self.assertEqual(tuple(nmr_batch.shape), (2, 2))
        self.assertTrue(torch.equal(indices, torch.tensor([5, 9], dtype=torch.long)))

    def test_ar_criticality_scores_follow_batch_indices(self) -> None:
        target_batch = torch.tensor(
            [
                [10, 11, 12, 0],
                [20, 21, 22, 0],
            ],
            dtype=torch.long,
        )
        cache = {
            "scores": torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.0],
                    [0.4, 0.5, 0.6, 0.0],
                    [0.7, 0.8, 0.9, 0.0],
                ],
                dtype=torch.float32,
            )
        }
        scores = resolve_position_scores(
            target_tokens=target_batch,
            mask_schedule="ar_criticality",
            ar_criticality_cache=cache,
            example_indices=torch.tensor([2, 0], dtype=torch.long),
        )
        expected = torch.tensor(
            [
                [0.7, 0.8, 0.9, 0.0],
                [0.1, 0.2, 0.3, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(scores, expected))

    def test_decode_cfg_uses_unconditional_branch(self) -> None:
        tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")
        c_token_id = tokenizer.encode("C", add_special_tokens=False)[0]
        o_token_id = tokenizer.encode("O", add_special_tokens=False)[0]

        class GuidanceAwareModel(nn.Module):
            def __init__(self, vocab_size: int) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))
                self.vocab_size = vocab_size
                self.unconditional_calls = 0

            def forward(self, nmr_tokens=None, ir_data=None, target_seq=None):
                batch_size, seq_len = target_seq.shape
                logits = torch.full((batch_size, seq_len, self.vocab_size), -10.0, device=target_seq.device)
                if nmr_tokens is None and ir_data is None:
                    self.unconditional_calls += 1
                    logits[:, :, o_token_id] = 8.0
                    logits[:, :, c_token_id] = 7.0
                else:
                    logits[:, :, c_token_id] = 8.0
                    logits[:, :, o_token_id] = 7.0
                return logits

        model = GuidanceAwareModel(len(tokenizer))
        inference = DiffusionInference(model, tokenizer, device=torch.device("cpu"))
        pred = inference.decode(
            nmr_tokens=torch.tensor([1, 2, 0, 0], dtype=torch.long),
            ir_data=None,
            max_len=6,
            steps=6,
            block_length=2,
            temperature=0.0,
            remasking="low_confidence",
            cfg_scale=1.5,
        )[0]
        self.assertEqual(pred, "CCCCC")
        self.assertGreater(model.unconditional_calls, 0)

    def test_decode_blocks_pad_token(self) -> None:
        tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")
        c_token_id = tokenizer.encode("C", add_special_tokens=False)[0]

        class PadPreferringModel(nn.Module):
            def __init__(self, vocab_size: int, pad_token_id: int, visible_token_id: int) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))
                self.vocab_size = vocab_size
                self.pad_token_id = pad_token_id
                self.visible_token_id = visible_token_id

            def forward(self, nmr_tokens=None, ir_data=None, target_seq=None):
                batch_size, seq_len = target_seq.shape
                logits = torch.full(
                    (batch_size, seq_len, self.vocab_size),
                    -10.0,
                    device=target_seq.device,
                )
                logits[:, :, self.pad_token_id] = 10.0
                logits[:, :, self.visible_token_id] = 9.0
                return logits

        model = PadPreferringModel(len(tokenizer), tokenizer.pad_token_id, c_token_id)
        inference = DiffusionInference(model, tokenizer, device=torch.device("cpu"))
        pred = inference.decode(
            nmr_tokens=torch.tensor([1, 2, 0, 0], dtype=torch.long),
            ir_data=None,
            max_len=6,
            steps=6,
            block_length=2,
            temperature=0.0,
            remasking="low_confidence",
        )[0]
        self.assertEqual(pred, "CCCCC")

    def test_block_decode_uses_blockwise_path(self) -> None:
        tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")
        c_token_id = tokenizer.encode("C", add_special_tokens=False)[0]

        class BlockModel(nn.Module):
            supports_block_diffusion = True

            def __init__(self, vocab_size: int) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))
                self.vocab_size = vocab_size
                self.block_calls = []

            def forward(self, nmr_tokens=None, ir_data=None, target_seq=None, block_start=None, block_end=None):
                self.block_calls.append((int(block_start), int(block_end)))
                batch_size, seq_len = target_seq.shape
                logits = torch.full((batch_size, seq_len, self.vocab_size), -10.0, device=target_seq.device)
                logits[:, :, c_token_id] = 8.0
                return logits

        model = BlockModel(len(tokenizer))
        inference = DiffusionInference(model, tokenizer, device=torch.device("cpu"))
        pred = inference.decode(
            nmr_tokens=torch.tensor([1, 2, 0, 0], dtype=torch.long),
            ir_data=None,
            max_len=6,
            steps=2,
            block_length=2,
            temperature=0.0,
            remasking="low_confidence",
        )[0]
        self.assertEqual(pred, "CCCCC")
        self.assertEqual(model.block_calls[0], (1, 3))

    def test_single_train_step_runs_and_updates_weights(self) -> None:
        tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")
        model = MultiModalDiffusionModel(
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

    def test_block_diffusion_train_step_runs_and_updates_weights(self) -> None:
        tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")
        model = MultiModalBlockDiffusionModel(
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
        loss_value = block_diffusion_train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.cls_token_id,
            device=torch.device("cpu"),
            block_size=4,
            nmr_pad_token_id=0,
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss_value)))
        after = model.decoder.out.weight.detach()
        self.assertGreater(torch.norm(after - before).item(), 0.0)

    def test_iterative_decode_can_overfit_single_example(self) -> None:
        torch.manual_seed(0)
        tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")
        model = MultiModalDiffusionModel(
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
