import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.smiles_tokenizer import SmilesTokenizer
from tests.helpers import materialize_tokenized_dataset
from training.core_dataset import SpectralSmilesDataset, collate_spectral_smiles


class CoreDatasetTests(unittest.TestCase):
    def test_dataset_and_collate_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenized_dir, nmr_tokenizer = materialize_tokenized_dataset(Path(tmpdir), num_rows=24)
            smiles_tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")

            dataset = SpectralSmilesDataset(
                data_dir=str(tokenized_dir),
                smiles_tokenizer=smiles_tokenizer,
                spectral_tokenizer=nmr_tokenizer,
                split="train",
                max_smiles_len=64,
                max_nmr_len=64,
            )
            self.assertGreater(len(dataset), 0)

            tgt, ir, nmr = dataset[0]
            self.assertEqual(tgt.dim(), 1)
            self.assertEqual(nmr.dim(), 1)
            self.assertTrue(ir is None or ir.dim() == 1)

            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=lambda b: collate_spectral_smiles(
                    b,
                    smiles_pad_token_id=smiles_tokenizer.pad_token_id,
                    nmr_pad_token_id=nmr_tokenizer["<PAD>"],
                ),
            )
            target_batch, ir_batch, nmr_batch = next(iter(loader))
            self.assertEqual(target_batch.dim(), 2)
            self.assertEqual(nmr_batch.dim(), 2)
            self.assertEqual(target_batch.size(0), 4)
            self.assertEqual(nmr_batch.size(0), 4)
            self.assertTrue(ir_batch is None or isinstance(ir_batch, torch.Tensor))

    def test_disk_token_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            tokenized_dir, nmr_tokenizer = materialize_tokenized_dataset(base, num_rows=20)
            smiles_tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")
            cache_dir = base / "cache"

            first = SpectralSmilesDataset(
                data_dir=str(tokenized_dir),
                smiles_tokenizer=smiles_tokenizer,
                spectral_tokenizer=nmr_tokenizer,
                split="train",
                max_smiles_len=64,
                max_nmr_len=64,
                pretokenize=True,
                use_disk_cache=True,
                write_disk_cache=True,
                cache_dir=str(cache_dir),
                ir_cache_mode="pt",
                ir_cache_dtype="float32",
            )
            self.assertFalse(first.token_cache_hit)
            self.assertIsNotNone(first.token_cache_path)
            self.assertTrue(first.token_cache_path.exists())
            self.assertFalse(first.ir_cache_hit)
            self.assertIsNotNone(first.ir_cache_path)
            self.assertTrue(first.ir_cache_path.exists())

            first_tgt, first_ir, first_nmr = first[0]

            second = SpectralSmilesDataset(
                data_dir=str(tokenized_dir),
                smiles_tokenizer=smiles_tokenizer,
                spectral_tokenizer=nmr_tokenizer,
                split="train",
                max_smiles_len=64,
                max_nmr_len=64,
                pretokenize=True,
                use_disk_cache=True,
                write_disk_cache=True,
                cache_dir=str(cache_dir),
                ir_cache_mode="pt",
                ir_cache_dtype="float32",
            )
            self.assertTrue(second.token_cache_hit)
            self.assertTrue(second.ir_cache_hit)
            second_tgt, second_ir, second_nmr = second[0]

            self.assertTrue(torch.equal(first_tgt, second_tgt))
            self.assertTrue(torch.equal(first_nmr, second_nmr))
            self.assertTrue(first_ir is None or second_ir is None or torch.equal(first_ir, second_ir))


if __name__ == "__main__":
    unittest.main()
