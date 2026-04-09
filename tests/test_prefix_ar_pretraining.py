import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.multimodal_prefix_to_smiles import MultiModalPrefixToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from tests.helpers import materialize_tokenized_dataset
from training.core_dataset import SpectralSmilesDataset, collate_spectral_smiles
from training.core_train import train_step


class PrefixARPretrainingStepTests(unittest.TestCase):
    def test_single_train_step_runs_and_updates_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenized_dir, nmr_tokenizer = materialize_tokenized_dataset(Path(tmpdir), num_rows=28)
            smiles_tokenizer = SmilesTokenizer(vocab_file="/Users/tensorqt/smiles_decoding/training/vocab.txt")

            dataset = SpectralSmilesDataset(
                data_dir=str(tokenized_dir),
                smiles_tokenizer=smiles_tokenizer,
                spectral_tokenizer=nmr_tokenizer,
                split="train",
                max_smiles_len=64,
                max_nmr_len=64,
            )
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

            model = MultiModalPrefixToSMILESModel(
                smiles_vocab_size=len(smiles_tokenizer),
                nmr_vocab_size=max(nmr_tokenizer.values()) + 1,
                max_seq_length=64,
                max_nmr_length=64,
                max_memory_length=32,
                embed_dim=64,
                num_heads=4,
                num_layers=2,
                dropout=0.0,
                ir_encoder_type="regular",
                ir_as_prompt=False,
                verbose=False,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            batch = next(iter(loader))
            before = model.decoder.out.weight.detach().clone()
            loss_value = train_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                pad_token_id=smiles_tokenizer.pad_token_id,
                device=torch.device("cpu"),
            )

            self.assertTrue(torch.isfinite(torch.tensor(loss_value)))
            after = model.decoder.out.weight.detach()
            self.assertGreater(torch.norm(after - before).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
