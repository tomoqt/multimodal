import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.create_tokenized_dataset_faster import process_parquet_file
from tests.helpers import build_raw_dataframe, materialize_tokenized_dataset


class RawPipelineTests(unittest.TestCase):
    def test_process_parquet_file_from_raw_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_df = build_raw_dataframe(num_rows=12, spectrum_points=32)
            parquet_file = tmp / "raw.parquet"
            raw_df.to_parquet(parquet_file, index=False)

            processed = process_parquet_file(
                parquet_file=parquet_file,
                h_nmr=True,
                c_nmr=True,
                ir=True,
                pos_msms=False,
                neg_msms=False,
                formula=True,
                original_x=np.linspace(400, 4000, 32),
                tokenize_ir=False,
            )

            self.assertEqual(len(processed), 12)
            self.assertIn("source", processed.columns)
            self.assertIn("target", processed.columns)
            self.assertIn("ir_data", processed.columns)
            self.assertTrue(processed.iloc[0]["target"])

    def test_materialize_tokenized_dataset_outputs_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenized_dir, _ = materialize_tokenized_dataset(Path(tmpdir), num_rows=20)
            expected = [
                "src-train.txt",
                "tgt-train.txt",
                "ir-train.npy",
                "src-val.txt",
                "tgt-val.txt",
                "ir-val.npy",
                "src-test.txt",
                "tgt-test.txt",
                "ir-test.npy",
            ]
            for name in expected:
                self.assertTrue((tokenized_dir / name).exists(), f"Missing {name}")


if __name__ == "__main__":
    unittest.main()
