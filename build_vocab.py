import os
from pathlib import Path
import json

def build_vocabulary(tokenized_dir: Path, 
                     output_vocab_file: Path, 
                     add_special_tokens: bool = True,
                     save_json: bool = True):
    """
    Build a vocabulary file from tokenized text files in `tokenized_dir`.
    Expects files like src-train.txt, tgt-train.txt, src-val.txt, tgt-val.txt, etc.

    Args:
        tokenized_dir: Path to the directory containing tokenized *.txt files.
        output_vocab_file: Path to the output vocabulary file (e.g. vocab.txt).
        add_special_tokens: Whether to add special tokens like <BOS>, <EOS>, etc.
        save_json: Whether to also save the vocabulary as a JSON token-to-id mapping.
    """
    # 1) Gather all tokenized files
    token_files = list(tokenized_dir.glob("*.txt"))

    # 2) Accumulate unique tokens in a set
    vocab_set = set()

    # 3) Read each line from each file and split into tokens
    for token_file in token_files:
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                for token in tokens:
                    vocab_set.add(token)

    # 4) Add special tokens (optional)
    special_tokens = []
    if add_special_tokens:
        # Basic special tokens
        special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
            # NMR-specific tokens
            "1HNMR",  # H-NMR indicator
            "13CNMR", # C-NMR indicator
            "H",      # For nH notation
            "J",      # For J-coupling constants
            "|",      # Separator in NMR data
            # Common NMR multiplicity categories
            "s", "d", "t", "q", "m", "dd", "dt", "td", "tt", "ddd",
            # MS/MS specific tokens
            "E0Pos", "E1Pos", "E2Pos",  # Positive mode energies
            "E0Neg", "E1Neg", "E2Neg",  # Negative mode energies
            # IR specific token
            "IR"
        ]
    
    # 5) Combine special tokens + sorted unique tokens
    #    Sorting keeps the vocabulary consistent across runs
    final_vocab = special_tokens + sorted(vocab_set)

    # 6) Write out to a file
    with open(output_vocab_file, "w", encoding="utf-8") as f:
        for token in final_vocab:
            f.write(token + "\n")

    # 7) Optionally save as JSON mapping
    if save_json:
        token_to_id = {token: idx for idx, token in enumerate(final_vocab)}
        json_path = output_vocab_file.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(token_to_id, f, indent=2)
        print(f"Token-to-ID mapping saved to {json_path}")

    print(f"Vocabulary saved to {output_vocab_file}")
    print(f"Total vocabulary size: {len(final_vocab)}")


if __name__ == "__main__":
    # Example usage:
    tokenized_data_dir = Path("tokenized_baseline/data")
    output_vocab = Path("tokenized_baseline/vocab.txt")

    build_vocabulary(
        tokenized_dir=tokenized_data_dir,
        output_vocab_file=output_vocab,
        add_special_tokens=True,
        save_json=True
    )
