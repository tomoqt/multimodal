import argparse
import os
import sys
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    """Parse command-line arguments for pre-tokenization."""
    parser = argparse.ArgumentParser(description="Pre-tokenize data for SFT training.")
    parser.add_argument("--model_name", type=str, required=True, help="The pre-trained model to use for tokenization (e.g., 'Qwen/Qwen3-0.6B').")
    parser.add_argument("--data_dir", type=str, default="data/reshaped_tokenized_data/data", help="Directory with src-*.txt and tgt-*.txt files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the tokenized datasets.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    return parser.parse_args()

def create_templated_dataset(data_dir, split):
    """Creates a dataset with a prompt template for NMR to SMILES prediction."""
    prompt_template = "Given the following NMR data, predict the corresponding SMILES string.\n\nNMR data: {nmr_data}\n\nSMILES:"
    
    src_path = os.path.join(data_dir, f"src-{split}.txt")
    tgt_path = os.path.join(data_dir, f"tgt-{split}.txt")

    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        print(f"Error: Data files not found for split '{split}' in {data_dir}", file=sys.stderr)
        sys.exit(1)

    with open(src_path, "r") as f:
        nmr_lines = [line.strip() for line in f]
    with open(tgt_path, "r") as f:
        smiles_lines = [line.strip().replace(" ", "") for line in f]

    data = []
    for nmr, smiles in zip(nmr_lines, smiles_lines):
        prompt = prompt_template.format(nmr_data=nmr)
        text = prompt + " " + smiles
        data.append({"text": text})

    return Dataset.from_dict({"text": [item["text"] for item in data]})

def main():
    """Main function to run the pre-tokenization."""
    args = parse_args()
    
    if os.path.exists(os.path.join(args.output_dir, "train")) and os.path.exists(os.path.join(args.output_dir, "val")):
        print(f"Tokenized data already exists in {args.output_dir}. Skipping.")
        try:
            # Quick check if it's usable
            load_from_disk(os.path.join(args.output_dir, "train"))
            load_from_disk(os.path.join(args.output_dir, "val"))
            print("Existing data looks valid.")
            return
        except Exception as e:
            print(f"Found existing data, but it failed to load: {e}. Re-tokenizing.")

    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # SFTTrainer expects the input sequences to be tokenized, including the EOS token.
        return tokenizer(
            [text + tokenizer.eos_token for text in examples["text"]],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    for split in ["train", "val"]:
        print(f"Processing {split} split...")
        raw_dataset = create_templated_dataset(args.data_dir, split)
        
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=["text"],
            desc=f"Tokenizing {split} split"
        )
        
        output_path = os.path.join(args.output_dir, split)
        print(f"Saving tokenized {split} dataset to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        tokenized_dataset.save_to_disk(output_path)
        
    print(f"Pre-tokenization complete. Saved to {args.output_dir}")

if __name__ == "__main__":
    main() 