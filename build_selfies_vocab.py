import os
from models.selfies_tokenizer import SelfiesTokenizer, BasicSelfiesTokenizer, load_vocab
from tqdm import tqdm
import selfies as sf
from models.smiles_utils import safe_selfies_conversion, process_smiles_to_selfies

def build_vocab_from_data(data_dir):
    """Build SELFIES vocabulary from data files"""
    # Read SMILES from data files
    smiles_list = []
    for split in ['train', 'val', 'test']:
        tgt_file = os.path.join(data_dir, f"tgt-{split}.txt")
        if os.path.exists(tgt_file):
            print(f"Reading {tgt_file}...")
            with open(tgt_file) as f:
                split_smiles = [line.strip() for line in f]
                print(f"Found {len(split_smiles)} SMILES in {split} set")
                smiles_list.extend(split_smiles)
    
    print(f"\nTotal SMILES collected: {len(smiles_list)}")
    print("Sample SMILES strings:")
    for smiles in smiles_list[:5]:
        print(f"  {smiles}")
    
    # Build vocabulary by collecting all unique SELFIES tokens
    print("\nBuilding vocabulary...")
    basic_tokenizer = BasicSelfiesTokenizer()
    unique_tokens = set()
    
    # Add special tokens first
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    unique_tokens.update(special_tokens)
    
    # Process each SMILES and collect unique SELFIES tokens
    for smiles in tqdm(smiles_list, desc="Processing SMILES"):
        try:
            # Convert SMILES to SELFIES using the safe conversion function
            selfies = process_smiles_to_selfies(smiles)
            if selfies is None:
                continue
                
            # Tokenize the SELFIES string
            tokens = basic_tokenizer.tokenize(selfies)
            unique_tokens.update(tokens)
        except Exception as e:
            print(f"Error processing SMILES: {smiles}")
            print(f"Error message: {e}")
            continue
    
    # Sort tokens (keeping special tokens at the start)
    vocab_list = special_tokens + sorted(list(unique_tokens - set(special_tokens)))
    
    # Save vocabulary to file
    vocab_file = "selfies_vocab.txt"
    with open(vocab_file, "w", encoding="utf-8") as f:
        for token in vocab_list:
            f.write(token + "\n")

    print(f"\nCreated vocabulary with {len(vocab_list)} tokens")
    print(f"- {len(special_tokens)} special tokens")
    print(f"- {len(unique_tokens) - len(special_tokens)} SELFIES tokens")
    
    # Verify the vocabulary
    print("\nVerifying vocabulary...")
    vocab = load_vocab(vocab_file)
    print(f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Test tokenization on a few examples
    print("\nTesting tokenization on sample SMILES:")
    for smiles in smiles_list[:5]:
        try:
            # Convert SMILES to SELFIES
            selfies = process_smiles_to_selfies(smiles)
            if selfies is None:
                print(f"\nError converting SMILES: {smiles}")
                continue
                
            tokens = basic_tokenizer.tokenize(selfies)
            print(f"\nSMILES: {smiles}")
            print(f"SELFIES: {selfies}")
            print(f"Tokens: {tokens}")
            
            # Check if tokens are in vocabulary
            missing_tokens = [t for t in tokens if t not in vocab]
            if missing_tokens:
                print(f"Warning: Tokens not in vocabulary: {missing_tokens}")
        except Exception as e:
            print(f"\nError tokenizing: {smiles}")
            print(f"Error message: {e}")
    
    return vocab_file

if __name__ == '__main__':
    data_dir = "tokenized_baseline/data"  # Update this path
    vocab_file = build_vocab_from_data(data_dir) 