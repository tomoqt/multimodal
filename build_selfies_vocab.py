import os
from models.selfies_tokenizer import SelfiesTokenizer, BasicSelfiesTokenizer, load_vocab
from tqdm import tqdm
import selfies as sf

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
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab_file = SelfiesTokenizer.build_vocab_from_smiles(smiles_list)
    print(f"\nCreated vocabulary file: {vocab_file}")
    
    # Verify the vocabulary
    print("\nVerifying vocabulary...")
    # Load vocab directly instead of using the full tokenizer
    vocab = load_vocab(vocab_file)
    print(f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Test tokenization on a few examples using BasicSelfiesTokenizer
    print("\nTesting tokenization on sample SMILES:")
    basic_tokenizer = BasicSelfiesTokenizer()
    for smiles in smiles_list[:5]:
        try:
            # Clean up SMILES and convert to SELFIES
            clean_smiles = ''.join(smiles.split())
            selfies = sf.encoder(clean_smiles)
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