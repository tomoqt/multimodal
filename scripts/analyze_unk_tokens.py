import os
from models.smiles_tokenizer import SmilesTokenizer
from pathlib import Path

# Initialize tokenizer the same way as in training script
current_dir = os.path.dirname(os.path.realpath(__file__))
vocab_path = os.path.join(current_dir, 'vocab.txt')
tokenizer = SmilesTokenizer(vocab_file=vocab_path)

def analyze_smiles_tokens(data_dir, split='train'):
    """
    Analyze where UNK tokens are being inserted in SMILES sequences.
    """
    # Load target sequences (SMILES)
    tgt_path = Path(data_dir) / f"tgt-{split}.txt"
    with open(tgt_path) as f:
        # Remove any extra spaces when reading the SMILES
        smiles_sequences = [line.strip().replace(" ", "") for line in f]

    print(f"\nAnalyzing {len(smiles_sequences)} SMILES sequences from {split} set...")
    
    # Track statistics
    sequences_with_unk = []
    unk_token_id = tokenizer.unk_token_id
    
    for idx, smiles in enumerate(smiles_sequences):
        # Tokenize and convert to ids
        tokens = tokenizer.tokenize(smiles)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Check if sequence contains UNK token
        if unk_token_id in token_ids:
            # Find which tokens were converted to UNK
            unk_positions = [i for i, tid in enumerate(token_ids) if tid == unk_token_id]
            unk_tokens = [tokens[i] for i in unk_positions]
            
            sequences_with_unk.append({
                'index': idx,
                'smiles': smiles,  # This will now be without spaces
                'tokens': tokens,
                'unk_positions': unk_positions,
                'unk_tokens': unk_tokens
            })
    
    # Print results
    print(f"\nFound {len(sequences_with_unk)} sequences containing UNK tokens")
    
    if sequences_with_unk:
        print("\nFirst 10 examples of sequences with UNK tokens:")
        for i, example in enumerate(sequences_with_unk[:10]):
            print(f"\nExample {i+1}:")
            print(f"Index: {example['index']}")
            print(f"SMILES: {example['smiles']}")
            print(f"Tokens: {example['tokens']}")
            print(f"UNK positions: {example['unk_positions']}")
            print(f"Problematic tokens: {example['unk_tokens']}")
        
        # Analyze most common tokens that become UNK
        all_unk_tokens = []
        for example in sequences_with_unk:
            all_unk_tokens.extend(example['unk_tokens'])
        
        from collections import Counter
        unk_counter = Counter(all_unk_tokens)
        
        print("\nMost common tokens converted to UNK:")
        for token, count in unk_counter.most_common(10):
            print(f"'{token}': {count} times")

if __name__ == '__main__':
    # Use the same data directory as in training script
    data_dir = "tokenized_baseline/data"
    
    # Analyze train, val, and test sets
    for split in ['train', 'val', 'test']:
        analyze_smiles_tokens(data_dir, split) 