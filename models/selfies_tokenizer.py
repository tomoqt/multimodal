import collections
import os
import re
from typing import List, Optional, Tuple
import selfies as sf
from transformers import PreTrainedTokenizer
from .smiles_utils import safe_selfies_conversion, canonicalize_smiles, process_smiles_to_selfies

VOCAB_FILES_NAMES = {"vocab_file": "selfies_vocab.txt"}

# Define regex pattern for SELFIES tokens
SELFIES_REGEX_PATTERN = r"(\[[^\]]+]|[^\[\]])"

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

class BasicSelfiesTokenizer:
    """Run basic SELFIES tokenization using regex pattern"""
    def __init__(self, regex_pattern: str = SELFIES_REGEX_PATTERN):
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize SELFIES string using regex pattern.
        Handles both direct SELFIES input and SMILES conversion.
        """
        if not (text.startswith('[') and text.endswith(']')):
            # Convert SMILES to SELFIES
            text = safe_selfies_conversion(text)
            if text is None:
                return []

        return [token for token in self.regex.findall(text) if token.strip()]

class SelfiesTokenizer(PreTrainedTokenizer):
    """
    A SELFIES tokenizer that inherits from HuggingFace PreTrainedTokenizer.
    Uses regex-based tokenization for SELFIES strings.
    """
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self, 
        vocab_file: str = '', 
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        # Load vocabulary first
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocab file at path '{vocab_file}'.")

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])
        
        # Initialize basic tokenizer
        self.basic_tokenizer = BasicSelfiesTokenizer()

        # Now initialize parent class
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )

    def get_vocab(self):
        """Returns vocab as a dict"""
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using regex-based SELFIES tokenizer
        """
        # Convert to SELFIES if needed using standardized function
        if not (text.startswith('[') and text.endswith(']')):
            text = process_smiles_to_selfies(text)
            if text is None:
                return []
        
        tokens = self.basic_tokenizer.tokenize(text)
        # Debug print vocabulary coverage
        for token in tokens:
            if token not in self.vocab:
                print(f"WARNING: Token not in vocabulary: '{token}'")
        return tokens

    def _convert_token_to_id(self, token: str):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of SELFIES tokens back to a SMILES string.
        """
        try:
            # Join tokens into SELFIES string
            selfies_str = ''.join(tokens)
            # Convert back to SMILES
            smiles = sf.decoder(selfies_str)
            return smiles
        except Exception as e:
            print(f"Error converting tokens to string: {e}")
            return ""

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save the tokenizer vocabulary to a file."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, _ in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                writer.write(token + "\n")

        return (vocab_file,)

    def get_special_tokens_mask(
        self, token_ids: List[int], already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence where 1 indicates a special token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids=token_ids,
                already_has_special_tokens=True
            )

        return [1 if token in [self.cls_token_id, self.sep_token_id, self.pad_token_id] else 0 
                for token in token_ids]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence by adding special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs for sequences.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    @classmethod
    def build_vocab_from_smiles(cls, smiles_list: List[str], special_tokens: List[str] = None) -> str:
        """
        Build a vocabulary file from a list of SMILES strings.
        Uses regex-based tokenization after converting to SELFIES.
        """
        if special_tokens is None:
            special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

        basic_tokenizer = BasicSelfiesTokenizer()
        
        vocab = set()
        for smiles in smiles_list:
            try:
                selfies = safe_selfies_conversion(smiles)
                if selfies is None:
                    continue
                tokens = basic_tokenizer.tokenize(selfies)
                vocab.update(tokens)
            except Exception as e:
                print(f"Error processing SMILES: {smiles}")
                print(f"Error message: {e}")
                continue

        # Add special tokens at the beginning
        vocab_list = special_tokens + sorted(list(vocab))
        
        # Save vocabulary to file
        vocab_file = "selfies_vocab.txt"
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token in vocab_list:
                f.write(token + "\n")

        print(f"Created vocabulary with {len(vocab_list)} tokens")
        print(f"- {len(special_tokens)} special tokens")
        print(f"- {len(vocab)} SELFIES tokens")
        
        return vocab_file 