import collections
import os
import re
from typing import List, Optional
from transformers import BertTokenizer
from logging import getLogger

logger = getLogger(__name__)

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

class BasicSmilesTokenizer(object):
    """Run basic SMILES tokenization using a regex pattern developed by Schwaller et. al."""
    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str) -> List[str]:
        return [token for token in self.regex.findall(text)]

class SmilesTokenizer(BertTokenizer):
    """
    A specialized SMILES tokenizer that inherits from HuggingFace BertTokenizer
    but uses a custom SMILES regex for tokenization.
    """
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, vocab_file: str = '', **kwargs):
        super().__init__(vocab_file, **kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocab file at path '{vocab_file}'."
            )

        # Load the vocab
        self.vocab = load_vocab(vocab_file)
        self.highest_unused_index = max([
            i for i, v in enumerate(self.vocab.keys())
            if v.startswith("[unused")
        ]) if any(v.startswith("[unused") for v in self.vocab.keys()) else 0

        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])

        # Basic tokenizer for SMILES
        self.basic_tokenizer = BasicSmilesTokenizer()

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    def _tokenize(self, text: str, max_seq_length: int = 512, **kwargs) -> List[str]:
        max_len_single_sentence = max_seq_length - 2  # typically for BERT
        split_tokens = self.basic_tokenizer.tokenize(text)[:max_len_single_sentence]
        return split_tokens

    def _convert_token_to_id(self, token: str):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]):
        out_string: str = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def add_special_tokens_ids_single_sequence(self, token_ids: List[Optional[int]]):
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens: List[str]):
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_ids_sequence_pair(self, token_ids_0: List[Optional[int]], token_ids_1: List[Optional[int]]):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(self, token_ids: List[Optional[int]], length: int, right: bool = True):
        padding = [self.pad_token_id] * (length - len(token_ids))
        if right:
            return token_ids + padding
        else:
            return padding + token_ids

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive. "
                        "Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,) 