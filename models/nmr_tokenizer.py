class NMRTokenizer:
    def __init__(self):
        # Special tokens
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "1HNMR": 2,
            "13CNMR": 3,
            "|": 4,
            "s": 5,
            "d": 6,
            "t": 7,
            "q": 8,
            "m": 9,
            "J": 10,
            "H": 11,
        }
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Initialize dynamic vocabulary for numerical values
        self.num_bins = 100  # Number of bins for chemical shifts
        self.shift_min = 0
        self.shift_max = 220  # Cover both 1H and 13C ranges
        self.bin_size = (self.shift_max - self.shift_min) / self.num_bins
        
        # Add numerical token IDs after special tokens
        self.shift_start_id = len(self.special_tokens)
        
    def _bin_value(self, value: float) -> int:
        """Convert a chemical shift value to a token ID"""
        bin_idx = int((value - self.shift_min) / self.bin_size)
        bin_idx = max(0, min(bin_idx, self.num_bins - 1))
        return bin_idx + self.shift_start_id
    
    def encode(self, nmr_string: str) -> list[int]:
        """Convert NMR string to token IDs"""
        tokens = []
        parts = nmr_string.split()
        
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                try:
                    value = float(part)
                    tokens.append(self._bin_value(value))
                except ValueError:
                    tokens.append(self.special_tokens["[UNK]"])
                    
        return tokens
    
    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to NMR string"""
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tokens.append(self.id_to_token[tid])
            elif tid >= self.shift_start_id:
                bin_idx = tid - self.shift_start_id
                value = self.shift_min + (bin_idx + 0.5) * self.bin_size
                tokens.append(f"{value:.2f}")
            else:
                tokens.append("[UNK]")
        
        return " ".join(tokens)
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens and bins"""
        return len(self.special_tokens) + self.num_bins 