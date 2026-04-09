import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectralSmilesDataset(Dataset):
    """
    Minimal dataset for pretraining from tokenized text + IR arrays.

    Expected files under ``data_dir``:
    - ``src-{split}.txt`` : tokenized NMR/source sequence
    - ``tgt-{split}.txt`` : target SMILES sequence
    - ``ir-{split}.npy``  : optional float32 memmap of IR spectra
    """

    def __init__(
        self,
        data_dir: str,
        smiles_tokenizer,
        spectral_tokenizer: Dict[str, int],
        split: str = "train",
        max_smiles_len: int = 256,
        max_nmr_len: int = 256,
        pretokenize: bool = True,
        preload_ir: bool = False,
        use_disk_cache: bool = True,
        write_disk_cache: bool = True,
        cache_dir: Optional[str] = None,
        ir_cache_mode: str = "none",
        ir_cache_dtype: str = "float32",
        return_index: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.smiles_tokenizer = smiles_tokenizer
        self.spectral_tokenizer = spectral_tokenizer
        self.max_smiles_len = max_smiles_len
        self.max_nmr_len = max_nmr_len
        self.pretokenize = pretokenize
        self.preload_ir = preload_ir
        self.use_disk_cache = use_disk_cache
        self.write_disk_cache = write_disk_cache
        self.cache_dir = Path(cache_dir) if cache_dir else (self.data_dir.parent / "cache")
        self.ir_cache_mode = str(ir_cache_mode).lower()
        self.ir_cache_dtype = str(ir_cache_dtype).lower()
        self.return_index = bool(return_index)
        if self.ir_cache_mode not in {"none", "pt"}:
            raise ValueError(f"Unsupported ir_cache_mode='{ir_cache_mode}'. Use 'none' or 'pt'.")
        if self.ir_cache_dtype not in {"float32", "float16"}:
            raise ValueError(f"Unsupported ir_cache_dtype='{ir_cache_dtype}'. Use 'float32' or 'float16'.")

        src_path = self.data_dir / f"src-{split}.txt"
        tgt_path = self.data_dir / f"tgt-{split}.txt"
        if not src_path.exists() or not tgt_path.exists():
            raise FileNotFoundError(f"Missing dataset split files under {self.data_dir} for split='{split}'")
        self.src_path = src_path
        self.tgt_path = tgt_path

        with src_path.open("r", encoding="utf-8") as f:
            self.sources = [line.strip() for line in f]
        with tgt_path.open("r", encoding="utf-8") as f:
            self.targets = [line.strip().replace(" ", "") for line in f]

        if len(self.sources) != len(self.targets):
            raise ValueError("Source and target lengths do not match.")

        ir_path = self.data_dir / f"ir-{split}.npy"
        self.ir_path = ir_path if ir_path.exists() else None
        self.ir_data: Optional[np.memmap] = None
        self.ir_tensor: Optional[torch.Tensor] = None
        self.ir_cache_path: Optional[Path] = None
        self.ir_cache_hit = False
        if self.ir_path is not None:
            loaded_ir = False
            if self.ir_cache_mode == "pt" and self.use_disk_cache:
                loaded_ir = self._try_load_ir_cache(split)
            if not loaded_ir:
                self.ir_data = self._load_ir_memmap()
                if self.preload_ir or self.ir_cache_mode == "pt":
                    self._materialize_ir_tensor(self.ir_cache_dtype if self.ir_cache_mode == "pt" else "float32")
                    if self.ir_cache_mode == "pt" and self.use_disk_cache and self.write_disk_cache:
                        self._save_ir_cache()

        self.smiles_pad_id = self.smiles_tokenizer.pad_token_id
        self.nmr_pad_id = self.spectral_tokenizer.get("<PAD>", 0)
        self.nmr_unk_id = self.spectral_tokenizer.get("<UNK>", 1)
        self.target_tokens: Optional[torch.Tensor] = None
        self.nmr_tokens: Optional[torch.Tensor] = None
        self.token_cache_path: Optional[Path] = None
        self.token_cache_hit = False
        if self.pretokenize:
            loaded = False
            if self.use_disk_cache:
                loaded = self._try_load_token_cache(split)
            if not loaded:
                self._build_token_cache()
                if self.use_disk_cache and self.write_disk_cache:
                    self._save_token_cache()

    def _file_signature(self, path: Optional[Path]) -> Optional[Dict[str, str]]:
        if path is None or not path.exists():
            return None
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": str(stat.st_size),
            "mtime_ns": str(stat.st_mtime_ns),
        }

    def _spectral_tokenizer_fingerprint(self) -> str:
        hasher = hashlib.sha1()
        for token, idx in sorted(self.spectral_tokenizer.items()):
            hasher.update(token.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(str(idx).encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def _cache_key(self, split: str) -> str:
        vocab_file = getattr(self.smiles_tokenizer, "vocab_file", None)
        payload = {
            "version": 1,
            "split": split,
            "max_smiles_len": self.max_smiles_len,
            "max_nmr_len": self.max_nmr_len,
            "smiles_pad_id": self.smiles_pad_id,
            "nmr_pad_id": self.nmr_pad_id,
            "nmr_unk_id": self.nmr_unk_id,
            "src_sig": self._file_signature(self.src_path),
            "tgt_sig": self._file_signature(self.tgt_path),
            "ir_sig": self._file_signature(self.ir_path),
            "smiles_vocab_sig": self._file_signature(Path(vocab_file)) if vocab_file else None,
            "spectral_tokenizer_sig": self._spectral_tokenizer_fingerprint(),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(serialized).hexdigest()[:20]

    def _ir_cache_key(self, split: str) -> str:
        payload = {
            "version": 1,
            "split": split,
            "ir_sig": self._file_signature(self.ir_path),
            "ir_cache_dtype": self.ir_cache_dtype,
            "num_rows": len(self.sources),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(serialized).hexdigest()[:20]

    def _load_ir_memmap(self) -> np.memmap:
        if self.ir_path is None:
            raise FileNotFoundError("IR path is not available for this split.")
        raw_ir = np.memmap(self.ir_path, dtype="float32", mode="r", shape=None)
        if raw_ir.ndim == 1:
            if len(self.sources) == 0 or raw_ir.shape[0] % len(self.sources) != 0:
                raise ValueError("Cannot infer IR feature dimension from memmap shape.")
            feat_dim = raw_ir.shape[0] // len(self.sources)
            raw_ir = raw_ir.reshape(len(self.sources), feat_dim)
        return raw_ir

    def _materialize_ir_tensor(self, dtype_name: str) -> None:
        if self.ir_data is None:
            return
        np_dtype = np.float16 if dtype_name == "float16" else np.float32
        self.ir_tensor = torch.from_numpy(np.array(self.ir_data, dtype=np_dtype, copy=True))
        self.ir_data = None

    def _try_load_ir_cache(self, split: str) -> bool:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._ir_cache_key(split)
        cache_path = self.cache_dir / f"{split}_ir_{cache_key}.pt"
        self.ir_cache_path = cache_path
        if not cache_path.exists():
            return False
        try:
            tensor = torch.load(cache_path, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                return False
            if tensor.dim() != 2:
                return False
            if tensor.size(0) != len(self.sources):
                return False
            expected_dtype = torch.float16 if self.ir_cache_dtype == "float16" else torch.float32
            if tensor.dtype != expected_dtype:
                return False
            self.ir_tensor = tensor.contiguous()
            self.ir_data = None
            self.ir_cache_hit = True
            return True
        except Exception:
            return False

    def _save_ir_cache(self) -> None:
        if self.ir_tensor is None:
            return
        if self.ir_cache_path is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.ir_cache_path.with_name(
            f"{self.ir_cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            torch.save(self.ir_tensor.cpu(), tmp_path)
            os.replace(tmp_path, self.ir_cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _try_load_token_cache(self, split: str) -> bool:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._cache_key(split)
        cache_path = self.cache_dir / f"{split}_tokens_{cache_key}.pt"
        self.token_cache_path = cache_path
        if not cache_path.exists():
            return False
        try:
            blob = torch.load(cache_path, map_location="cpu")
            target_tokens = blob["target_tokens"]
            nmr_tokens = blob["nmr_tokens"]
            if target_tokens.size(0) != len(self.sources) or nmr_tokens.size(0) != len(self.sources):
                return False
            if target_tokens.size(1) != self.max_smiles_len or nmr_tokens.size(1) != self.max_nmr_len:
                return False
            self.target_tokens = target_tokens
            self.nmr_tokens = nmr_tokens
            self.sources = []
            self.targets = []
            self.token_cache_hit = True
            return True
        except Exception:
            return False

    def _save_token_cache(self) -> None:
        if self.target_tokens is None or self.nmr_tokens is None:
            return
        if self.token_cache_path is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.token_cache_path.with_name(
            f"{self.token_cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        payload = {
            "target_tokens": self.target_tokens.cpu(),
            "nmr_tokens": self.nmr_tokens.cpu(),
            "max_smiles_len": self.max_smiles_len,
            "max_nmr_len": self.max_nmr_len,
        }
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, self.token_cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _build_token_cache(self) -> None:
        n = len(self.sources)
        target_tensor = torch.full((n, self.max_smiles_len), self.smiles_pad_id, dtype=torch.long)
        nmr_tensor = torch.full((n, self.max_nmr_len), self.nmr_pad_id, dtype=torch.long)

        for idx, (source_text, target_smiles) in enumerate(zip(self.sources, self.targets)):
            target_ids = self.smiles_tokenizer.encode(
                target_smiles,
                add_special_tokens=True,
                max_length=self.max_smiles_len,
                truncation=True,
            )
            tgt_len = min(len(target_ids), self.max_smiles_len)
            if tgt_len:
                target_tensor[idx, :tgt_len] = torch.as_tensor(target_ids[:tgt_len], dtype=torch.long)

            nmr_ids = [
                self.spectral_tokenizer.get(tok, self.nmr_unk_id)
                for tok in source_text.split()
            ][: self.max_nmr_len]
            if nmr_ids:
                nmr_tensor[idx, : len(nmr_ids)] = torch.as_tensor(nmr_ids, dtype=torch.long)

        self.target_tokens = target_tensor
        self.nmr_tokens = nmr_tensor
        self.sources = []
        self.targets = []

    def __len__(self) -> int:
        if self.target_tokens is not None:
            return int(self.target_tokens.size(0))
        return len(self.sources)

    def __getitem__(self, idx: int):
        if self.target_tokens is not None and self.nmr_tokens is not None:
            target_tokens = self.target_tokens[idx]
            nmr_tokens = self.nmr_tokens[idx]
        else:
            target_smiles = self.targets[idx]
            target_ids = self.smiles_tokenizer.encode(
                target_smiles,
                add_special_tokens=True,
                max_length=self.max_smiles_len,
                truncation=True,
            )
            target_tokens = torch.tensor(target_ids, dtype=torch.long)

            nmr_ids = [
                self.spectral_tokenizer.get(tok, self.nmr_unk_id)
                for tok in self.sources[idx].split()
            ][: self.max_nmr_len]
            nmr_tokens = torch.tensor(nmr_ids, dtype=torch.long)

        ir_tensor: Optional[torch.Tensor] = None
        if self.ir_tensor is not None:
            ir_tensor = self.ir_tensor[idx]
        elif self.ir_data is not None:
            ir_tensor = torch.tensor(self.ir_data[idx], dtype=torch.float32)

        if self.return_index:
            return target_tokens, ir_tensor, nmr_tokens, int(idx)
        return target_tokens, ir_tensor, nmr_tokens

    def __del__(self) -> None:
        if hasattr(self, "ir_data") and self.ir_data is not None:
            del self.ir_data
        if hasattr(self, "ir_tensor") and self.ir_tensor is not None:
            del self.ir_tensor


def collate_spectral_smiles(
    batch,
    smiles_pad_token_id: int = 0,
    nmr_pad_token_id: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Pad variable-length sequences when needed; fast-path stacks fixed-shape batches."""
    has_indices = len(batch[0]) == 4
    if has_indices:
        targets, ir_list, nmr_list, indices = zip(*batch)
    else:
        targets, ir_list, nmr_list = zip(*batch)

    same_tgt_shape = all(seq.size(0) == targets[0].size(0) for seq in targets)
    same_nmr_shape = all(seq.size(0) == nmr_list[0].size(0) for seq in nmr_list)

    if same_tgt_shape:
        target_batch = torch.stack(targets, dim=0)
    else:
        max_tgt = max(seq.size(0) for seq in targets)
        padded_targets = []
        for seq in targets:
            if seq.size(0) < max_tgt:
                pad = torch.full((max_tgt - seq.size(0),), smiles_pad_token_id, dtype=torch.long)
                seq = torch.cat([seq, pad], dim=0)
            padded_targets.append(seq)
        target_batch = torch.stack(padded_targets, dim=0)

    if same_nmr_shape:
        nmr_batch = torch.stack(nmr_list, dim=0)
    else:
        max_nmr = max(seq.size(0) for seq in nmr_list)
        padded_nmr = []
        for seq in nmr_list:
            if seq.size(0) < max_nmr:
                pad = torch.full((max_nmr - seq.size(0),), nmr_pad_token_id, dtype=torch.long)
                seq = torch.cat([seq, pad], dim=0)
            padded_nmr.append(seq)
        nmr_batch = torch.stack(padded_nmr, dim=0)

    ir_batch: Optional[torch.Tensor] = None
    if all(t is not None for t in ir_list):
        ir_batch = torch.stack([t for t in ir_list if t is not None], dim=0)

    if has_indices:
        return target_batch, ir_batch, nmr_batch, torch.tensor(indices, dtype=torch.long)
    return target_batch, ir_batch, nmr_batch
