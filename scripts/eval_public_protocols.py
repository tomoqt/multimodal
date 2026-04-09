#!/usr/bin/env python3
"""
Evaluate a checkpoint under public-facing protocol variants for the
Multimodal Spectroscopic Dataset.

Supported protocol presets:
- alberts_like: formula + 1H-NMR + 13C-NMR, no IR
- nmiracle_like: IR + 1H-NMR + 13C-NMR, no formula
- ours_full: formula + IR + 1H-NMR + 13C-NMR

This script reports:
- top-k exact match
- top-k approximate enantiomer-aware match
- top-1 validity
- top-1 similarity metrics

Important: these results are only directly comparable to published work if the
same split and equivalence rules are used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

from inference.inference import ModelInference
from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from training.core_dataset import SpectralSmilesDataset


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    keep_formula: bool
    keep_1h: bool
    keep_13c: bool
    keep_ir: bool


PROTOCOLS: Dict[str, ProtocolSpec] = {
    "alberts_like": ProtocolSpec(
        name="alberts_like",
        keep_formula=True,
        keep_1h=True,
        keep_13c=True,
        keep_ir=False,
    ),
    "nmiracle_like": ProtocolSpec(
        name="nmiracle_like",
        keep_formula=False,
        keep_1h=True,
        keep_13c=True,
        keep_ir=True,
    ),
    "ours_full": ProtocolSpec(
        name="ours_full",
        keep_formula=True,
        keep_1h=True,
        keep_13c=True,
        keep_ir=True,
    ),
}


def load_config(config_path: Optional[str]) -> Dict:
    config = {
        "model": {
            "max_seq_length": 128,
            "max_nmr_length": 256,
            "max_memory_length": 128,
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "dropout": 0.1,
            "ir_encoder_type": "regular",
            "ir_as_prompt": False,
            "use_stablemax": False,
            "use_rmsnorm": False,
        },
        "data": {
            "tokenized_dir": "data/tokenized_baseline/data",
        },
    }
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        for section, section_val in user_cfg.items():
            if isinstance(section_val, dict) and section in config:
                config[section].update(section_val)
            else:
                config[section] = section_val
    return config


def auto_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_nmr_tokenizer(tokenized_dir: Path, vocab_json_override: Optional[str] = None) -> Dict[str, int]:
    vocab_json = Path(vocab_json_override) if vocab_json_override else (tokenized_dir.parent / "vocab.json")
    if not vocab_json.exists():
        raise FileNotFoundError(f"Missing NMR vocabulary: {vocab_json}")
    with vocab_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_source_tokens(source_text: str) -> Tuple[List[str], List[str], List[str]]:
    tokens = source_text.split()
    idx_1h = tokens.index("1HNMR") if "1HNMR" in tokens else -1
    idx_13c = tokens.index("13CNMR") if "13CNMR" in tokens else -1

    if idx_1h >= 0:
        formula_tokens = tokens[:idx_1h]
    elif idx_13c >= 0:
        formula_tokens = tokens[:idx_13c]
    else:
        formula_tokens = tokens

    one_h_tokens: List[str] = []
    if idx_1h >= 0:
        end_1h = idx_13c if idx_13c >= 0 else len(tokens)
        one_h_tokens = tokens[idx_1h:end_1h]

    carbon_tokens: List[str] = []
    if idx_13c >= 0:
        carbon_tokens = tokens[idx_13c:]

    return formula_tokens, one_h_tokens, carbon_tokens


def filter_source_text(source_text: str, spec: ProtocolSpec) -> str:
    formula_tokens, one_h_tokens, carbon_tokens = split_source_tokens(source_text)
    out: List[str] = []
    if spec.keep_formula:
        out.extend(formula_tokens)
    if spec.keep_1h:
        out.extend(one_h_tokens)
    if spec.keep_13c:
        out.extend(carbon_tokens)
    return " ".join(out).strip()


def encode_source_tokens(
    source_text: str,
    spectral_tokenizer: Dict[str, int],
    max_nmr_len: int,
) -> torch.Tensor:
    pad_id = spectral_tokenizer.get("<PAD>", 0)
    unk_id = spectral_tokenizer.get("<UNK>", 1)
    tokens = source_text.split()
    token_ids = [spectral_tokenizer.get(tok, unk_id) for tok in tokens][:max_nmr_len]
    if not token_ids:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(token_ids, dtype=torch.long)


def decode_token_ids(
    tokenizer: SmilesTokenizer,
    seq: Sequence[int],
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> str:
    cleaned: List[int] = []
    for token_id in seq:
        if token_id == eos_token_id:
            break
        if token_id in (bos_token_id, pad_token_id):
            continue
        cleaned.append(int(token_id))
    return tokenizer.decode(cleaned).replace(" ", "").strip()


def pad_nmr_batch(
    token_tensors: Sequence[torch.Tensor],
    pad_token_id: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not token_tensors:
        return None, None
    max_len = max((int(t.numel()) for t in token_tensors), default=0)
    if max_len == 0:
        return None, None

    batch_size = len(token_tensors)
    batch = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    for i, tokens in enumerate(token_tensors):
        length = int(tokens.numel())
        if length == 0:
            continue
        batch[i, :length] = tokens[:length]
        padding_mask[i, :length] = False
    return batch, padding_mask


def beam_search_nbest(
    model: MultiModalToSMILESModel,
    tokenizer: SmilesTokenizer,
    device: torch.device,
    nmr_tokens: Optional[torch.Tensor],
    ir_data: Optional[torch.Tensor],
    max_len: int,
    beam_width: int,
    n_best: int,
    length_penalty: float,
    ir_as_prompt: bool,
) -> List[str]:
    bos_token_id = tokenizer.cls_token_id
    eos_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    if nmr_tokens is not None and nmr_tokens.dim() == 1:
        nmr_tokens = nmr_tokens.unsqueeze(0)
    if ir_data is not None and ir_data.dim() == 1:
        ir_data = ir_data.unsqueeze(0)

    if nmr_tokens is not None:
        nmr_tokens = nmr_tokens.to(device)
    if ir_data is not None:
        ir_data = ir_data.to(device)

    if ir_as_prompt:
        if ir_data is None:
            memory = torch.zeros(1, 1, model.embed_dim, device=device)
        else:
            memory = model.ir_embed(ir_data.long())
    else:
        if ir_data is None:
            memory = torch.zeros(1, model.max_memory_length, model.embed_dim, device=device)
        else:
            memory = model.encoder(None, ir_data, None)

    beams: List[Tuple[List[int], float, bool]] = [([bos_token_id], 0.0, False)]
    for _ in range(max_len):
        candidates: List[Tuple[List[int], float, bool]] = []
        for seq_ids, score, ended in beams:
            if ended:
                candidates.append((seq_ids, score, True))
                continue

            seq_tensor = torch.tensor(seq_ids, dtype=torch.long, device=device).unsqueeze(0)
            logits = model.decoder(seq_tensor, memory, nmr_tokens=nmr_tokens)[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            top_vals, top_idx = torch.topk(log_probs, k=min(beam_width, log_probs.numel()))
            for lp, tok in zip(top_vals.tolist(), top_idx.tolist()):
                new_seq = seq_ids + [int(tok)]
                candidates.append((new_seq, score + float(lp), int(tok) == eos_token_id))

        def rank_key(item: Tuple[List[int], float, bool]) -> float:
            seq_ids, score, _ = item
            norm = (len(seq_ids) ** length_penalty) if length_penalty > 0 else 1.0
            return score / norm

        beams = sorted(candidates, key=rank_key, reverse=True)[:beam_width]
        if all(ended for _, _, ended in beams):
            break

    decoded: List[str] = []
    seen = set()
    for seq_ids, _, _ in sorted(beams, key=rank_key, reverse=True):
        smi = decode_token_ids(tokenizer, seq_ids, bos_token_id, eos_token_id, pad_token_id)
        if smi not in seen:
            seen.add(smi)
            decoded.append(smi)
        if len(decoded) >= n_best:
            break
    return decoded


def greedy_decode_one(
    model: MultiModalToSMILESModel,
    tokenizer: SmilesTokenizer,
    device: torch.device,
    nmr_tokens: Optional[torch.Tensor],
    ir_data: Optional[torch.Tensor],
    max_len: int,
    ir_as_prompt: bool,
) -> str:
    bos_token_id = tokenizer.cls_token_id
    eos_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    if nmr_tokens is not None and nmr_tokens.dim() == 1:
        nmr_tokens = nmr_tokens.unsqueeze(0)
    if ir_data is not None and ir_data.dim() == 1:
        ir_data = ir_data.unsqueeze(0)

    if nmr_tokens is not None:
        nmr_tokens = nmr_tokens.to(device)
    if ir_data is not None:
        ir_data = ir_data.to(device)

    if ir_as_prompt:
        if ir_data is None:
            memory = torch.zeros(1, 1, model.embed_dim, device=device)
        else:
            memory = model.ir_embed(ir_data.long())
    else:
        if ir_data is None:
            memory = torch.zeros(1, model.max_memory_length, model.embed_dim, device=device)
        else:
            memory = model.encoder(None, ir_data, None)

    seq_ids = [bos_token_id]
    for _ in range(max_len):
        seq_tensor = torch.tensor(seq_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = model.decoder(seq_tensor, memory, nmr_tokens=nmr_tokens)[0, -1, :]
        next_token = int(torch.argmax(logits).item())
        seq_ids.append(next_token)
        if next_token == eos_token_id:
            break

    return decode_token_ids(tokenizer, seq_ids, bos_token_id, eos_token_id, pad_token_id)


def greedy_decode_batch(
    model: MultiModalToSMILESModel,
    tokenizer: SmilesTokenizer,
    device: torch.device,
    nmr_tokens: Optional[torch.Tensor],
    ir_data: Optional[torch.Tensor],
    max_len: int,
    ir_as_prompt: bool,
    nmr_padding_mask: Optional[torch.Tensor] = None,
) -> List[str]:
    bos_token_id = tokenizer.cls_token_id
    eos_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    batch_size = 1
    if nmr_tokens is not None:
        batch_size = nmr_tokens.size(0)
        nmr_tokens = nmr_tokens.to(device)
    if ir_data is not None:
        batch_size = ir_data.size(0)
        ir_data = ir_data.to(device)
    if nmr_padding_mask is not None:
        nmr_padding_mask = nmr_padding_mask.to(device)

    if ir_as_prompt:
        if ir_data is None:
            memory = torch.zeros(batch_size, 1, model.embed_dim, device=device)
        else:
            memory = model.ir_embed(ir_data.long())
    else:
        if ir_data is None:
            memory = torch.zeros(batch_size, model.max_memory_length, model.embed_dim, device=device)
        else:
            memory = model.encoder(None, ir_data, None)

    seq = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    eos_fill = torch.full((batch_size,), eos_token_id, dtype=torch.long, device=device)

    for _ in range(max_len):
        logits = model.decoder(
            seq,
            memory,
            nmr_tokens=nmr_tokens,
            nmr_padding_mask=nmr_padding_mask,
        )[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        next_token = torch.where(finished, eos_fill, next_token)
        seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_token_id)
        if bool(finished.all()):
            break

    return [
        decode_token_ids(tokenizer, row.tolist(), bos_token_id, eos_token_id, pad_token_id)
        for row in seq
    ]


def canonical_smiles(mol: Chem.Mol, isomeric: bool = True) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def are_enantiomers_approx(pred_mol: Chem.Mol, target_mol: Chem.Mol) -> bool:
    if canonical_smiles(pred_mol, isomeric=False) != canonical_smiles(target_mol, isomeric=False):
        return False

    match = pred_mol.GetSubstructMatch(target_mol, useChirality=False)
    if not match:
        return False

    found_stereo = False
    for target_idx, pred_idx in enumerate(match):
        a_pred = pred_mol.GetAtomWithIdx(pred_idx)
        a_tgt = target_mol.GetAtomWithIdx(target_idx)
        tag_pred = a_pred.GetChiralTag()
        tag_tgt = a_tgt.GetChiralTag()
        if tag_pred == Chem.rdchem.ChiralType.CHI_UNSPECIFIED and tag_tgt == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            continue
        if tag_pred == Chem.rdchem.ChiralType.CHI_UNSPECIFIED or tag_tgt == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            return False
        found_stereo = True
        if tag_pred == tag_tgt:
            return False

    return found_stereo


def top1_metrics(pred_smiles: str, target_smiles: str) -> Dict[str, float | bool]:
    out: Dict[str, float | bool] = {
        "valid": False,
        "exact": False,
        "enantiomer_approx": False,
        "tanimoto": 0.0,
    }
    mol_pred = Chem.MolFromSmiles(pred_smiles)
    mol_target = Chem.MolFromSmiles(target_smiles)
    if mol_target is None:
        raise ValueError(f"Target SMILES failed RDKit parse: {target_smiles}")
    if mol_pred is None:
        return out

    out["valid"] = True
    can_pred = canonical_smiles(mol_pred, isomeric=True)
    can_target = canonical_smiles(mol_target, isomeric=True)
    out["exact"] = can_pred == can_target
    out["enantiomer_approx"] = bool(out["exact"]) or are_enantiomers_approx(mol_pred, mol_target)

    fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, radius=2, nBits=2048)
    fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, radius=2, nBits=2048)
    out["tanimoto"] = float(DataStructs.TanimotoSimilarity(fp_pred, fp_target))
    return out


def candidate_flags(candidate_smiles: str, target_smiles: str) -> Tuple[bool, bool]:
    mol_pred = Chem.MolFromSmiles(candidate_smiles)
    mol_target = Chem.MolFromSmiles(target_smiles)
    if mol_pred is None or mol_target is None:
        return False, False
    exact = canonical_smiles(mol_pred, isomeric=True) == canonical_smiles(mol_target, isomeric=True)
    enant = exact or are_enantiomers_approx(mol_pred, mol_target)
    return exact, enant


def evaluate_protocol(
    model: MultiModalToSMILESModel,
    tokenizer: SmilesTokenizer,
    dataset: SpectralSmilesDataset,
    spectral_tokenizer: Dict[str, int],
    device: torch.device,
    protocol: ProtocolSpec,
    max_examples: int,
    beam_width: int,
    n_best: int,
    max_len: int,
    length_penalty: float,
    ir_as_prompt: bool,
    decode_strategy: str,
    greedy_batch_size: int,
    beam_batch_size: int,
) -> Dict:
    n = len(dataset) if max_examples <= 0 else min(max_examples, len(dataset))
    topk_values = [1, 5, 10, 15]
    topk_exact = {k: 0 for k in topk_values}
    topk_enant = {k: 0 for k in topk_values}
    valid_top1 = 0
    tanimoto_sum = 0.0
    valid_pairs = 0
    sample_rows = []

    with torch.inference_mode():
        if decode_strategy == "greedy":
            pad_id = spectral_tokenizer.get("<PAD>", 0)
            for start in range(0, n, greedy_batch_size):
                end = min(start + greedy_batch_size, n)
                indices = list(range(start, end))
                filtered_sources = [
                    filter_source_text(dataset.sources[idx], protocol)
                    for idx in indices
                ]
                nmr_tensors = [
                    encode_source_tokens(
                        src,
                        spectral_tokenizer=spectral_tokenizer,
                        max_nmr_len=dataset.max_nmr_len,
                    )
                    for src in filtered_sources
                ]
                nmr_batch, nmr_padding_mask = pad_nmr_batch(nmr_tensors, pad_token_id=pad_id)

                ir_batch: Optional[torch.Tensor] = None
                if protocol.keep_ir:
                    ir_rows = []
                    for idx in indices:
                        _, raw_ir, _ = dataset[idx]
                        ir_rows.append(raw_ir)
                    ir_batch = torch.stack(ir_rows, dim=0) if ir_rows else None

                predictions = greedy_decode_batch(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    nmr_tokens=nmr_batch,
                    ir_data=ir_batch,
                    max_len=max_len,
                    ir_as_prompt=ir_as_prompt,
                    nmr_padding_mask=nmr_padding_mask,
                )

                for local_offset, idx in enumerate(indices):
                    target_smiles = dataset.targets[idx]
                    candidates = [predictions[local_offset]]
                    top1 = candidates[0]
                    m = top1_metrics(top1, target_smiles)
                    valid_top1 += int(bool(m["valid"]))
                    if bool(m["valid"]):
                        valid_pairs += 1
                        tanimoto_sum += float(m["tanimoto"])

                    exact_flags: List[bool] = []
                    enant_flags: List[bool] = []
                    for cand in candidates[: max(topk_values)]:
                        exact, enant = candidate_flags(cand, target_smiles)
                        exact_flags.append(exact)
                        enant_flags.append(enant)

                    for k in topk_values:
                        topk_exact[k] += int(any(exact_flags[:k]))
                        topk_enant[k] += int(any(enant_flags[:k]))

                    if idx < 5:
                        sample_rows.append(
                            {
                                "idx": idx,
                                "target": target_smiles,
                                "top1": top1,
                                "top5": candidates[:5],
                                "top1_valid": bool(m["valid"]),
                                "top1_exact": bool(m["exact"]),
                                "top1_enantiomer_approx": bool(m["enantiomer_approx"]),
                                "top1_tanimoto": float(m["tanimoto"]),
                            }
                        )

                print(f"[{protocol.name}] {end}/{n}", flush=True)
        else:
            inference = ModelInference(model, tokenizer, device=device, ir_as_prompt=ir_as_prompt)
            pad_id = spectral_tokenizer.get("<PAD>", 0)
            for start in range(0, n, beam_batch_size):
                end = min(start + beam_batch_size, n)
                indices = list(range(start, end))
                filtered_sources = [
                    filter_source_text(dataset.sources[idx], protocol)
                    for idx in indices
                ]
                nmr_tensors = [
                    encode_source_tokens(
                        src,
                        spectral_tokenizer=spectral_tokenizer,
                        max_nmr_len=dataset.max_nmr_len,
                    )
                    for src in filtered_sources
                ]
                nmr_batch, nmr_padding_mask = pad_nmr_batch(nmr_tensors, pad_token_id=pad_id)

                ir_batch: Optional[torch.Tensor] = None
                if protocol.keep_ir:
                    ir_rows = []
                    for idx in indices:
                        _, raw_ir, _ = dataset[idx]
                        ir_rows.append(raw_ir)
                    ir_batch = torch.stack(ir_rows, dim=0) if ir_rows else None

                batch_candidates = inference.beam_search_nbest(
                    nmr_tokens=nmr_batch,
                    ir_data=ir_batch,
                    max_len=max_len,
                    beam_width=beam_width,
                    length_penalty=length_penalty,
                    n_best=n_best,
                    nmr_padding_mask=nmr_padding_mask,
                )

                for local_offset, idx in enumerate(indices):
                    target_smiles = dataset.targets[idx]
                    candidates = batch_candidates[local_offset] if local_offset < len(batch_candidates) else []
                    top1 = candidates[0] if candidates else ""
                    m = top1_metrics(top1, target_smiles)
                    valid_top1 += int(bool(m["valid"]))
                    if bool(m["valid"]):
                        valid_pairs += 1
                        tanimoto_sum += float(m["tanimoto"])

                    exact_flags: List[bool] = []
                    enant_flags: List[bool] = []
                    for cand in candidates[: max(topk_values)]:
                        exact, enant = candidate_flags(cand, target_smiles)
                        exact_flags.append(exact)
                        enant_flags.append(enant)

                    for k in topk_values:
                        topk_exact[k] += int(any(exact_flags[:k]))
                        topk_enant[k] += int(any(enant_flags[:k]))

                    if idx < 5:
                        sample_rows.append(
                            {
                                "idx": idx,
                                "target": target_smiles,
                                "top1": top1,
                                "top5": candidates[:5],
                                "top1_valid": bool(m["valid"]),
                                "top1_exact": bool(m["exact"]),
                                "top1_enantiomer_approx": bool(m["enantiomer_approx"]),
                                "top1_tanimoto": float(m["tanimoto"]),
                            }
                        )

                print(f"[{protocol.name}] {end}/{n}", flush=True)

    return {
        "protocol": protocol.name,
        "num_examples": n,
        "beam_width": beam_width,
        "n_best": n_best,
        "decode_strategy": decode_strategy,
        "topk_exact": {str(k): topk_exact[k] / max(n, 1) for k in topk_values},
        "topk_enantiomer_approx": {str(k): topk_enant[k] / max(n, 1) for k in topk_values},
        "top1_validity": valid_top1 / max(n, 1),
        "top1_avg_tanimoto_valid_pairs": tanimoto_sum / max(valid_pairs, 1),
        "top1_valid_pairs": valid_pairs,
        "samples": sample_rows,
        "protocol_flags": {
            "keep_formula": protocol.keep_formula,
            "keep_1h": protocol.keep_1h,
            "keep_13c": protocol.keep_13c,
            "keep_ir": protocol.keep_ir,
        },
    }


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint under public protocol variants.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--nmr_vocab_json", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--protocols", type=str, default="alberts_like,nmiracle_like")
    parser.add_argument("--beam_width", type=int, default=15)
    parser.add_argument("--n_best", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--decode_strategy", type=str, default="beam", choices=["beam", "greedy"])
    parser.add_argument("--greedy_batch_size", type=int, default=32)
    parser.add_argument("--beam_batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    tokenized_dir = Path(args.data_dir or config["data"]["tokenized_dir"])
    spectral_tokenizer = load_nmr_tokenizer(tokenized_dir, vocab_json_override=args.nmr_vocab_json)
    smiles_tokenizer = SmilesTokenizer(vocab_file=str(Path("training/vocab.txt")))
    device = auto_device(args.device)

    model = MultiModalToSMILESModel(
        smiles_vocab_size=len(smiles_tokenizer),
        nmr_vocab_size=max(spectral_tokenizer.values()) + 1,
        max_seq_length=config["model"]["max_seq_length"],
        max_nmr_length=config["model"]["max_nmr_length"],
        max_memory_length=config["model"]["max_memory_length"],
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        verbose=False,
        use_stablemax=config["model"].get("use_stablemax", False),
        ir_encoder_type=config["model"].get("ir_encoder_type", "regular"),
        ir_as_prompt=config["model"].get("ir_as_prompt", False),
        use_rmsnorm=config["model"].get("use_rmsnorm", False),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    dataset = SpectralSmilesDataset(
        data_dir=str(tokenized_dir),
        smiles_tokenizer=smiles_tokenizer,
        spectral_tokenizer=spectral_tokenizer,
        split=args.split,
        max_smiles_len=config["model"]["max_seq_length"],
        max_nmr_len=config["model"]["max_nmr_length"],
        pretokenize=False,
        preload_ir=False,
        use_disk_cache=False,
        write_disk_cache=False,
    )

    protocol_names = [p.strip() for p in args.protocols.split(",") if p.strip()]
    results = []
    for name in protocol_names:
        if name not in PROTOCOLS:
            raise ValueError(f"Unknown protocol '{name}'. Valid: {', '.join(sorted(PROTOCOLS))}")
        result = evaluate_protocol(
            model=model,
            tokenizer=smiles_tokenizer,
            dataset=dataset,
            spectral_tokenizer=spectral_tokenizer,
            device=device,
            protocol=PROTOCOLS[name],
            max_examples=args.max_examples,
            beam_width=args.beam_width,
            n_best=args.n_best,
            max_len=args.max_len,
            length_penalty=args.length_penalty,
            ir_as_prompt=config["model"].get("ir_as_prompt", False),
            decode_strategy=args.decode_strategy,
            greedy_batch_size=args.greedy_batch_size,
            beam_batch_size=args.beam_batch_size,
        )
        results.append(result)

    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "data_dir": str(tokenized_dir.resolve()),
        "nmr_vocab_json": str(Path(args.nmr_vocab_json).resolve()) if args.nmr_vocab_json else str((tokenized_dir.parent / "vocab.json").resolve()),
        "split": args.split,
        "device": str(device),
        "results": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
