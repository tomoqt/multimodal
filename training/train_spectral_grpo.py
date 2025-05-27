import argparse
import json
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from nanoVLM.models.spectral_vision_language_model import SpectralVisionLanguageModel
from nanoVLM.models.config import VLMConfig
from training.data.processors import get_tokenizer
from training.train_grpo import GRPO

class SpectralVLMRLDataset(Dataset):
    """Simple dataset returning SMILES tokens, IR spectra and NMR tokens."""
    def __init__(self, data_dir, tokenizer, split="train", max_length=512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self._load_data()

    def _load_data(self):
        with open(self.data_dir / f"tgt-{self.split}.txt") as f:
            self.smiles = [line.strip().replace(" ", "") for line in f]
        with open(self.data_dir / f"src-{self.split}.txt") as f:
            self.nmr = [line.strip() for line in f]
        ir_path = self.data_dir / f"ir-{self.split}.npy"
        self.ir = None
        if ir_path.exists():
            self.ir = np.load(ir_path, mmap_mode="r")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        nmr = self.nmr[idx]
        tgt_ids = self.tokenizer.encode(
            smi, add_special_tokens=True, max_length=self.max_length, truncation=True
        )
        nmr_ids = self.tokenizer.encode(
            nmr, add_special_tokens=True, max_length=self.max_length, truncation=True
        )
        ir_tensor = None
        if self.ir is not None:
            ir_tensor = torch.tensor(self.ir[idx], dtype=torch.float32)
        return (
            torch.tensor(tgt_ids, dtype=torch.long),
            ir_tensor,
            torch.tensor(nmr_ids, dtype=torch.long),
            None,
        )


def collate_rl_fn(batch, tokenizer):
    targets, ir_list, nmr_list, _ = zip(*batch)
    max_t = max(len(t) for t in targets)
    pad_id = tokenizer.pad_token_id
    tgt_batch = torch.stack([
        torch.cat([t, torch.full((max_t - len(t),), pad_id, dtype=torch.long)]) if len(t) < max_t else t
        for t in targets
    ])
    max_n = max(len(n) for n in nmr_list)
    nmr_batch = torch.stack([
        torch.cat([n, torch.full((max_n - len(n),), pad_id, dtype=torch.long)]) if len(n) < max_n else n
        for n in nmr_list
    ])
    ir_batch = None
    if ir_list[0] is not None:
        ir_batch = torch.stack(ir_list)
    return tgt_batch, ir_batch, nmr_batch, None


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def create_data_loaders(tokenizer, cfg):
    data_dir = cfg["data"]["data_dir"]
    max_len = cfg["model"].get("max_length", 512)
    batch_size = cfg["training"].get("batch_size", 1)
    train_ds = SpectralVLMRLDataset(data_dir, tokenizer, split="train", max_length=max_len)
    val_ds = SpectralVLMRLDataset(data_dir, tokenizer, split="val", max_length=max_len)
    collate = lambda b: collate_rl_fn(b, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def load_model(checkpoint_path, cfg, spectral_cfg, device):
    with open(Path(cfg["vlm_checkpoint"]["path"]) / "config.json") as f:
        vlm_conf = VLMConfig(**json.load(f).get("vlm_config", {}))
    model = SpectralVisionLanguageModel(vlm_conf, load_backbone=True, spectral_cfg=spectral_cfg)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for SpectralVLM")
    parser.add_argument("--config", default="configs/spectral_grpo_config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained SpectralVLM checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    train_loader, val_loader = create_data_loaders(tokenizer, cfg)

    spectral_cfg = {
        "embed_dim": cfg.get("embed_dim", 768),
        "encoder_type": cfg["model"].get("spectral_encoder_type", "convnext"),
        "ir_as_prompt": cfg["model"].get("ir_as_prompt", False),
    }

    model = load_model(args.checkpoint, cfg, spectral_cfg, device)

    grpo_cfg = cfg["rl"]["grpo"]
    grpo = GRPO(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        group_size=grpo_cfg.get("group_size", 8),
        micro_group_size=grpo_cfg.get("micro_group_size", 1),
        batch_size=grpo_cfg.get("batch_size", 1),
        max_iterations=grpo_cfg.get("max_iterations", 1000),
        train_loader=train_loader,
        val_loader=val_loader,
        log_wandb=grpo_cfg.get("log_wandb", False),
        lr=grpo_cfg.get("learning_rate", 1e-5),
        beta=grpo_cfg.get("beta", 0.1),
        epsilon=grpo_cfg.get("epsilon", 1.0),
        temperature=grpo_cfg.get("temperature", 1.0),
        use_cot_reward=grpo_cfg.get("cot_reward", False),
        device=device,
        use_kl=True,
        log_frequency=cfg["training"].get("logging_frequency", 1),
        validation_frequency=cfg["training"].get("validation_frequency", 10),
    )

    grpo.train(num_iterations=grpo_cfg.get("max_iterations", 1000))


if __name__ == "__main__":
    main()
