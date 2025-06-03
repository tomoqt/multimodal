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
            try:
                # Load IR data using np.memmap, similar to SpectralVLMDataset
                self.ir = np.memmap(ir_path, dtype='float32', mode='r')
                
                # Reshape if the loaded array is 1D
                array_shape = self.ir.shape
                if len(array_shape) == 1:
                    num_samples = len(self.smiles) # num_samples derived from loaded SMILES data
                    if num_samples > 0 and array_shape[0] % num_samples == 0:
                        feature_dim = array_shape[0] // num_samples
                        self.ir = self.ir.reshape(num_samples, feature_dim)
                        print(f"[SpectralVLMRLDataset - {self.split}] Reshaped IR data to: {self.ir.shape}")
                    else:
                        print(f"[SpectralVLMRLDataset - {self.split}] Warning: IR data is 1D but cannot be reshaped based on SMILES count ({num_samples}). Original shape: {array_shape}")
                        # Not raising an error, but this could be problematic later.
                        # Consider if this case needs more specific handling, e.g., setting self.ir to None or raising error.
                else:
                    print(f"[SpectralVLMRLDataset - {self.split}] Loaded IR data with shape: {self.ir.shape}")

            except Exception as e:
                print(f"[SpectralVLMRLDataset - {self.split}] Warning: Failed to load or process IR data using memmap: {e}")
                self.ir = None

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
            # Use .copy() when creating tensor from memmapped array slice
            ir_tensor = torch.tensor(self.ir[idx].copy(), dtype=torch.float32)
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


def load_model(cfg, spectral_cfg, device):
    """
    Loads the SpectralVLM model, initializing from the base VLM checkpoint 
    specified in cfg["vlm_checkpoint"]["path"], similar to how train_spectral_vlm.py 
    initializes its model from the base nanoVLM.
    """
    
    base_vlm_checkpoint_dir = Path(cfg["vlm_checkpoint"]["path"]) # e.g., "checkpoints/nanoVLM-222M"

    # Initialize a default VLMConfig. 
    # The architecture details will be default, but SpectralVisionLanguageModel 
    # will attempt to load weights from vlm_conf.vlm_checkpoint_path.
    vlm_conf = VLMConfig()
    
    # Set the vlm_checkpoint_path attribute on vlm_conf so SpectralVisionLanguageModel 
    # knows where to load the base VLM weights from.
    vlm_conf.vlm_checkpoint_path = str(base_vlm_checkpoint_dir)
    
    # Determine if the backbone should be loaded from the GRPO config, defaulting to True.
    # This controls whether weights from vlm_conf.vlm_checkpoint_path are actually loaded.
    should_load_backbone = cfg["vlm_checkpoint"].get("load_backbone", True)

    if should_load_backbone:
        print(f"[train_spectral_grpo] Initializing model and loading backbone from: {base_vlm_checkpoint_dir}")
    else:
        print(f"[train_spectral_grpo] Initializing model without loading backbone from: {base_vlm_checkpoint_dir}")

    model = SpectralVisionLanguageModel(
        vlm_conf, 
        load_backbone=should_load_backbone,
        spectral_cfg=spectral_cfg
    )
    
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for SpectralVLM")
    parser.add_argument("--config", default="configs/spectral_grpo_config.yaml")
    # The --checkpoint argument is no longer needed as it's specified in the config.
    # parser.add_argument("--checkpoint", required=True, help="Path to pretrained SpectralVLM checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    train_loader, val_loader = create_data_loaders(tokenizer, cfg)

    # Construct spectral_cfg based on the GRPO config.
    # This configuration must be compatible with the checkpoint being loaded.
    spectral_cfg = {
        "embed_dim": cfg.get("embed_dim", 768), # Ensure this matches the trained model's spectral embed_dim
        "encoder_type": cfg["model"].get("spectral_encoder_type", "convnext"),
        "ir_as_prompt": cfg["model"].get("ir_as_prompt", False),
    }

    # Load the model using the path from the configuration file
    model = load_model(cfg, spectral_cfg, device)

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
