import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from .utils import top_k_top_p_filtering
from .vision_transformer import ViT
from .language_model import LanguageModel
from .modality_projector import ModalityProjector
from .config import VLMConfig
from .spectral_encoder import MultimodalSpectralEncoder

DEBUG = True # Or False, to toggle debugging

class SpectralVisionLanguageModel(nn.Module):
    """
    A tri-modal Vision-Language Model that combines:
    1. Vision: Images processed through ViT
    2. IR Spectra: Processed through spectral encoder
    3. NMR Text: Processed as tokenized text input
    
    Target: SMILES generation wrapped in XML tags
    """
    def __init__(self, cfg: VLMConfig, load_backbone=True, spectral_cfg=None):
        super().__init__()
        self.cfg = cfg
        
        # Vision encoder (keep original ViT)
        if load_backbone:
            print("Loading vision backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
        
        # Language model decoder
        if load_backbone:
            print("Loading language backbone weights")
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.decoder = LanguageModel(cfg)
        
        # Modality projectors
        self.vision_projector = ModalityProjector(cfg)
        
        # IR Spectral encoder and projector
        self.spectral_cfg = spectral_cfg or {
            'embed_dim': cfg.vit_hidden_dim,
            'encoder_type': 'convnext',
            'ir_as_prompt': False
        }
        
        self.ir_encoder = MultimodalSpectralEncoder(
            embed_dim=self.spectral_cfg['embed_dim'],
            encoder_type=self.spectral_cfg['encoder_type'],
            ir_as_prompt=self.spectral_cfg['ir_as_prompt'],
            verbose=False
        )
        
        # Project IR embeddings to language model dimension
        self.ir_projector = nn.Linear(self.spectral_cfg['embed_dim'], cfg.lm_hidden_dim)
        
        self.load_backbone = load_backbone

    def forward(self, input_ids, image=None, ir_data=None, attention_mask=None, targets=None):
        """
        Forward pass for tri-modal input:
        - input_ids: Base prompt tokens (now includes NMR text)
        - image: Vision input (optional)
        - ir_data: IR spectra data (optional) 
        - attention_mask: Attention mask for input
        - targets: Target SMILES tokens for training
        """
        
        embeddings_list = []
        attention_masks = []
        
        # Process image if provided
        if image is not None:
            image_embd = self.vision_encoder(image)
            image_embd = self.vision_projector(image_embd)
            embeddings_list.append(image_embd)
            
            # Create attention mask for image tokens
            batch_size = image_embd.size(0)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), 
                                            device=image_embd.device, 
                                            dtype=torch.bool)
            attention_masks.append(image_attention_mask)
        
        # Process IR spectra if provided
        if ir_data is not None:
            # IR encoder returns embeddings of shape (batch, seq_len, embed_dim)
            ir_embd = self.ir_encoder(nmr_data=None, ir_data=ir_data, c_nmr_data=None)
            if ir_embd is not None:
                ir_embd = self.ir_projector(ir_embd)
                embeddings_list.append(ir_embd)
                
                # Create attention mask for IR tokens
                batch_size = ir_embd.size(0)
                ir_seq_len = ir_embd.size(1)
                ir_attention_mask = torch.ones((batch_size, ir_seq_len), 
                                             device=ir_embd.device, 
                                             dtype=torch.bool)
                attention_masks.append(ir_attention_mask)
        
        # Process main input tokens
        token_embd = self.decoder.token_embedding(input_ids)
        embeddings_list.append(token_embd)
        
        if attention_mask is not None:
            attention_masks.append(attention_mask)
        else:
            batch_size = token_embd.size(0)
            token_seq_len = token_embd.size(1)
            token_attention_mask = torch.ones((batch_size, token_seq_len), 
                                            device=token_embd.device, 
                                            dtype=torch.bool)
            attention_masks.append(token_attention_mask)
        
        # Concatenate all embeddings
        combined_embd = torch.cat(embeddings_list, dim=1)
        
        # Concatenate all attention masks
        if attention_masks:
            combined_attention_mask = torch.cat(attention_masks, dim=1)
        else:
            combined_attention_mask = None
        
        # Pass through language model
        logits = self.decoder(combined_embd, combined_attention_mask)
        
        loss = None
        if targets is not None:
            # Calculate offset for target alignment
            prefix_length = combined_embd.size(1) - token_embd.size(1)
            
            # Apply head to get logits if needed
            if not self.decoder.lm_use_tokens:
                logits = self.decoder.head(logits)
            
            # Extract logits corresponding to target positions
            target_logits = logits[:, prefix_length:, :]
            
            # --- BEGIN DEBUG: Inspect inputs to cross_entropy ---
            if DEBUG and targets is not None:
                print(f"DEBUG: target_logits shape: {target_logits.shape}")
                print(f"DEBUG: targets shape: {targets.shape}")
                print(f"DEBUG: target_logits min: {target_logits.min().item()}, max: {target_logits.max().item()}, mean: {target_logits.mean().item()}")
                if torch.isnan(target_logits).any() or torch.isinf(target_logits).any():
                    print("DEBUG: NaN/Inf found in target_logits!")
                
                print(f"DEBUG: targets min: {targets.min().item()}, max: {targets.max().item()}")
                print(f"DEBUG: unique targets: {torch.unique(targets)}")
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                     print("DEBUG: NaN/Inf found in targets before cross_entropy!")
                
                # Check if all targets are ignore_index
                num_valid_targets = (targets != -100).sum().item()
                print(f"DEBUG: Number of valid targets (not ignore_index): {num_valid_targets}")
                if num_valid_targets == 0:
                    print("DEBUG: Warning! All targets are ignore_index (-100).")

                reshaped_target_logits = target_logits.reshape(-1, target_logits.size(-1))
                reshaped_targets = targets.reshape(-1)
                print(f"DEBUG: reshaped_target_logits shape: {reshaped_target_logits.shape}")
                print(f"DEBUG: reshaped_targets shape: {reshaped_targets.shape}")
            # --- END DEBUG ---

            # Calculate loss
            loss = F.cross_entropy(
                target_logits.reshape(-1, target_logits.size(-1)), 
                targets.reshape(-1), 
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, image=None, ir_data=None, 
                 attention_mask=None, max_new_tokens=100, top_k=50, top_p=0.9, 
                 temperature=0.7, greedy=False):
        """
        Generate SMILES from tri-modal input
        """
        self.eval()
        
        embeddings_list = []
        attention_masks = []
        
        # Process image if provided
        if image is not None:
            image_embd = self.vision_encoder(image)
            image_embd = self.vision_projector(image_embd)
            embeddings_list.append(image_embd)
            
            batch_size = image_embd.size(0)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), 
                                            device=image_embd.device, 
                                            dtype=torch.bool)
            attention_masks.append(image_attention_mask)
        
        # Process IR spectra if provided
        if ir_data is not None:
            ir_embd = self.ir_encoder(nmr_data=None, ir_data=ir_data, c_nmr_data=None)
            if ir_embd is not None:
                ir_embd = self.ir_projector(ir_embd)
                embeddings_list.append(ir_embd)
                
                batch_size = ir_embd.size(0)
                ir_seq_len = ir_embd.size(1)
                ir_attention_mask = torch.ones((batch_size, ir_seq_len), 
                                             device=ir_embd.device, 
                                             dtype=torch.bool)
                attention_masks.append(ir_attention_mask)
        
        # Process main input tokens
        token_embd = self.decoder.token_embedding(input_ids)
        embeddings_list.append(token_embd)
        
        if attention_mask is not None:
            attention_masks.append(attention_mask)
        else:
            batch_size = token_embd.size(0)
            token_seq_len = token_embd.size(1)
            token_attention_mask = torch.ones((batch_size, token_seq_len), 
                                            device=token_embd.device, 
                                            dtype=torch.bool)
            attention_masks.append(token_attention_mask)
        
        # Concatenate initial embeddings
        combined_embd = torch.cat(embeddings_list, dim=1)
        combined_attention_mask = torch.cat(attention_masks, dim=1) if attention_masks else None
        
        batch_size = combined_embd.size(0)
        generated_tokens = torch.zeros((batch_size, max_new_tokens), 
                                     device=input_ids.device, dtype=input_ids.dtype)
        
        # Current embeddings for generation
        current_embd = combined_embd
        current_mask = combined_attention_mask
        
        for i in range(max_new_tokens):
            # Forward pass
            model_out = self.decoder(current_embd, current_mask)
            
            # Get last token logits
            last_token_logits = model_out[:, -1, :]
            
            # Apply head if needed
            if not self.decoder.lm_use_tokens:
                last_token_logits = self.decoder.head(last_token_logits)
            
            # Sample next token
            if greedy:
                next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens[:, i] = next_token.squeeze(-1)
            
            # Update embeddings and mask for next iteration
            next_embd = self.decoder.token_embedding(next_token)
            current_embd = torch.cat((current_embd, next_embd), dim=1)
            
            if current_mask is not None:
                next_mask = torch.ones((batch_size, 1), device=current_mask.device, dtype=current_mask.dtype)
                current_mask = torch.cat((current_mask, next_mask), dim=1)
        
        return generated_tokens

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str, *, revision: Optional[str] = None, 
                       spectral_cfg=None) -> "SpectralVisionLanguageModel":
        """Load model from local directory or Hugging Face Hub"""
        
        # Handle paths similar to original VLM
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(f"Config file not found at {config_path}")
            if not os.path.exists(weights_path):
                raise ValueError(f"Weights file not found at {weights_path}")
        else:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=repo_id_or_path, filename="config.json", revision=revision)
            weights_path = hf_hub_download(repo_id=repo_id_or_path, filename="model.safetensors", revision=revision)

        # Load config
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
            cfg = VLMConfig(**cfg_dict.get('vlm_config', cfg_dict))

        # Initialize model
        model = cls(cfg, load_backbone=False, spectral_cfg=spectral_cfg)

        # Load weights
        load_model(model, weights_path)

        return model

    def save_pretrained(self, save_directory: str) -> None:
        """Save model and config to directory"""
        os.makedirs(save_directory, exist_ok=True)

        # Save config including spectral config
        config_dict = {
            'vlm_config': asdict(self.cfg),
            'spectral_config': self.spectral_cfg
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

        # Save weights
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """Push model to Hugging Face Hub"""
        from huggingface_hub import create_repo, upload_folder

        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            self.save_pretrained(save_path)

            # Create model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(SPECTRAL_MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload SpectralVLM using push_to_hub",
            )


SPECTRAL_MODEL_CARD_TEMPLATE = """
---
library_name: spectral_vlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - spectroscopy
  - multimodal
  - chemistry
  - smiles
---

**SpectralVLM** is a tri-modal Vision-Language Model designed for molecular structure prediction from spectroscopic data. 
It combines:

1. **Vision**: Images processed through ViT encoder
2. **IR Spectra**: Processed through convolutional spectral encoder  
3. **NMR Text**: Processed as tokenized text sequences

The model generates SMILES (Simplified Molecular Input Line Entry System) representations wrapped in XML tags.

**Usage:**

```python
from models.spectral_vision_language_model import SpectralVisionLanguageModel

model = SpectralVisionLanguageModel.from_pretrained("{repo_id}")
```

For more information, check out the base model and training details.
""" 