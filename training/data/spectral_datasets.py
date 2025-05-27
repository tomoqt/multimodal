import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random

DEBUG_DATASET = True # Set to False to disable these specific prints

class SpectralVLMDataset(Dataset):
    """
    Dataset for tri-modal spectral VLM training:
    - Vision: Images (optional)
    - IR Spectra: Raw spectral data
    - NMR: Text sequences formatted as XML
    - Target: SMILES wrapped in XML tags
    """
    
    def __init__(
        self, 
        data_dir, 
        tokenizer, 
        image_processor=None,
        split='train',
        max_length=512,
        include_images=True,
        # nmr_vocab_path=None # Removed
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.include_images = include_images
        self.split = split
        
        # Load NMR vocabulary if provided
        # self.nmr_vocab = None # Removed
        # if nmr_vocab_path and Path(nmr_vocab_path).exists(): # Removed
        #     with open(nmr_vocab_path) as f: # Removed
        #         self.nmr_vocab = json.load(f) # Removed
        
        # Load data files
        self._load_data()
        
        print(f"[SpectralVLMDataset] Loaded {len(self.smiles_data)} samples for {split}")
    
    def _load_data(self):
        """Load all data components"""
        
        # Load SMILES targets
        smiles_path = self.data_dir / f"tgt-{self.split}.txt"
        with open(smiles_path) as f:
            self.smiles_data = [line.strip().replace(" ", "") for line in f]
        
        # Load NMR text data
        nmr_path = self.data_dir / f"src-{self.split}.txt"
        with open(nmr_path) as f:
            self.nmr_data = [line.strip() for line in f]
        
        # Load IR spectra data
        ir_path = self.data_dir / f"ir-{self.split}.npy"
        self.ir_data = None
        if ir_path.exists():
            try:
                self.ir_data = np.memmap(ir_path, dtype='float32', mode='r')
                # Reshape if needed
                array_shape = self.ir_data.shape
                if len(array_shape) == 1:
                    num_samples = len(self.smiles_data)
                    feature_dim = array_shape[0] // num_samples
                    self.ir_data = self.ir_data.reshape(num_samples, feature_dim)
                print(f"[Dataset] Loaded IR data with shape: {self.ir_data.shape}")
            except Exception as e:
                print(f"[Warning] Failed to load IR data: {e}")
                self.ir_data = None
        
        # Load images if available and requested
        self.image_paths = None
        if self.include_images:
            image_dir = self.data_dir.parent / "images" / self.split
            if image_dir.exists():
                # Assume images are named by index or have a mapping file
                self.image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
                if len(self.image_paths) != len(self.smiles_data):
                    print(f"[Warning] Image count ({len(self.image_paths)}) doesn't match data count ({len(self.smiles_data)})")
                    self.image_paths = None
    
    def _create_prompt(self, nmr_text, smiles_target, include_target=True):
        """Create XML-formatted prompt with NMR data and optional SMILES target"""
        
        # Format NMR data as XML
        nmr_formatted = f"<nmr_data>{nmr_text}</nmr_data>"
        
        # Base prompt
        prompt = f"""Analyze the following spectroscopic data and predict the molecular structure.

{nmr_formatted}

Please provide the SMILES representation of the molecule:"""
        
        if include_target:
            prompt += f"\n<smiles>{smiles_target}</smiles>"
        
        return prompt
    
    # def _tokenize_nmr(self, nmr_text): # Removed
    #     """Convert NMR text to token IDs if vocabulary is available""" # Removed
    #     if self.nmr_vocab is None: # Removed
    #         # Fallback: use main tokenizer # Removed
    #         return self.tokenizer.encode(nmr_text, add_special_tokens=False, max_length=128, truncation=True) # Removed
        
    #     # Use NMR-specific vocabulary # Removed
    #     tokens = nmr_text.split() # Removed
    #     token_ids = [self.nmr_vocab.get(token, self.nmr_vocab.get("<UNK>", 0)) for token in tokens] # Removed
    #     # Limit length # Removed
    #     if len(token_ids) > 128: # Removed
    #         token_ids = token_ids[:128] # Removed
        
    #     return token_ids # Removed
    
    def __len__(self):
        return len(self.smiles_data)
    
    def __getitem__(self, idx):
        """
        Returns:
        {
            'input_ids': tokenized prompt (full prompt including target),
            'attention_mask': attention mask for the full prompt,
            'labels': tokenized prompt, with -100 for prompt tokens before <smiles> tag,
            'ir_data': IR spectra (optional),
            'image': processed image (optional)
        }
        """
        
        smiles = self.smiles_data[idx]
        nmr_text = self.nmr_data[idx]
        
        full_prompt_text = self._create_prompt(nmr_text, smiles, include_target=True)
        
        encoding = self.tokenizer(
            full_prompt_text,
            max_length=self.max_length, 
            truncation=True,
            padding=False, 
            return_tensors=None 
        )
        
        input_ids_list = encoding['input_ids']
        attention_mask_list = encoding['attention_mask']
        
        # Create labels: -100 for prompt, actual tokens for SMILES part
        # Default to all -100 if <smiles> tag is not found or content is missing
        labels_list = [-100] * len(input_ids_list)

        try:
            # Find the start of the SMILES content to set labels appropriately
            # Tokenize parts of the prompt to find the boundary robustly
            prompt_part_text = self._create_prompt(nmr_text, "", include_target=False) # Prompt before SMILES
            # Remove the trailing part of the prompt that asks for SMILES prediction if it exists
            prediction_request_text = "Please provide the SMILES representation of the molecule:"
            if prompt_part_text.endswith(prediction_request_text):
                 prompt_part_text = prompt_part_text[:-len(prediction_request_text)].strip()
            
            # Simplified: find the start of <smiles> tag in the tokenized full prompt
            # This is an approximation. A more robust way would be to tokenize the prompt part and target part separately
            # and align, but that's more complex with potential tokenizer variations.
            smiles_tag_start_tokens = self.tokenizer.encode("<smiles>", add_special_tokens=False)
            
            found_idx = -1
            if len(smiles_tag_start_tokens) > 0:
                for i in range(len(input_ids_list) - len(smiles_tag_start_tokens) + 1):
                    if input_ids_list[i:i+len(smiles_tag_start_tokens)] == smiles_tag_start_tokens:
                        found_idx = i
                        break
            
            if found_idx != -1:
                # Start labels from the beginning of the <smiles> tag
                # The model should predict the tag and its content
                label_start_index = found_idx
                if label_start_index < len(input_ids_list):
                    labels_list = [-100] * label_start_index + input_ids_list[label_start_index:]
                # else: all labels remain -100 if tag is at the very end or beyond truncated length
            # else: if <smiles> tag not found, all labels remain -100 (might happen if truncated before tag)

        except Exception as e:
            # If any error in finding split point, all labels remain -100 to be safe
            # This helps in debugging data issues if they cause errors here
            print(f"Warning: Error creating labels for item {idx}, defaulting to all -100. Error: {e}")
            # labels_list remains all -100s
            pass # Keep default all -100 labels

        result = {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long),
            'labels': torch.tensor(labels_list, dtype=torch.long)
        }
        
        if self.ir_data is not None:
            result['ir_data'] = torch.tensor(self.ir_data[idx].copy(), dtype=torch.float32)
        
        # Add image if available
        if self.image_paths and self.image_processor:
            try:
                image_path = self.image_paths[idx]
                image = Image.open(image_path).convert('RGB')
                result['image'] = self.image_processor(image)
            except Exception as e:
                print(f"[Warning] Failed to load image {idx}: {e}")
                # Create dummy image
                dummy_image = Image.new('RGB', (224, 224), color='white')
                result['image'] = self.image_processor(dummy_image)
        
        return result


class SpectralCollator:
    """Collator for batching spectral VLM data"""
    
    def __init__(self, tokenizer, max_length=512, pad_to_max=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max
    
    def __call__(self, batch):
        """
        Collate batch of spectral VLM samples
        """
        
        # Extract components
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        # ir_data = [item['ir_data'] for item in batch if 'ir_data' in item] # Keep ir_data if present
        # nmr_token_ids = [item['nmr_token_ids'] for item in batch if 'nmr_token_ids' in item] # Removed
        
        # Handle optional modalities
        ir_data = None
        if 'ir_data' in batch[0]: # Check if ir_data is present in the first item (assuming consistency)
            ir_data = [item['ir_data'] for item in batch]
        
        # Pad sequences
        if self.pad_to_max:
            max_len = self.max_length
        else:
            max_len = max(len(seq) for seq in input_ids)
            max_len = min(max_len, self.max_length)
        
        # Pad input_ids and attention_masks
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        # padded_nmr_token_ids = [] # Removed
        
        for i in range(len(batch)):
            # Pad input_ids
            seq_len = len(input_ids[i])
            if seq_len < max_len:
                pad_length = max_len - seq_len
                padded_input_ids.append(
                    torch.cat([
                        input_ids[i], 
                        torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                    ])
                )
                padded_attention_masks.append(
                    torch.cat([
                        attention_masks[i],
                        torch.zeros(pad_length, dtype=torch.long)
                    ])
                )
                padded_labels.append(
                    torch.cat([
                        labels[i][:seq_len],
                        torch.full((pad_length,), -100, dtype=torch.long)
                    ])
                )
            else: # if seq_len == max_len or seq_len > max_len (already truncated)
                padded_input_ids.append(input_ids[i][:max_len])
                padded_attention_masks.append(attention_masks[i][:max_len])
                padded_labels.append(labels[i][:max_len])

            # Pad nmr_token_ids if present # Removed
            # if nmr_token_ids: # Removed
            #     nmr_seq_len = len(nmr_token_ids[i]) # Removed
            #     if nmr_seq_len < max_nmr_len: # Removed
            #         pad_length = max_nmr_len - nmr_seq_len # Removed
            #         padded_nmr_token_ids.append( # Removed
            #             torch.cat([ # Removed
            #                 nmr_token_ids[i], # Removed
            #                 torch.full((pad_length,), 0, dtype=torch.long) # Assuming 0 for padding NMR tokens # Removed
            #             ]) # Removed
            #         ) # Removed
            #     else: # Removed
            #         padded_nmr_token_ids.append(nmr_token_ids[i][:max_nmr_len]) # Removed
        
        collated_batch = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': torch.stack(padded_labels),
        }
        
        if ir_data:
            collated_batch['ir_data'] = torch.stack(ir_data)
        # if nmr_token_ids: # Removed
        #     collated_batch['nmr_token_ids'] = torch.stack(padded_nmr_token_ids) # Removed
            
        return collated_batch


# Example usage for creating prompt templates
def create_few_shot_prompt(examples, query_nmr, query_smiles=None):
    """Create few-shot learning prompt with examples"""
    
    prompt_parts = [
        "You are an expert in molecular spectroscopy. Given NMR data, predict the SMILES structure.",
        "",
        "Examples:",
        ""
    ]
    
    # Add examples
    for i, (nmr, smiles) in enumerate(examples):
        prompt_parts.extend([
            f"Example {i+1}:",
            f"<nmr_data>{nmr}</nmr_data>",
            f"<smiles>{smiles}</smiles>",
            ""
        ])
    
    # Add query
    prompt_parts.extend([
        "Now predict the structure for this data:",
        f"<nmr_data>{query_nmr}</nmr_data>",
        ""
    ])
    
    if query_smiles:
        prompt_parts.append(f"<smiles>{query_smiles}</smiles>")
    
    return "\n".join(prompt_parts) 