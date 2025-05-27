from transformers import AutoTokenizer, AutoImageProcessor
from torchvision import transforms
from PIL import Image
import torch


def get_tokenizer(model_name="HuggingFaceTB/cosmo2-tokenizer"):
    """Get tokenizer for language model"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure we have necessary special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def get_image_processor(image_size=224):
    """Get image processor for vision encoder"""
    
    # Use basic torchvision transforms instead of AutoImageProcessor for flexibility
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform


def preprocess_spectral_data(ir_data, nmr_data, normalize_ir=True):
    """Preprocess spectral data for model input"""
    
    processed_data = {}
    
    # Process IR data
    if ir_data is not None:
        ir_tensor = torch.tensor(ir_data, dtype=torch.float32)
        
        if normalize_ir:
            # Normalize IR spectra to [0, 1] range
            ir_min = ir_tensor.min()
            ir_max = ir_tensor.max()
            if ir_max > ir_min:
                ir_tensor = (ir_tensor - ir_min) / (ir_max - ir_min)
        
        processed_data['ir_data'] = ir_tensor
    
    # Process NMR data (already tokenized in dataset)
    if nmr_data is not None:
        processed_data['nmr_data'] = nmr_data
    
    return processed_data


def create_xml_prompt(nmr_text=None, ir_description=None, image_description=None, smiles=None):
    """Create XML-formatted prompt for spectral analysis"""
    
    prompt_parts = [
        "You are an expert in molecular spectroscopy and structure prediction.",
        "Analyze the provided spectroscopic data and predict the molecular structure.",
        ""
    ]
    
    # Add available modalities
    if nmr_text:
        prompt_parts.extend([
            "<nmr_data>",
            nmr_text,
            "</nmr_data>",
            ""
        ])
    
    if ir_description:
        prompt_parts.extend([
            "<ir_data>",
            ir_description,
            "</ir_data>",
            ""
        ])
    
    if image_description:
        prompt_parts.extend([
            "<molecular_image>",
            image_description,
            "</molecular_image>",
            ""
        ])
    
    prompt_parts.append("Please provide the SMILES representation of the molecule:")
    
    if smiles:
        prompt_parts.extend([
            "",
            "<smiles>",
            smiles,
            "</smiles>"
        ])
    
    return "\n".join(prompt_parts)


def extract_smiles_from_response(response_text):
    """Extract SMILES from model response with XML tags"""
    
    # Try to extract from XML tags first
    if '<smiles>' in response_text and '</smiles>' in response_text:
        start_idx = response_text.find('<smiles>') + len('<smiles>')
        end_idx = response_text.find('</smiles>')
        smiles = response_text[start_idx:end_idx].strip()
        return smiles
    
    # Fallback: try to find SMILES-like patterns
    import re
    
    # Basic SMILES pattern (simplified)
    smiles_pattern = r'[A-Za-z0-9@+\-\[\]()=#$%:/.\\]+'
    matches = re.findall(smiles_pattern, response_text)
    
    # Return the longest match that looks like a SMILES
    if matches:
        longest_match = max(matches, key=len)
        if len(longest_match) > 5:  # Minimum reasonable SMILES length
            return longest_match
    
    return response_text.strip()


def format_nmr_data(nmr_tokens):
    """Format NMR token data into readable text"""
    
    if isinstance(nmr_tokens, list):
        # Join tokens with spaces
        return " ".join(str(token) for token in nmr_tokens)
    elif isinstance(nmr_tokens, str):
        return nmr_tokens
    else:
        return str(nmr_tokens)


def create_few_shot_examples():
    """Create few-shot examples for in-context learning"""
    
    examples = [
        {
            "nmr": "7.25 7.30 d 2H 7.15 7.20 d 2H 4.35 4.40 q 1H 3.85 s 3H 1.35 1.40 d 3H",
            "smiles": "COc1ccc(C(C)O)cc1",
            "description": "Simple aromatic compound with methoxy and secondary alcohol groups"
        },
        {
            "nmr": "8.20 8.25 d 1H 7.60 7.65 m 2H 7.40 7.45 m 1H 4.45 q 2H 1.45 t 3H",
            "smiles": "CCOC(=O)c1ccccc1",
            "description": "Ethyl benzoate - aromatic ester"
        },
        {
            "nmr": "3.70 s 6H 2.30 s 4H",
            "smiles": "COC(=O)CC(=O)OC",
            "description": "Dimethyl succinate - simple diester"
        }
    ]
    
    return examples


def batch_process_spectra(batch_data, tokenizer, image_processor=None):
    """Process a batch of spectral data for training"""
    
    processed_batch = {}
    
    # Process text data
    if 'text' in batch_data:
        text_data = batch_data['text']
        if isinstance(text_data, list):
            # Batch tokenization
            encoding = tokenizer(
                text_data,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            processed_batch.update(encoding)
    
    # Process images if available
    if 'images' in batch_data and image_processor is not None:
        images = batch_data['images']
        if isinstance(images, list):
            # Process each image
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    # Load from path
                    img = Image.open(img).convert('RGB')
                processed_img = image_processor(img)
                processed_images.append(processed_img)
            processed_batch['images'] = torch.stack(processed_images)
    
    # Process spectral data
    for key in ['ir_data', 'nmr_data']:
        if key in batch_data:
            data = batch_data[key]
            if isinstance(data, list):
                # Convert to tensor
                processed_batch[key] = torch.tensor(data, dtype=torch.float32)
            else:
                processed_batch[key] = data
    
    return processed_batch 