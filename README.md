# MultiModal

### Introduction
This work aims at developing a set of models capable of representing the underlying physics of spectral information and utilizing to carry out a set of tasks. The dataset utilized, as well as some of the choices, are taken from https://arxiv.org/pdf/2407.17492 .
[UPDATE] take a look at the 📄 [preprint](./preprint.pdf)
! 

### Spectral Vision-Language Model (SpectralVLM)

**NEW:** We have developed a tri-modal Vision-Language Model that combines:
1. **Vision**: Molecular images processed through Vision Transformer (ViT)
2. **IR Spectra**: Raw spectral data processed through convolutional encoders
3. **NMR Text**: Tokenized NMR sequences formatted as XML

The model generates SMILES (Simplified Molecular Input Line Entry System) representations wrapped in XML tags, providing a structured approach to molecular structure prediction from multi-modal spectroscopic data.

#### Key Features:
- **Tri-modal Architecture**: Seamlessly integrates vision, IR spectral, and NMR text modalities
- **XML-Formatted Prompts**: Structured input/output format for better parsing and evaluation
- **Pretrained Backbone**: Built on top of nanoVLM with SigLIP vision encoder and SmolLM language model
- **Flexible Training**: Supports distributed training with mixed precision
- **Spectral Encoder Options**: Choice between ConvNeXt1D and regular CNN encoders for IR data

#### Architecture Overview:
```
Input Modalities:
├── Images (optional) → ViT Encoder → Vision Projector
├── IR Spectra → Spectral Encoder → IR Projector  
└── NMR Text → Tokenizer → Token Embeddings

All embeddings are concatenated and fed to the language model decoder
Target: <smiles>MOLECULAR_STRUCTURE</smiles>
```

#### Training the Spectral VLM:

```bash
# Basic training
cd training
python train_spectral_vlm.py --data_dir tokenized_baseline/data

# With custom parameters
python train_spectral_vlm.py \
    --data_dir tokenized_baseline/data \
    --lr_spectral 2e-3 \
    --lr_backbones 1e-4 \
    --batch_size 16 \
    --epochs 5 \
    --include_images

# Distributed training
torchrun --nproc_per_node=2 train_spectral_vlm.py \
    --data_dir tokenized_baseline/data \
    --lr_spectral 2e-3 \
    --lr_backbones 1e-4
```

#### Using the Spectral VLM:

```python
from nanoVLM.models.spectral_vision_language_model import SpectralVisionLanguageModel
from training.data.processors import get_tokenizer, get_image_processor

# Load model
model = SpectralVisionLanguageModel.from_pretrained("path/to/checkpoint")
tokenizer = get_tokenizer()

# Prepare input
prompt = """Analyze the following spectroscopic data and predict the molecular structure.

<nmr_data>7.25 7.30 d 2H 7.15 7.20 d 2H 4.35 4.40 q 1H 3.85 s 3H</nmr_data>

Please provide the SMILES representation of the molecule:"""

input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

# Generate SMILES
generated = model.generate(
    input_ids=input_ids,
    ir_data=ir_spectra,  # Optional: IR spectral data
    nmr_text_ids=nmr_tokens,  # Optional: NMR token IDs
    image=molecular_image,  # Optional: molecular image
    max_new_tokens=100,
    temperature=0.7
)

response = tokenizer.decode(generated[0], skip_special_tokens=True)
print(response)  # Should contain: <smiles>PREDICTED_STRUCTURE</smiles>
```

#### Data Format:

The spectral VLM expects data in XML format:

**Input Prompt:**
```xml
Analyze the following spectroscopic data and predict the molecular structure.

<nmr_data>7.25 7.30 d 2H 7.15 7.20 d 2H 4.35 4.40 q 1H 3.85 s 3H</nmr_data>

Please provide the SMILES representation of the molecule:
```

**Target Output:**
```xml
<smiles>COc1ccc(C(C)O)cc1</smiles>
```

### Spectra to SMILES (Original Approach)
Our current effort is mainly allocated towards building a model that decodes molecular structure, in the shape of SMILES, from spectral information. The current implementation contains a set of convolutional encoders and a transformer-based decoder to autoregressively predict SMILES. 
Currently, our work is taking the following (rapidly changing) directions: 
- architecture validation among enc-dec, dec-only, and various modality fusion alternatives.
- autoregressive vs non-autoregressive approach. Specifically, the non-autoregressive approach would mean decoding to a graph (or some othe perm-invariant sturcture) losing though some flexibility in dynamically determining molecule size.
- detecting failure patterns in the model

### Other tasks
There's a bunch of other interesting things that could be done with a dataset like this. For example, we could train a cross-spectral reconstruction model. another idea is to invert the process and try to simulate the spectra from the smiles. 

### Installation

* install and download data
```bash
python3 download_data.py  

# install packages and download
pip install -r requirements.txt
# OR [if want faster, at your own risk] python3 download_data_parallel.py 
```

* Tokenize data (or skip if you're not in development, not needed to actually train):
```bash
pip install rxn-chem-utils
# test: 
python3 create_tokenized_dataset_smallram.py --analytical_data "data_extraction/multimodal_spectroscopic_dataset" --out_path "tokenized_baseline" --h_nmr --c_nmr --ir --formula
# real
python3 create_tokenized_dataset_faster.py --analytical_data "data_extraction/multimodal_spectroscopic_dataset" --out_path "tokenized_baseline" --h_nmr --c_nmr --ir --formula
```

* Download the (pre-tokenized) data: 
```bash
python3 data/download_tokenized_dataset.py  
python3 data/build_vocab.py
```

### Training Scripts

#### Original Autoregressive Training:
```bash
# train the original model
torchrun --nproc_per_node=1 training/train_autoregressive.py --config configs/test_config.yaml 

#test inference modes
python test_inference --config your_config_path --checkpoint your_checkpoint_path
```

#### Spectral VLM Training:
```bash
# Train spectral VLM (new approach)
cd training
python train_spectral_vlm.py --data_dir tokenized_baseline/data

# With distributed training
torchrun --nproc_per_node=2 train_spectral_vlm.py --data_dir tokenized_baseline/data
```

#### GRPO Fine-Tuning:
```bash
# Fine-tune a pretrained Spectral VLM using GRPO
cd training
python train_spectral_grpo.py --config configs/spectral_grpo_config.yaml --checkpoint path/to/spectral_checkpoint.pt
```
The `spectral_grpo_config.yaml` file enables `cot_reward` to encourage the `<thinking>...</thinking><answer>...</answer>` output format.


### Model Checkpoints

#### Pretrained Models:
- **nanoVLM-222M**: Base vision-language model from [nanoVLM](https://huggingface.co/lusxvr/nanoVLM-222M)
- **SpectralVLM**: Tri-modal model for spectroscopic analysis (coming soon)

### Evaluation Metrics

The SpectralVLM is evaluated on:
- **Valid SMILES Rate**: Percentage of generated SMILES that are chemically valid
- **Exact Match Rate**: Percentage of predictions that exactly match the target SMILES
- **Molecular Similarity**: Tanimoto similarity between generated and target molecules
- **MCS Ratio**: Maximum Common Substructure ratio
- **ECFP6 IoU**: Extended Connectivity Fingerprint Intersection over Union

### Current TODOs:
- [x] Implement tri-modal SpectralVLM architecture
- [x] Create XML-formatted data processing pipeline
- [x] Develop distributed training script
- [ ] Benchmark SpectralVLM vs original encoder-decoder approach
- [ ] Add molecular image generation capabilities
- [ ] Implement cross-spectral reconstruction
- [ ] Add few-shot learning support
- [ ] Integrate with RDKit for SMILES validation
- [ ] Create web demo for spectral analysis

### Research Directions:
1. **Multi-modal Fusion**: Investigating optimal combination strategies for vision, IR, and NMR modalities
2. **Prompt Engineering**: Developing effective XML formatting and instruction prompts
3. **Transfer Learning**: Leveraging pretrained vision-language models for spectroscopic tasks
4. **Evaluation Metrics**: Comprehensive molecular similarity and validity assessment
5. **Real-world Deployment**: Integration with laboratory workflows and spectroscopy equipment









