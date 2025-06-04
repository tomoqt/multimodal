# MultiModal

MultiModal explores how to generate molecular structures in SMILES format from spectroscopic information such as NMR, IR and MS. The architecture combines convolutional encoders for the spectra and a Transformer decoder.

The dataset and several design choices are based on the paper at <https://arxiv.org/pdf/2407.17492>. For more details see the [preprint](./preprint.pdf).

## Repository layout

- **`models/`** – encoder/decoder modules and the SMILES tokenizer
- **`training/`** – training scripts and vocabulary file
- **`inference/`** – inference utilities and decoding strategies
- **`data/`** – scripts for downloading and tokenizing the dataset
- **`configs/`** – YAML configuration files for experiments
- **`utils/`** – logging helpers and custom optimizers
- Gradio demos (`gradio_app*.py`) for interactive testing

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download raw data**
   ```bash
   python3 download_data.py
   # optionally, for parallel download
   python3 download_data_parallel.py
   ```
3. **Tokenize the data** (optional if you download the prepared version)
   ```bash
   pip install rxn-chem-utils
   python3 create_tokenized_dataset_faster.py \
       --analytical_data data_extraction/multimodal_spectroscopic_dataset \
       --out_path tokenized_baseline \
       --h_nmr --c_nmr --ir --formula
   ```
Alternatively **Fetch pre-tokenized data and build the vocabulary**
   ```bash
   python3 data/download_tokenized_dataset.py
   python3 data/build_vocab.py
   ```

## Training

Run the autoregressive training script with a configuration file:

```bash
torchrun --nproc_per_node=1 train_autoregressive.py --config configs/test_config.yaml
```

Configuration files in `configs/` control model size, dataset paths and optimization parameters.

### RL Training with Verifiers

To fine-tune a model using Grouped Relative Policy Optimization provided by the
[verifiers](https://github.com/willccbb/verifiers) library on raw NMR spectra
you can run:

```bash
python training/train_verifiers_nmr.py --data-path /path/to/data \
    --model-name Qwen/Qwen2.5-1.5B-Instruct
```

`data-path` should contain `train.jsonl` and `val.jsonl` files with `nmr` and
`smiles` keys. The script computes rewards using Tanimoto similarity between the
predicted and target molecules.

## Inference

After training you can test different decoding strategies using:

```bash
python test_inference.py --config your_config_path --checkpoint your_checkpoint_path
```

`inference/INFERENCE_README.md` describes the available strategies: greedy decoding, beam search, sampling, nucleus sampling and Entropix.

## Reference

The approach and background are discussed in detail in the [preprint](./preprint.pdf).
