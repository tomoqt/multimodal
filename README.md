# MultiModal

### Introduction
This work aims at developing a set of models capable of representing the underlying physics of spectral information and utilizing to carry out a set of tasks. The dataset utilized, as well as some of the choices, are taken from https://arxiv.org/pdf/2407.17492 .

### Spectra to SMILES
Our current effort is mainly allocated towards building a model that decodes molecular structure, in the shape of SMILES, from spectral information. The current implementation contains a set of convolutional encoders and a transformer-based decoder to autoregressively predict SMILES. 
Currently, our work is taking the following (rapidly changing) directions: 
- architecture validation among enc-dec, dec-only, and various modality fusion alternatives.
- autoregressive vs non-autoregressive approach. Specifically, the non-autoregressive approach would mean decoding to a graph (or some othe perm-invariant sturcture) losing though some flexibility in dynamically determining molecule size.
- detecting failure patterns in the model
### Other tasks
There's a bunch of other interesting things that could be done with a dataset like this. For example, we could train a cross-spectral reconstruction model. another idea is to invert the process and try to simulate the spectra from the smiles. 

Current TODOs:
-replicating original paper processing for encoders (both in current enc-dec structure, possibly directly tokenizing and decoding)
-branching to try non-autoregressive methods
-audit current training dynamics
-fix ezmup for hp sweeping (current problem is pwconv in encoder)

### Installation

* install and download data
```bash
# install packages and download
pip install -r requirements.txt
python3 download_data.py  
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
* Download the data: 
```bash
python3 download_tokenized_dataset.py
python3 build_vocab.py

# train the model
CUDA_VISIBLE_DEVICES=0 python3 train_autoregressive.py --config configs/local_config.yaml
CUDA_VISIBLE_DEVICES=2 python3 train_autoregressive.py --config configs/real_config.yaml

# MUP
CUDA_VISIBLE_DEVICES=1 python3 train_autoregressive_mup.py --config configs/real_config.yaml


```


### Paper Notes: 


#### Ablations

To investigate the effect of molecular (in) validity of model outputs, we train a model to produce the SELFIES encoding of molecules instead of SMILES. The results are provided in Table 4. 







