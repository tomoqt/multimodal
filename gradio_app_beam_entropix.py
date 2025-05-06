import gradio as gr
import torch
import os
import json
import yaml
import random
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import io
import base64

# --- Matplotlib for plotting ---
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: Matplotlib not found. Spectral plotting will be disabled.")

# --- RDKit Imports ---
try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not found. Molecule drawing will be disabled.")

# --- Model/Helper Imports ---
try:
    from models.multimodal_to_smiles import MultiModalToSMILESModel
    from models.smiles_tokenizer import SmilesTokenizer
    from inference.inference import ModelInference, DecodingStrategy
    from test_inference import load_config as util_load_config
    from test_inference import get_ir_tokenizer as util_get_ir_tokenizer
    from test_inference import detect_ir_as_prompt as util_detect_ir_as_prompt
    from test_inference import SimpleSpectralSmilesDataset
    from test_inference import evaluate_predictions, aggregate_metrics
    MODEL_FILES_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    MODEL_FILES_AVAILABLE = False

# --- Configuration: Update these paths ---
CHECKPOINT_PATH = "checkpoints/best_model.pt"
CONFIG_PATH = "configs/real_config.yaml"
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SMILES_VOCAB_PATH = os.path.join(CURRENT_SCRIPT_DIR, 'training/vocab.txt')

# --- Globals ---
MODEL = None
TOKENIZER = None
NMR_TOKENIZER = None
IR_TOKENIZER = None
INFERENCE = None
DATASET = None
DEVICE = None
CONFIG = None
IR_AS_PROMPT = False
MODEL_LOADED = False

# --- Helpers ---
def create_blank_image(size=(300,150), text="No Data"):
    img = Image.new('RGB', size, (230,230,230))
    d = ImageDraw.Draw(img)
    try:
        bbox = d.textbbox((0,0), text)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        d.text(((size[0]-w)/2,(size[1]-h)/2), text, fill=(0,0,0))
    except:
        pass
    return img


def plot_token_ids_as_image(token_ids, title="Tokens", size=(300,150)):
    if not MATPLOTLIB_AVAILABLE or token_ids is None:
        return create_blank_image(text=title, size=size)
    try:
        fig, ax = plt.subplots(figsize=(size[0]/100,size[1]/100), dpi=100)
        arr = token_ids.cpu().numpy() if torch.is_tensor(token_ids) else np.array(token_ids)
        ax.bar(range(len(arr)), arr, color='gray')
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=6)
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf,format='png'); buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
    except Exception as e:
        print(f"Plot error: {e}")
        return create_blank_image(text=title, size=size)


def get_2d_image(smiles, size=(300,300)):
    if not RDKIT_AVAILABLE or not smiles:
        return create_blank_image(size=size, text="Molecule")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return create_blank_image(size=size, text="Invalid SMILES")
    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))

# New helper: plot the IR or other continuous spectrum
def plot_spectrum(spectrum, title="Spectrum", size=(300,150)):
    """Plot a continuous spectrum (e.g., IR) as a line plot and return as PIL Image."""
    if not MATPLOTLIB_AVAILABLE or spectrum is None:
        return create_blank_image(text=f"{title}\n(no data)", size=size)
    try:
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
        arr = spectrum.cpu().numpy() if torch.is_tensor(spectrum) else np.array(spectrum)
        x = np.arange(len(arr))
        ax.plot(x, arr, color='blue')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Index", fontsize=8)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.tick_params(labelsize=6)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
    except Exception as e:
        print(f"Error plotting spectrum {title}: {e}")
        return create_blank_image(text=f"Error plotting {title}", size=size)

# Helper to convert PIL image to base64 string for HTML embedding
def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Load Model & Data ---
def load_model_and_data():
    global MODEL, TOKENIZER, NMR_TOKENIZER, IR_TOKENIZER, INFERENCE, DATASET, DEVICE, CONFIG, IR_AS_PROMPT, MODEL_LOADED
    if not MODEL_FILES_AVAILABLE:
        # Disable run button on failure
        return "Model files missing", gr.update(interactive=False)
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(CONFIG_PATH) or not os.path.exists(SMILES_VOCAB_PATH):
        return "Checkpoint/config/vocab path error", gr.update(interactive=False)
    CONFIG = util_load_config(CONFIG_PATH)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TOKENIZER = SmilesTokenizer(vocab_file=SMILES_VOCAB_PATH)
    nmr_path = Path(CONFIG['data']['tokenized_dir']).parent / 'vocab.json'
    with open(nmr_path) as f: NMR_TOKENIZER = json.load(f)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    IR_AS_PROMPT, extra = util_detect_ir_as_prompt(ckpt, CONFIG)
    if IR_AS_PROMPT: IR_TOKENIZER = util_get_ir_tokenizer(CONFIG)
    model_kwargs = {
        'smiles_vocab_size': len(TOKENIZER),
        'nmr_vocab_size': max(NMR_TOKENIZER.values())+1,
        'max_seq_length': CONFIG['model']['max_seq_length'],
        'max_nmr_length': CONFIG['model']['max_nmr_length'],
        'max_memory_length': CONFIG['model']['max_memory_length'],
        'embed_dim': CONFIG['model']['embed_dim'],
        'num_heads': CONFIG['model']['num_heads'],
        'num_layers': CONFIG['model']['num_layers'],
        'dropout': CONFIG['model']['dropout'],
        'verbose': False,
        'ir_as_prompt': IR_AS_PROMPT,
        'ir_encoder_type': CONFIG['model'].get('ir_encoder_type','regular'),
        'use_loop_concat': CONFIG['model'].get('use_loop_concat',True),
        'use_rmsnorm': CONFIG['model'].get('use_rmsnorm',True)
    }
    if IR_AS_PROMPT:
        model_kwargs['ir_vocab_size'] = extra.get('ir_vocab_size', len(IR_TOKENIZER))
    MODEL = MultiModalToSMILESModel(**model_kwargs).to(DEVICE)
    MODEL.load_state_dict(ckpt['model_state_dict']); MODEL.eval()
    INFERENCE = ModelInference(MODEL, TOKENIZER, DEVICE, ir_as_prompt=IR_AS_PROMPT)
    DATASET = SimpleSpectralSmilesDataset(
        data_dir=CONFIG['data']['tokenized_dir'], split=CONFIG['data'].get('val_split','val'),
        smiles_tokenizer=TOKENIZER, spectral_tokenizer=NMR_TOKENIZER,
        max_smiles_len=CONFIG['model']['max_seq_length'], max_nmr_len=CONFIG['model']['max_nmr_length'],
        ir_as_prompt=IR_AS_PROMPT, ir_tokenizer=IR_TOKENIZER
    )
    MODEL_LOADED = True
    # Enable run button after successful load
    return "Loaded model and dataset", gr.update(interactive=True)

# --- Core Function for Beam & Entropix ---
def get_random_beam_entropix():
    global DATASET, DEVICE, IR_AS_PROMPT, MODEL_LOADED
    if not MODEL_LOADED:
        return "", create_blank_image(), create_blank_image(), "", "", "", "", "Model not loaded."
    # Sample data
    idx = random.randrange(len(DATASET))
    try:
        _, (ir_data, _), nmr_data, _ = DATASET[idx]
    except ValueError:
        _, nmr_data, _ = DATASET[idx]
        ir_data = None
    # Prepare tensors
    nmr_tensor = nmr_data.unsqueeze(0).to(DEVICE) if nmr_data.ndim == 1 else nmr_data.to(DEVICE)
    if nmr_tensor.ndim == 1: nmr_tensor = nmr_tensor.unsqueeze(0)
    ir_tensor = None
    if IR_AS_PROMPT and ir_data is not None:
        ir_tensor = ir_data.unsqueeze(0).to(DEVICE) if ir_data.ndim == 1 else ir_data.to(DEVICE)
        if ir_tensor.ndim == 1: ir_tensor = ir_tensor.unsqueeze(0)
    # Clean target SMILES
    target_clean = DATASET.targets[idx].replace(" ", "")
    # Input plots
    nmr_plot = plot_token_ids_as_image(nmr_tensor.squeeze(), "NMR Tokens")
    if IR_AS_PROMPT and ir_tensor is not None:
        ir_plot = plot_spectrum(ir_tensor.squeeze(), "IR Spectrum")
    else:
        ir_plot = create_blank_image(text="IR Spectrum")
    # Beam: get top-5 and pick best by Tanimoto
    beam_seqs = INFERENCE.beam_search(nmr_tokens=nmr_tensor, ir_data=ir_tensor,
                                      max_len=CONFIG['model']['max_seq_length'], beam_width=5)
    beam_metrics_list = evaluate_predictions(beam_seqs, [target_clean] * len(beam_seqs))
    best_beam_idx = max(range(len(beam_metrics_list)), key=lambda i: beam_metrics_list[i]['tanimoto'])
    best_beam_smiles = beam_seqs[best_beam_idx].replace(' ', '')
    best_beam_img = get_2d_image(best_beam_smiles, size=(200,200))
    beam_b64 = pil_to_base64(best_beam_img)
    bm = beam_metrics_list[best_beam_idx]
    beam_html = f"<h3>Beam Best (Tanimoto={bm['tanimoto']:.4f})</h3>"
    beam_html += f"<img src='data:image/png;base64,{beam_b64}' style='width:200px;height:200px'/><br>{best_beam_smiles}"
    beam_metrics_html = "<ul>" + "".join(
        f"<li>{k}: {bm[k]}</li>" for k in ['valid_pred','exact_match','tanimoto','#mcs/#target','ecfp6_iou']
    ) + "</ul>"
    # Entropix: get top-5 and pick best by Tanimoto
    ent_seqs, ent_loops = INFERENCE.entropix_decode(nmr_tokens=nmr_tensor, ir_data=ir_tensor,
                                                   max_len=CONFIG['model']['max_seq_length'], top_k=5, max_loops=10)
    ent_metrics_list = evaluate_predictions(ent_seqs, [target_clean] * len(ent_seqs))
    best_ent_idx = max(range(len(ent_metrics_list)), key=lambda i: ent_metrics_list[i]['tanimoto'])
    best_ent_smiles = ent_seqs[best_ent_idx].replace(' ', '')
    best_ent_img = get_2d_image(best_ent_smiles, size=(200,200))
    ent_b64 = pil_to_base64(best_ent_img)
    em = ent_metrics_list[best_ent_idx]
    ent_html = f"<h3>Entropix Best (Tanimoto={em['tanimoto']:.4f})</h3>"
    ent_html += f"<img src='data:image/png;base64,{ent_b64}' style='width:200px;height:200px'/><br>{best_ent_smiles}"
    ent_metrics_html = "<ul>" + "".join(
        f"<li>{k}: {em[k]}</li>" for k in ['valid_pred','exact_match','tanimoto','#mcs/#target','ecfp6_iou']
    ) + "</ul>"
    status = "Completed. Showing best Beam and Entropix only."
    return target_clean, nmr_plot, ir_plot, beam_html, ent_html, beam_metrics_html, ent_metrics_html, status

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Beam Search vs Entropix Visualization")
    load_status = gr.Textbox(label="Load Status", interactive=False)
    run_btn = gr.Button("Run Beam & Entropix", interactive=False)
    # Input spectra and target in the first row
    with gr.Row():
        with gr.Column():
            nmr_out = gr.Image(label="NMR Plot", type="pil")
            ir_out = gr.Image(label="IR Plot", type="pil")
            target_out = gr.Textbox(label="Target SMILES")

    # Predictions and metrics below
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Beam Search Predictions")
            beam_out = gr.HTML(label="Beam Search Results")
            gr.Markdown("### Beam Search Metrics")
            beam_metrics_out = gr.HTML(label="Beam Metrics")
            gr.Markdown("### Entropix Predictions")
            ent_out = gr.HTML(label="Entropix Results")
            gr.Markdown("### Entropix Metrics")
            ent_metrics_out = gr.HTML(label="Entropix Metrics")
        with gr.Column():
            status_out = gr.Textbox(label="Status", interactive=False)

    # Trigger inference and update all outputs
    run_btn.click(
        fn=get_random_beam_entropix,
        inputs=[],
        outputs=[target_out, nmr_out, ir_out, beam_out, ent_out, beam_metrics_out, ent_metrics_out, status_out]
    )
    demo.load(load_model_and_data, outputs=[load_status, run_btn])

demo.launch() 