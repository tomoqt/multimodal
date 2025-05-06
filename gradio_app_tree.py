import gradio as gr
import torch
import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import io
import matplotlib.pyplot as plt
import base64
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available, 2D drawing disabled.")

# Model & Dataset imports
from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer
from inference.inference import ModelInference, DecodingStrategy
from test_inference import (
    load_config as util_load_config,
    detect_ir_as_prompt as util_detect_ir_as_prompt,
    get_ir_tokenizer as util_get_ir_tokenizer,
    SimpleSpectralSmilesDataset
)
import animated_inference

# --- Configuration ---
CHECKPOINT_PATH = "checkpoints/best_model.pt"
CONFIG_PATH     = "configs/real_config.yaml"
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SMILES_VOCAB_PATH  = os.path.join(CURRENT_SCRIPT_DIR, 'training/vocab.txt')

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
def create_blank_image(size=(200,150), text="No Data"):
    img = Image.new('RGB', size, (240,240,240))
    d = ImageDraw.Draw(img)
    try:
        bbox = d.textbbox((0,0), text)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        d.text(((size[0]-w)/2,(size[1]-h)/2), text, fill=(0,0,0))
    except:
        pass
    return img

# Plot NMR token IDs
def plot_token_ids_as_image(token_ids, title="NMR Tokens", size=(300,150)):
    if token_ids is None:
        return create_blank_image(text=title, size=size)
    fig, ax = plt.subplots(figsize=(size[0]/100,size[1]/100), dpi=100)
    arr = token_ids.cpu().numpy() if torch.is_tensor(token_ids) else np.array(token_ids)
    ax.bar(range(len(arr)), arr, color='gray')
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf,format='png'); buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

# Plot continuous IR
def plot_spectrum(spectrum, title="IR Spectrum", size=(300,150)):
    if spectrum is None:
        return create_blank_image(text=title, size=size)
    fig, ax = plt.subplots(figsize=(size[0]/100,size[1]/100), dpi=100)
    arr = spectrum.cpu().numpy() if torch.is_tensor(spectrum) else np.array(spectrum)
    x = np.arange(len(arr))
    ax.plot(x, arr, color='blue')
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf,format='png'); buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

# Convert PIL to base64 for hidden state
def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf,format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Load model & data once
def load_model_and_data_global():
    global MODEL, TOKENIZER, NMR_TOKENIZER, IR_TOKENIZER, INFERENCE, DATASET, DEVICE, CONFIG, IR_AS_PROMPT, MODEL_LOADED
    if MODEL_LOADED:
        return "Model already loaded", gr.update(interactive=True)
    # Check paths
    if not os.path.exists(CHECKPOINT_PATH):
        return f"Checkpoint not found: {CHECKPOINT_PATH}", gr.update(interactive=False)
    if not os.path.exists(CONFIG_PATH):
        return f"Config not found: {CONFIG_PATH}", gr.update(interactive=False)
    if not os.path.exists(SMILES_VOCAB_PATH):
        return f"Vocab not found: {SMILES_VOCAB_PATH}", gr.update(interactive=False)
    # Load config
    CONFIG = util_load_config(CONFIG_PATH)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load tokenizers
    TOKENIZER = SmilesTokenizer(vocab_file=SMILES_VOCAB_PATH)
    vocab_json = Path(CONFIG['data']['tokenized_dir']).parent / 'vocab.json'
    with open(vocab_json) as f:
        NMR_TOKENIZER = json.load(f)
    # Load checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    IR_AS_PROMPT, extra = util_detect_ir_as_prompt(ckpt, CONFIG)
    IR_TOKENIZER = util_get_ir_tokenizer(CONFIG) if IR_AS_PROMPT else None
    # Build model
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
    # Load dataset
    DATASET = SimpleSpectralSmilesDataset(
        data_dir=CONFIG['data']['tokenized_dir'], split=CONFIG['data'].get('val_split','val'),
        smiles_tokenizer=TOKENIZER, spectral_tokenizer=NMR_TOKENIZER,
        max_smiles_len=CONFIG['model']['max_seq_length'], max_nmr_len=CONFIG['model']['max_nmr_length'],
        ir_as_prompt=IR_AS_PROMPT, ir_tokenizer=IR_TOKENIZER
    )
    MODEL_LOADED = True
    return "Model loaded", gr.update(interactive=True)

# Format active states
def format_beam_state(active_list):
    lines = []
    for i,(tok_ids, score) in enumerate(active_list):
        # remove BOS
        clean = tok_ids[1:] if tok_ids[0]==TOKENIZER.cls_token_id else tok_ids
        smi = TOKENIZER.decode(clean).replace(' ','')
        lines.append(f"{i+1}: {smi} (logp={score:.2f})")
    return "\n".join(lines)

def format_ent_state(active_list):
    lines=[]
    for i,(tok_ids, loops) in enumerate(active_list):
        clean = tok_ids[1:] if tok_ids[0]==TOKENIZER.cls_token_id else tok_ids
        smi = TOKENIZER.decode(clean).replace(' ','')
        loop_sum = sum(loops)
        lines.append(f"{i+1}: {smi} (loops={loop_sum})")
    return "\n".join(lines)

def draw_beam_tree_image(active_list, size=(400,300)):
    G = nx.DiGraph()
    root = "ROOT"
    G.add_node(root)
    # Detect entropix state (loops list) vs beam (score)
    is_ent = bool(active_list and isinstance(active_list[0][1], list))
    node_metrics = {}
    for tok_ids, metric in active_list:
        parent = root
        # compute metric value: sum loops for entropix, or score directly
        value = sum(metric) if is_ent else metric
        for tid in tok_ids[1:]:
            tok = TOKENIZER.convert_ids_to_tokens(tid)
            if isinstance(tok, list): tok = tok[0]
            node_id = f"{parent}|{tid}"
            if not G.has_node(node_id):
                G.add_node(node_id, label=tok)
            if not G.has_edge(parent, node_id):
                G.add_edge(parent, node_id)
            parent = node_id
        # assign metric to leaf node
        node_metrics[parent] = value
    pos = graphviz_layout(G, prog='dot')
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    if is_ent:
        # Color-code nodes by discrete Entropix state: branch, argmax, or loop
        # Determine state per active sequence: 0=branch, 1=argmax, 2=loop
        state_codes = {}
        num_active = len(active_list)
        for tok_ids, loops in active_list:
            leaf = "ROOT"
            for tid in tok_ids[1:]:
                leaf = f"{leaf}|{tid}"
            last_loops = loops[-1] if loops else 1
            if last_loops > 1:
                code = 2
            else:
                if num_active > 1:
                    code = 0
                else:
                    code = 1
            state_codes[leaf] = code
        cmap = ListedColormap(['blue','green','orange'])
        norm = BoundaryNorm([0,1,2,3], cmap.N)
        colors = []
        for n in G.nodes():
            code = state_codes.get(n, -1)
            if code == -1:
                colors.append('lightgray')
            else:
                colors.append(cmap(code))
        nx.draw(
            G,
            pos,
            with_labels=False,
            arrows=False,
            node_size=100,
            node_color=colors
        )
        # Add legend for Entropix states
        handles = [
            mpatches.Patch(color='blue', label='branch'),
            mpatches.Patch(color='green', label='argmax'),
            mpatches.Patch(color='orange', label='loop'),
        ]
        plt.legend(handles=handles, title='Entropix State', loc='best')
    else:
        nx.draw(G, pos, with_labels=False, arrows=False, node_size=100)
    # draw labels on nodes
    for n, d in G.nodes(data=True):
        plt.text(pos[n][0], pos[n][1], d.get('label',''), fontsize=8)
    plt.axis('off')
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

# Sample and init
def sample_and_init():
    # sample data
    idx = random.randrange(len(DATASET))
    tt, (ir_d,_), nmr_d, _ = DATASET[idx] if IR_AS_PROMPT else DATASET[idx]
    nmr_tensor = nmr_d.unsqueeze(0).to(DEVICE) if nmr_d.ndim==1 else nmr_d.to(DEVICE)
    if nmr_tensor.ndim==1: nmr_tensor=nmr_tensor.unsqueeze(0)
    if IR_AS_PROMPT:
        ir_tensor = (ir_d.unsqueeze(0).to(DEVICE) if (ir_d is not None and ir_d.ndim==1) else ir_d.to(DEVICE))
    else:
        ir_tensor = None
    target = DATASET.targets[idx].replace(' ','')
    # plots
    nmr_plot = plot_token_ids_as_image(nmr_tensor.squeeze())
    ir_plot  = plot_spectrum(ir_tensor.squeeze()) if IR_AS_PROMPT and ir_tensor is not None else create_blank_image(text='IR Spectrum')
    # create generators
    beam_gen = animated_inference.beam_search_tree(INFERENCE, nmr_tensor, ir_tensor)
    ent_gen  = animated_inference.entropix_tree(INFERENCE, nmr_tensor, ir_tensor)
    # get initial states
    beam_state = next(beam_gen)
    ent_state  = next(ent_gen)
    beam_text  = format_beam_state(beam_state)
    ent_text   = format_ent_state(ent_state)
    beam_tree_img = draw_beam_tree_image(beam_state)
    ent_tree_img  = draw_beam_tree_image(ent_state)
    step = 0
    status = f"Step {step}"
    # enable next button and return tree images
    return target, nmr_plot, ir_plot, beam_tree_img, ent_tree_img, beam_text, ent_text, status, beam_gen, ent_gen, step, gr.update(interactive=True), gr.update(interactive=True)

# Advance one step
def next_step(beam_gen, ent_gen, step):
    try:
        beam_state = next(beam_gen)
    except StopIteration:
        pass
    try:
        ent_state  = next(ent_gen)
    except StopIteration:
        pass
    beam_text = format_beam_state(beam_state)
    ent_text  = format_ent_state(ent_state)
    step +=1
    status = f"Step {step}"
    # update and return tree images
    beam_tree_img = draw_beam_tree_image(beam_state)
    ent_tree_img  = draw_beam_tree_image(ent_state)
    return beam_tree_img, ent_tree_img, beam_text, ent_text, status, beam_gen, ent_gen, step

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Beam vs Entropix Tree Animation")
    load_status = gr.Textbox(interactive=False, label="Load Status")
    sample_btn  = gr.Button("🧪 Get Random Sample", interactive=False)
    next_btn    = gr.Button("⏭ Next Step", interactive=False)
    auto_btn    = gr.Button("▶️ Auto-Advance", interactive=False)

    with gr.Row():
        target_out    = gr.Textbox(label="Target SMILES")
        nmr_plot_out  = gr.Image(label="NMR Plot", type="pil")
        ir_plot_out   = gr.Image(label="IR Spectrum", type="pil")

    # Display tree graph and active sequences
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Beam Search Tree")
            beam_tree_out = gr.Image(label="Beam Tree", type="pil")
            beam_out      = gr.Textbox(label="Beam Active Sequences", lines=8)
        with gr.Column():
            gr.Markdown("### Entropix Tree")
            ent_tree_out  = gr.Image(label="Entropix Tree", type="pil")
            ent_out       = gr.Textbox(label="Entropix Active Sequences", lines=8)

    status_out  = gr.Textbox(interactive=False, label="Step")

    beam_state = gr.State(None)
    ent_state  = gr.State(None)
    step_state = gr.State(0)

    demo.load(load_model_and_data_global, outputs=[load_status, sample_btn])

    sample_btn.click(
        fn=sample_and_init,
        inputs=[],
        outputs=[target_out, nmr_plot_out, ir_plot_out, beam_tree_out, ent_tree_out, beam_out, ent_out, status_out, beam_state, ent_state, step_state, next_btn, auto_btn]
    )

    next_btn.click(
        fn=next_step,
        inputs=[beam_state, ent_state, step_state],
        outputs=[beam_tree_out, ent_tree_out, beam_out, ent_out, status_out, beam_state, ent_state, step_state]
    )

    auto_btn.click(
        fn=next_step,
        inputs=[beam_state, ent_state, step_state],
        outputs=[beam_tree_out, ent_tree_out, beam_out, ent_out, status_out, beam_state, ent_state, step_state]
    )

    demo.launch() 