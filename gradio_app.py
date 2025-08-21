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
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not found. 2D molecule visualization will not be available.")

# --- Model/Helper Imports (ensure these paths are correct) ---
# Assuming 'models', 'inference' are subdirectories and 'test_inference.py' is accessible
# Add project root to sys.path if necessary, or ensure PYTHONPATH is set
# import sys
# PROJECT_ROOT_DIR = "/path/to/your/project_root" # MODIFY IF NEEDED
# sys.path.append(PROJECT_ROOT_DIR)

try:
    from models.multimodal_to_smiles import MultiModalToSMILESModel
    from models.smiles_tokenizer import SmilesTokenizer
    from inference.inference import ModelInference, DecodingStrategy
    # Functions from test_inference.py (or include them directly)
    from test_inference import load_config as util_load_config
    from test_inference import get_ir_tokenizer as util_get_ir_tokenizer
    from test_inference import detect_ir_as_prompt as util_detect_ir_as_prompt
    from test_inference import SimpleSpectralSmilesDataset, evaluate_predictions
    MODEL_FILES_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import model files: {e}. Please ensure paths are correct and dependencies are installed.")
    MODEL_FILES_AVAILABLE = False

# --- User Configuration: MODIFY THESE PATHS ---
CHECKPOINT_PATH = "checkpoints/100k.pt"  # e.g., 'checkpoints/model.pth'
CONFIG_PATH = "configs/real_config.yaml"        # e.g., 'configs/test_config.yaml'
# SMILES_VOCAB_PATH is often relative to the script or a known 'training' dir
# Defaulting to a common pattern, adjust if your vocab.txt is elsewhere.
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SMILES_VOCAB_PATH = 'training/vocab.txt'

# --- Global Variables ---
MODEL = None
TOKENIZER = None
NMR_TOKENIZER = None
IR_TOKENIZER = None
INFERENCE = None
DATASET = None
DEVICE = None
CONFIG = None
IR_AS_PROMPT = False
MODEL_LOADED_SUCCESSFULLY = False

GRADIO_MAX_REFINE_LOOPS = 30 # Number of internal model loops to show step-by-step
GRADIO_MAX_DISPLAY_STEPS = 195 # Max autoregressive steps to show in Gradio

# --- Helper Functions ---
def get_2d_image(smiles_string, size=(300, 300)):
    if not RDKIT_AVAILABLE or not smiles_string or not isinstance(smiles_string, str):
        return None
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        try:
            # Try Cairo for better quality if available
            d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            d.DrawMolecule(mol)
            d.FinishDrawing()
            png_data = d.GetDrawingText()
            return Image.open(io.BytesIO(png_data))
        except Exception:  # Fallback if Cairo fails or not fully set up
            try:
                return Draw.MolToImage(mol, size=size)
            except Exception:
                return None # Final fallback
    return None

def create_blank_image(size=(300, 300), text="Preview"):
    img = Image.new('RGB', size, color = (220, 220, 220))
    d = ImageDraw.Draw(img)
    try:
        # Basic text rendering
        text_bbox = d.textbbox((0,0), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (size[0] - text_width) / 2
        y = (size[1] - text_height) / 2
        d.text((x, y), text, fill=(0,0,0))
    except Exception:
        pass # Ignore if text rendering fails
    return img

def plot_token_ids_as_image(token_ids, title="Spectral Tokens", size=(300,150)):
    if not MATPLOTLIB_AVAILABLE or token_ids is None:
        return create_blank_image(text=f"{title}\n(Plotting disabled or no data)", size=size)
    
    try:
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
        if torch.is_tensor(token_ids):
            token_ids_np = token_ids.cpu().numpy()
        elif isinstance(token_ids, list):
            token_ids_np = np.array(token_ids)
        else:
            token_ids_np = token_ids # Assume it's already a numpy array

        # Simple bar plot of token values
        ax.bar(range(len(token_ids_np)), token_ids_np)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Token Index", fontsize=8)
        ax.set_ylabel("Token ID", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig) # Close the figure to free memory
        return img
    except Exception as e:
        print(f"Error plotting tokens for {title}: {e}")
        return create_blank_image(text=f"Error plotting {title}", size=size)

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

BLANK_NMR_IMAGE = create_blank_image(text="NMR Input")
BLANK_IR_IMAGE = create_blank_image(text="IR Input")
BLANK_MOL_IMAGE = create_blank_image(text="Molecule")

def format_metrics_html(metrics_dict, title="Metrics"):
    if not metrics_dict:
        return f"<h3>{title}</h3><p style='font-size: 0.9em;'>N/A</p>"
    
    metric_keys_to_display = ['valid_pred', 'exact_match', 'tanimoto', '#mcs/#target', 'ecfp6_iou']
    html_items = ""
    for k in metric_keys_to_display:
        value = metrics_dict.get(k)
        # Format float values nicely
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        elif isinstance(value, (np.float32, np.float64)): # Numpy floats
            value_str = f"{float(value):.4f}"
        else:
            value_str = str(value)
        html_items += f"<li>{k}: {value_str}</li>"
            
    return f"<h3>{title}</h3><ul style='font-size: 0.9em; margin-top: 0; padding-left: 20px; list-style-type: none;'>{html_items}</ul>"

BLANK_METRICS_HTML = format_metrics_html(None, "Prediction Metrics")

def load_model_and_data_global():
    global MODEL, TOKENIZER, NMR_TOKENIZER, IR_TOKENIZER, INFERENCE, DATASET, DEVICE, CONFIG, IR_AS_PROMPT, MODEL_LOADED_SUCCESSFULLY

    if not MODEL_FILES_AVAILABLE:
        return "Model definition files not found. Cannot load model."
    if not os.path.exists(CHECKPOINT_PATH):
        return f"Checkpoint file not found: {CHECKPOINT_PATH}"
    if not os.path.exists(CONFIG_PATH):
        return f"Config file not found: {CONFIG_PATH}"
    if not os.path.exists(SMILES_VOCAB_PATH):
         return f"SMILES vocab file not found: {SMILES_VOCAB_PATH}"

    try:
        CONFIG = util_load_config(CONFIG_PATH)
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        TOKENIZER = SmilesTokenizer(vocab_file=SMILES_VOCAB_PATH)

        nmr_vocab_path = Path(CONFIG['data']['tokenized_dir']).parent / 'vocab.json'
        if not nmr_vocab_path.exists():
            return f"NMR vocabulary not found at {nmr_vocab_path}"
        with open(nmr_vocab_path) as f:
            NMR_TOKENIZER = json.load(f)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        auto_ir_as_prompt, extra_params = util_detect_ir_as_prompt(checkpoint, CONFIG)
        IR_AS_PROMPT = auto_ir_as_prompt

        IR_TOKENIZER = None
        if IR_AS_PROMPT:
            IR_TOKENIZER = util_get_ir_tokenizer(CONFIG)

        smiles_vocab_size = len(TOKENIZER)
        token_ids = list(NMR_TOKENIZER.values())
        nmr_vocab_size = max(token_ids) + 1 if token_ids else 0
        
        model_kwargs = {
            'smiles_vocab_size': smiles_vocab_size,
            'nmr_vocab_size': nmr_vocab_size,
            'max_seq_length': CONFIG['model']['max_seq_length'],
            'max_nmr_length': CONFIG['model']['max_nmr_length'],
            'max_memory_length': CONFIG['model']['max_memory_length'],
            'embed_dim': CONFIG['model']['embed_dim'],
            'num_heads': CONFIG['model']['num_heads'],
            'num_layers': CONFIG['model']['num_layers'],
            'dropout': CONFIG['model']['dropout'],
            'verbose': False,
            'use_stablemax': CONFIG['model'].get('use_stablemax', False),
            'ir_as_prompt': IR_AS_PROMPT,
            'ir_encoder_type': CONFIG['model'].get('ir_encoder_type', 'regular'),
            'max_loops': CONFIG['model'].get('max_loops', GRADIO_MAX_REFINE_LOOPS), # Use config's max_loops
            'loops_representation': True, # Crucial for step-by-step
            'use_loop_concat': CONFIG['model'].get('use_loop_concat', True),
            'use_rmsnorm': CONFIG['model'].get('use_rmsnorm', True),
            'vanilla_mode': CONFIG['model'].get('vanilla_mode', False)
        }
        if IR_AS_PROMPT:
            model_kwargs['ir_vocab_size'] = extra_params.get('ir_vocab_size', len(IR_TOKENIZER) if IR_TOKENIZER else 0)

        MODEL = MultiModalToSMILESModel(**model_kwargs).to(DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.eval()

        INFERENCE = ModelInference(MODEL, TOKENIZER, DEVICE, ir_as_prompt=IR_AS_PROMPT)

        DATASET = SimpleSpectralSmilesDataset(
            data_dir=CONFIG['data']['tokenized_dir'],
            split=CONFIG['data'].get('val_split', 'val'), # Use val split for demo
            smiles_tokenizer=TOKENIZER,
            spectral_tokenizer=NMR_TOKENIZER,
            max_smiles_len=CONFIG['model']['max_seq_length'],
            max_nmr_len=CONFIG['model']['max_nmr_length'],
            ir_as_prompt=IR_AS_PROMPT,
            ir_tokenizer=IR_TOKENIZER
        )
        if not DATASET or len(DATASET) == 0:
            return "Dataset is empty or failed to load. Check config and data."

        MODEL_LOADED_SUCCESSFULLY = True
        return "Model and data loaded successfully."
    except Exception as e:
        MODEL_LOADED_SUCCESSFULLY = False
        return f"Error loading model/data: {str(e)}"

def predict_step_by_step_gradio(nmr_data_tensor, ir_data_tensor, target_smiles_str):
    global INFERENCE, CONFIG, TOKENIZER, GRADIO_MAX_DISPLAY_STEPS, BLANK_MOL_IMAGE, BLANK_METRICS_HTML

    # Strip spaces from the displayed target SMILES as well
    target_clean = target_smiles_str.replace(" ", "")
    yield target_clean, "Starting...", BLANK_MOL_IMAGE, "Target SMILES loaded. Beginning autoregressive generation...", BLANK_METRICS_HTML

    last_valid_image_for_display = BLANK_MOL_IMAGE
    generated_smiles_at_eos = None
    max_generation_len = CONFIG['model'].get('max_seq_length', GRADIO_MAX_DISPLAY_STEPS)
    current_metrics_html = BLANK_METRICS_HTML

    # This means we are not focusing on the model's internal refinement loops here,
    # but on the token-by-token generation.
    step_iterator = INFERENCE.greedy_decode_step_by_step(
        nmr_tokens=nmr_data_tensor,
        ir_data=ir_data_tensor,
        max_len=max_generation_len, 
        num_loops=15
    )

    for i, current_tokens_tensor in enumerate(step_iterator):
        current_tokens_list = current_tokens_tensor.cpu().numpy().tolist()
        # Filter out special BOS/SEP tokens
        filtered_ids = [tid for tid in current_tokens_list if tid not in (TOKENIZER.cls_token_id, TOKENIZER.sep_token_id)]
        # Decode remaining token IDs to SMILES string and remove spaces
        decoded = TOKENIZER.decode(filtered_ids) if filtered_ids else ""
        current_raw_smiles = decoded.replace(" ", "")
        
        current_image = get_2d_image(current_raw_smiles)

        display_smiles = current_raw_smiles
        img_to_show = BLANK_MOL_IMAGE
        # Determine status message
        if not filtered_ids:
            status = f"Step {i}: no tokens generated yet"
        else:
            last_token_id = current_tokens_list[-1]
            if last_token_id == TOKENIZER.sep_token_id:
                status = f"Step {i}: EOS generated"
            else:
                # Convert last token ID to its token string
                try:
                    token_repr = TOKENIZER.convert_ids_to_tokens(last_token_id)
                    token_name = token_repr[0] if isinstance(token_repr, list) else token_repr
                except Exception:
                    token_name = str(last_token_id)
                status = f"Step {i}: appended '{token_name}' -> {display_smiles}"

        if current_image:
            img_to_show = current_image
            last_valid_image_for_display = current_image
        else:
            img_to_show = last_valid_image_for_display

        yield target_clean, display_smiles, img_to_show, status, current_metrics_html
        
        is_eos = (len(current_tokens_list) > 0 and current_tokens_list[-1] == TOKENIZER.sep_token_id)

        if is_eos:
            generated_smiles_at_eos = display_smiles
            status = f"Step {i}: EOS generated. Final: {display_smiles}"
            final_img = get_2d_image(display_smiles) or last_valid_image_for_display
            # Calculate metrics
            metrics_list = evaluate_predictions([display_smiles], [target_clean])
            if metrics_list:
                current_metrics_html = format_metrics_html(metrics_list[0], "Final Prediction Metrics")
            else:
                current_metrics_html = format_metrics_html(None, "Final Prediction Metrics")
            yield target_clean, display_smiles, final_img, status, current_metrics_html
            break 

        if i >= GRADIO_MAX_DISPLAY_STEPS -1: 
            status = f"Reached max display steps ({GRADIO_MAX_DISPLAY_STEPS})."
            if not is_eos : status += " Full generation may differ."
            # Calculate metrics for the current (possibly incomplete) SMILES
            metrics_list = evaluate_predictions([display_smiles], [target_clean])
            if metrics_list:
                current_metrics_html = format_metrics_html(metrics_list[0], "Metrics at Max Steps")
            else:
                current_metrics_html = format_metrics_html(None, "Metrics at Max Steps")
            # Yield current state before breaking display loop
            yield target_clean, display_smiles, img_to_show, status, current_metrics_html 
            break
        
        time.sleep(0.15) # Adjust for speed

    # After the loop, if it finished due to max_len in underlying iterator but not EOS or max display steps
    if not generated_smiles_at_eos and i < GRADIO_MAX_DISPLAY_STEPS -1:
        # This means the underlying greedy_decode_step_by_step finished (e.g. max_len reached)
        # The last state was already yielded within the loop.
        # We can add a final status message if needed.
        final_status_message = f"Generation completed after {i+1} steps (max model length or other limit reached without EOS)."
        # Re-calculate metrics for the final state
        metrics_list = evaluate_predictions([display_smiles], [target_clean])
        if metrics_list:
            current_metrics_html = format_metrics_html(metrics_list[0], "Final Prediction Metrics")
        else:
            current_metrics_html = format_metrics_html(None, "Final Prediction Metrics")
        # Re-yield the last known state with this new message and metrics
        yield target_clean, display_smiles, img_to_show, final_status_message, current_metrics_html


def get_random_sample_and_predict_gradio():
    global DATASET, DEVICE, IR_AS_PROMPT, MODEL_LOADED_SUCCESSFULLY, BLANK_METRICS_HTML
    if not MODEL_LOADED_SUCCESSFULLY or DATASET is None:
        # Ensure all outputs are updated, including the new metrics display
        return "Dataset not loaded or model error.", BLANK_MOL_IMAGE, BLANK_NMR_IMAGE, BLANK_IR_IMAGE, "N/A", BLANK_MOL_IMAGE, "Error: Load data and model first.", BLANK_METRICS_HTML

    idx = random.randint(0, len(DATASET) - 1)
    
    # Dataset returns: target_tokens, (ir_input_ids, ir_attention_mask), nmr_input_ids, nmr_attention_mask
    # We need ir_input_ids and nmr_input_ids
    try:
        target_tokens, (ir_data, _), nmr_data, _ = DATASET[idx]
    except ValueError: # If dataset item is not a tuple of 4 (e.g. if IR not present and dataset returns 3 items)
         target_tokens, nmr_data, _ = DATASET[idx] # Assuming this structure if IR is missing
         ir_data = None

    # NMR DATA
    nmr_tensor = nmr_data.to(DEVICE) # Assuming nmr_data from dataset is a tensor
    if nmr_tensor.ndim == 0: # Was scalar
        nmr_tensor = nmr_tensor.view(1, 1) # Reshape to [1,1]
    elif nmr_tensor.ndim == 1: # Was 1D array [S]
        nmr_tensor = nmr_tensor.unsqueeze(0) # Reshape to [1,S]
    # If nmr_tensor.ndim >= 2, assume it's already like [1,S] for a single sample.

    # IR DATA
    ir_tensor = None
    if IR_AS_PROMPT and ir_data is not None:
        ir_tensor = ir_data.to(DEVICE) # Assuming ir_data from dataset is a tensor
        if ir_tensor.ndim == 0: # Was scalar
            ir_tensor = ir_tensor.view(1, 1) # Reshape to [1,1]
        elif ir_tensor.ndim == 1: # Was 1D array [S]
            ir_tensor = ir_tensor.unsqueeze(0) # Reshape to [1,S]
        # If ir_tensor.ndim >= 2, assume it's already like [1,S] for a single sample.


    target_smiles_str = DATASET.targets[idx] # Assumes DATASET.targets is populated
    # Remove any spaces from the target SMILES
    target_smiles_str = target_smiles_str.replace(" ", "")
    target_mol_image = get_2d_image(target_smiles_str) or BLANK_MOL_IMAGE # Generate target molecule image
    
    # Initial state to clear previous outputs and show target SMILES and input plots
    nmr_plot_img = plot_token_ids_as_image(nmr_tensor.squeeze(), "NMR Tokens")
    # Plot IR as continuous spectrum
    if IR_AS_PROMPT and ir_tensor is not None:
        ir_plot_img = plot_spectrum(ir_tensor.squeeze(), "IR Spectrum")
    else:
        ir_plot_img = create_blank_image(text="IR Spectrum")
    
    # Yield initial plots and target SMILES before starting step-by-step prediction
    # Include BLANK_METRICS_HTML for the new metrics display component
    yield target_smiles_str, target_mol_image, nmr_plot_img, ir_plot_img, "Starting...", BLANK_MOL_IMAGE, "Inputs loaded. Starting generation...", BLANK_METRICS_HTML

    # Now yield from the step-by-step prediction generator
    # The step-by-step generator now yields: target_smiles, predicted_smiles, molecule_image, status, metrics_html
    # We need to ensure the output structure matches the UI components
    for step_outputs in predict_step_by_step_gradio(nmr_tensor, ir_tensor, target_smiles_str):
        # step_outputs is expected to be (target_smiles, predicted_smiles, molecule_image, status, metrics_html)
        step_target_smiles, step_predicted_smiles, step_mol_image, step_status, step_metrics_html = step_outputs
        yield step_target_smiles, target_mol_image, nmr_plot_img, ir_plot_img, step_predicted_smiles, step_mol_image, step_status, step_metrics_html


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multimodal to SMILES Prediction (Step-by-Step)")
    gr.Markdown(f"Ensure RDKit is installed for 2D visualization (RDKit available: {RDKIT_AVAILABLE}).")
    gr.Markdown(f"Model files available: {MODEL_FILES_AVAILABLE}. Check console for errors if False.")

    load_status_textbox = gr.Textbox(label="Model Loading Status", interactive=False)

    with gr.Row():
        sample_button = gr.Button("🧪 Get Random Sample & Predict", variant="primary", interactive=MODEL_LOADED_SUCCESSFULLY)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Spectra")
            nmr_plot_output = gr.Image(label="NMR Data (Tokens Plot)", type="pil", value=BLANK_NMR_IMAGE, interactive=False)
            ir_plot_output = gr.Image(label="IR Spectrum", type="pil", value=BLANK_IR_IMAGE, interactive=False, visible=IR_AS_PROMPT)
            target_smiles_output = gr.Textbox(label="Target SMILES (from Dataset)", interactive=False)
            target_mol_image_output = gr.Image(label="Target 2D Structure", type="pil", value=BLANK_MOL_IMAGE, interactive=False)
        
        with gr.Column(scale=2):
            gr.Markdown("### Prediction Process")
            predicted_smiles_output = gr.Textbox(label="Predicted SMILES (Step-by-Step)", interactive=False, lines=2)
            molecule_image_output = gr.Image(label="2D Molecular Structure", type="pil", value=BLANK_MOL_IMAGE, interactive=False)
            status_predict_output = gr.Textbox(label="Prediction Status", interactive=False, lines=2)
            metrics_output = gr.HTML(label="Prediction Metrics", value=BLANK_METRICS_HTML)

    sample_button.click(
        fn=get_random_sample_and_predict_gradio,
        inputs=[],
        outputs=[target_smiles_output, target_mol_image_output, nmr_plot_output, ir_plot_output, predicted_smiles_output, molecule_image_output, status_predict_output, metrics_output]
    )

    # Load model when the app starts
    # The result of load_model_and_data_global will be displayed in load_status_textbox
    # sample_button interactivity will be updated based on MODEL_LOADED_SUCCESSFULLY
    def startup_wrapper():
        load_message = load_model_and_data_global()
        # Update button interactivity based on load success
        sample_button.interactive = MODEL_LOADED_SUCCESSFULLY
        # Return a dictionary to update multiple components
        return {
            load_status_textbox: load_message,
            sample_button: gr.update(interactive=MODEL_LOADED_SUCCESSFULLY)
        }

    demo.load(startup_wrapper, outputs=[load_status_textbox, sample_button])


if __name__ == '__main__':
    if not MODEL_FILES_AVAILABLE:
        print("CRITICAL: Model or helper Python files are missing. The application may not function.")
        print("Please check your project structure and Python path.")
    if not RDKIT_AVAILABLE:
        print("WARNING: RDKit is not installed. Molecule visualizations will be disabled.")
    
    # For development, you might want to enable queue for handling multiple users or long processes
    # demo.queue()
    demo.launch()
