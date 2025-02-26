import gradio as gr
import torch
import json
import yaml
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import py3Dmol
import random

# Global test mode flag
TEST_MODE = True

# Import necessary functions from search_and_infer.py
from search_and_infer import load_config, load_raw_spectrum_tokens, load_raw_ir, SimpleSpectralSmilesDataset

# Import model and tokenizer classes
from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer

# Load configuration. Adjust config_path if needed.
config_path = "configs/test_config.yaml"  # ensure you have this config file or adjust the path
config = load_config(config_path)

current_dir = Path(__file__).parent

# Load the SMILES tokenizer vocabulary (vocab.txt should be in the current directory)
vocab_path = current_dir / "vocab.txt"
tokenizer = SmilesTokenizer(vocab_file=str(vocab_path))

# Load spectral tokenizer from vocab.json (adjust path if necessary)
nmr_vocab_path = Path(config['data']['tokenized_dir']).parent / 'vocab.json'
if not nmr_vocab_path.exists():
    raise FileNotFoundError(f"NMR vocabulary not found at {nmr_vocab_path}")
with open(nmr_vocab_path) as f:
    spectral_tokenizer = json.load(f)

# Set device and determine vocabulary sizes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
smiles_vocab_size = len(tokenizer)
nmr_token_ids = list(spectral_tokenizer.values())
nmr_vocab_size = max(nmr_token_ids) + 1

# Initialize the model
model = MultiModalToSMILESModel(
    smiles_vocab_size=smiles_vocab_size,
    nmr_vocab_size=nmr_vocab_size,
    max_seq_length=config['model']['max_seq_length'],
    max_nmr_length=config['model']['max_nmr_length'],
    max_memory_length=config['model']['max_memory_length'],
    embed_dim=config['model']['embed_dim'],
    num_heads=config['model']['num_heads'],
    num_layers=config['model']['num_layers'],
    dropout=config['model']['dropout'],
    ir_encoder_type=config['model']['ir_encoder_type'],
    use_stablemax=config['model'].get('use_stablemax', False)
).to(device)

# Load a pretrained model checkpoint; update the path as necessary.
checkpoint_path = "/home/consorzio/Technoscience/Research/multimodal/checkpoints/best_model.pt"  # change this as needed
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test dataset for random sampling in test mode
test_dataset = SimpleSpectralSmilesDataset(
    data_dir=config['data']['tokenized_dir'],
    split='test',
    smiles_tokenizer=tokenizer,
    spectral_tokenizer=spectral_tokenizer,
    max_smiles_len=config['model']['max_seq_length'],
    max_nmr_len=config['model']['max_nmr_length']
)

# Function to create 2D molecule visualization
def visualize_smiles(smiles):
    try:
        # Clean SMILES by removing spaces
        clean_smiles = smiles.replace(" ", "")
        
        # Try to create molecule from cleaned SMILES
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return "<p>Invalid molecule</p>"

        # Compute 2D coordinates for the molecule
        AllChem.Compute2DCoords(mol)
        
        # Generate image using RDKit's Draw
        img = Draw.MolToImage(mol, size=(300, 300))
        
        # Encode the image to base64 string
        from io import BytesIO
        import base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Return HTML containing the image
        html = f'<div style="text-align: center;"><img src="data:image/png;base64,{img_str}" alt="Molecule Visualization" /></div>'
        return html
    except Exception as e:
        print(f"Error generating 2D structure: {e}")
        return "<p>Error generating 2D structure</p>"

# Gradio prediction function
def predict_molecule(c_nmr_file, h_nmr_file, ir_file):
    if TEST_MODE:
        # Randomly select a sample from test dataset
        idx = random.randint(0, len(test_dataset) - 1)
        _, (ir_data, _), nmr_tokens, _ = test_dataset[idx]
        
        # Move tensors to device
        if ir_data is not None:
            ir_data = ir_data.to(device)
        nmr_tokens = nmr_tokens.to(device)
        
        # Add batch dimension if needed
        if len(nmr_tokens.shape) == 1:
            nmr_tokens = nmr_tokens.unsqueeze(0)
        if ir_data is not None and len(ir_data.shape) == 1:
            ir_data = ir_data.unsqueeze(0)
        
        print(f"\nUsing random test sample {idx}")
    else:
        # Ensure all files are uploaded
        if c_nmr_file is None or h_nmr_file is None or ir_file is None:
             return "Please upload all three spectra files", "<p>No visualization available</p>"
        
        # Extract file paths. Gradio file uploads may be objects with a 'name' attribute
        c_nmr_path = c_nmr_file.name if hasattr(c_nmr_file, "name") else c_nmr_file
        h_nmr_path = h_nmr_file.name if hasattr(h_nmr_file, "name") else h_nmr_file
        ir_path = ir_file.name if hasattr(ir_file, "name") else ir_file

        # Load raw spectrum tokens for C-NMR and H-NMR
        max_nmr_len = config['model']['max_nmr_length']
        try:
            c_tokens = load_raw_spectrum_tokens(c_nmr_path, spectral_tokenizer, max_nmr_len)
            h_tokens = load_raw_spectrum_tokens(h_nmr_path, spectral_tokenizer, max_nmr_len)
        except Exception as e:
            return f"Error processing NMR files: {str(e)}", "<p>Error in visualization</p>"

        # Combine the tokens from both NMR files
        nmr_tokens = torch.cat([c_tokens, h_tokens])
        
        # Load IR spectrum
        try:
            ir_data = load_raw_ir(ir_path)
        except Exception as e:
            return f"Error processing IR file: {str(e)}", "<p>Error in visualization</p>"

        nmr_tokens = nmr_tokens.to(device)
        ir_data = ir_data.to(device)

        # Add batch dimension if needed
        if len(nmr_tokens.shape) == 1:
            nmr_tokens = nmr_tokens.unsqueeze(0)
        if len(ir_data.shape) == 1:
            ir_data = ir_data.unsqueeze(0)

    # Run the model's greedy decoding to predict SMILES
    predicted_smiles = greedy_decode(
        model=model,
        nmr_tokens=nmr_tokens,
        ir_data=ir_data,
        tokenizer=tokenizer,
        max_len=config['model']['max_seq_length'],
        device=device
    )[0]  # Take first prediction since we only have one sample

    # Remove any spaces from the predicted SMILES
    predicted_smiles = predicted_smiles.replace(" ", "")

    # Try to create visualization, but always return the predicted SMILES
    try:
        mol = Chem.MolFromSmiles(predicted_smiles)
        if mol is None:
            print(f"Invalid SMILES generated: {predicted_smiles}")
            return predicted_smiles, "<p>Invalid SMILES - cannot visualize molecule</p>"
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Create visualization
        img = Draw.MolToImage(mol, size=(300, 300))
        
        # Convert to base64
        from io import BytesIO
        import base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        html = f'<div style="text-align: center;"><img src="data:image/png;base64,{img_str}" alt="Molecule Visualization" /></div>'
        return predicted_smiles, html
    except Exception as e:
        print(f"Error in visualization: {e}")
        return predicted_smiles, "<p>Error generating visualization</p>"

# Inserting custom CSS for font fallback
custom_css = '''
@font-face {
  font-family: "ui-sans-serif";
  src: local("Arial"), local("sans-serif");
}
@font-face {
  font-family: "system-ui";
  src: local("Arial"), local("sans-serif");
}
'''

# Build Gradio Interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# SMILES Prediction from Spectra")
    gr.Markdown("Upload your C-NMR, H-NMR, and IR spectra (in txt format) to predict the molecule's SMILES and visualize its 2D structure.")
    
    with gr.Row():
        with gr.Column():
            c_file = gr.File(label="Upload C-NMR Spectrum (txt)", file_types=[".txt"])
            h_file = gr.File(label="Upload H-NMR Spectrum (txt)", file_types=[".txt"])
            ir_file = gr.File(label="Upload IR Spectrum (txt)", file_types=[".txt"])
            predict_btn = gr.Button("Predict SMILES")
        with gr.Column():
            smiles_out = gr.Textbox(label="Predicted SMILES")
            # Use HTML component for 2D molecule viewer
            vis_out = gr.HTML(label="2D Molecule Viewer")
    
    predict_btn.click(
        fn=predict_molecule,
        inputs=[c_file, h_file, ir_file],
        outputs=[smiles_out, vis_out],
        api_name="predict"
    )

# Add the local greedy_decode function (identical to the one in train_autoregressive.py)
from train_autoregressive import greedy_decode

if __name__ == "__main__":
    # Configure launch parameters
    demo.launch(
        server_name="0.0.0.0",  # Listen on all network interfaces
        share=False,  # Don't create public URL
        show_api=False,  # Don't show API docs
        allowed_paths=[],  # Restrict file access
        quiet=True  # Reduce console output
    ) 