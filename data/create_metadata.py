import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_metadata():
    """Create and save metadata files"""
    
    # Create dimension arrays
    # 10ppm to -2ppm in 100000 steps
    hnmr_ppm = np.linspace(10, -2, 10000).tolist()
    
    # 230ppm to -20ppm in 100000 steps
    cnmr_ppm = np.linspace(230, -20, 10000).tolist()
    
    # 400cm^{-1} to -4000cm^{-1} in 1800 steps
    ir_cm = np.linspace(400, 4000, 1800).tolist()
    
    # 10ppm to -2ppm in 512 steps
    two_d_h_nmr = np.linspace(10, -2, 512).tolist()
    # 230ppm to -20ppm in 512 steps
    two_d_c_nmr = np.linspace(230, -20, 512).tolist()
    
    # Define spectrum dimensions
    spectrum_dimensions = {
        "h_nmr_spectra": {"range": [-2, 10], "points": 10000, "unit": "ppm", "dimensions":hnmr_ppm},
        "c_nmr_spectra": {"range": [-20, 230], "points": 10000, "unit": "ppm","dimensions":cnmr_ppm},
        "hsqc_nmr_spectrum_h": {"range": [-2, 10], "points": 512, "unit": "ppm","dimensions":two_d_h_nmr},
        "hsqc_nmr_spectrum_c": {"range": [-20, 230], "points": 512, "unit": "ppm","dimensions":two_d_c_nmr},
        "ir_spectra": {"range": [400, 4000], "points": 1800, "unit": "cm^{-1}","dimensions":ir_cm},
    }

    meta_data_dict = {
        "smiles": {
            "format": "string",
            "unit": "SMILES",
            "info": "Canonical SMILES string generated using RDKit",
            "example": "CC(C)CCNc1ncc(F)cc1C(=O)O"
        },
        "molecular_formula": {
            "format": "string",
            "unit": "Molecular formula",
            "info": "Molecular formula determined by RDKit",
            "example": "C11H15FN2O2"
        },
        "h_nmr_spectra": {
            "format": "np.array(float)",
            "dimensions": hnmr_ppm,
            "info": "1D proton NMR spectrum intensity values",
            "unit": "ppm",
            "shape": [10000]
        },
        "c_nmr_spectra": {
            "format": "np.array(float)",
            "dimensions": cnmr_ppm,
            "info": "1D carbon-13 NMR spectrum intensity values",
            "unit": "ppm",
            "shape": [10000]
        },
        "hsqc_nmr_spectrum": {
            "format": "np.array(float)",
            "dimensions": {
                "h": two_d_h_nmr,
                "c": two_d_c_nmr
            },
            "shape": [512, 512],
            "unit": "ppm",
            "info": "2D HSQC NMR spectrum intensity matrix"
        },
        "ir_spectra": {
            "format": "np.array(float)",
            "dimensions": ir_cm,
            "info": "IR absorption spectrum intensity values",
            "unit": "cm^{-1}",
            "shape": [1800]
        }
    }

    # Create metadata directory if it doesn't exist
    metadata_dir = Path("data_extraction/multimodal_spectroscopic_dataset/meta_data")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata files
    with open(metadata_dir / "meta_data_dict.json", "w") as f:
        json.dump(meta_data_dict, f)
    
    with open(metadata_dir / "spectrum_dimensions.json", "w") as f:
        json.dump(spectrum_dimensions, f)
        
    print("Metadata files created successfully!")

if __name__ == "__main__":
    create_metadata() 