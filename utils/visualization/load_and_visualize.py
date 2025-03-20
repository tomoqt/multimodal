import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_data():
    """Load parquet files and metadata"""
    # Load parquet files from directory
    data_path = Path("data_extraction/multimodal_spectroscopic_dataset")
    
    # Get all parquet files
    parquet_files = list(data_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No parquet files found in the specified directory")
    
    # Count total number of spectra across all files
    total_count = 0
    for file in parquet_files:
        # Read just the count without loading the full data
        count = pd.read_parquet(file, columns=['smiles']).shape[0]
        total_count += count
        print(f"File {file.name}: {count} spectra")
    
    print(f"\nTotal number of spectra in dataset: {total_count}")
    
    # Calculate statistics using sampling
    sample_size = 10000  # Adjust this number based on your memory constraints
    missing_stats = {'IR': 0, '1H NMR': 0, '13C NMR': 0}
    sampled_count = 0
    
    for file in parquet_files:
        # Get number of row groups in this file
        pf = pd.read_parquet(file, columns=['smiles'])
        n_rows = len(pf)
        
        # Calculate how many rows to sample from this file
        file_sample_size = int((n_rows / total_count) * sample_size)
        if file_sample_size == 0:
            continue
            
        # Randomly sample row indices
        sample_indices = np.random.choice(n_rows, file_sample_size, replace=False)
        
        # Read sampled rows
        df_chunk = pd.read_parquet(
            file,
            columns=['smiles', 'molecular_formula', 'ir_spectra', 'h_nmr_spectra', 'c_nmr_spectra'],
            engine='fastparquet'
        ).iloc[sample_indices]
        
        # Update statistics
        missing_stats['IR'] += (df_chunk['ir_spectra'].isna() | df_chunk['ir_spectra'].apply(lambda x: len(x) == 0)).sum()
        missing_stats['1H NMR'] += (df_chunk['h_nmr_spectra'].isna() | df_chunk['h_nmr_spectra'].apply(lambda x: len(x) == 0)).sum()
        missing_stats['13C NMR'] += (df_chunk['c_nmr_spectra'].isna() | df_chunk['c_nmr_spectra'].apply(lambda x: len(x) == 0)).sum()
        sampled_count += len(df_chunk)
    
    print("\nMissing Spectra Statistics (Based on Random Sampling):")
    print("-" * 40)
    print(f"Sample size: {sampled_count:,} molecules ({(sampled_count/total_count)*100:.1f}% of total)")
    for spectrum_type, missing_count in missing_stats.items():
        percentage = (missing_count / sampled_count) * 100
        print(f"{spectrum_type}: {missing_count:,} missing ({percentage:.2f}%)")
    print(f"\nTotal molecules: {total_count:,}")
    
    # Load just one file for visualization
    df = pd.read_parquet(
        parquet_files[0],
        columns=['smiles', 'molecular_formula', 'ir_spectra', 'h_nmr_spectra', 'c_nmr_spectra', 'hsqc_nmr_spectrum'],
        engine='fastparquet'
    )
    
    # Load metadata
    meta_path = data_path / "meta_data/meta_data_dict.json"
    if not meta_path.exists():
        raise FileNotFoundError("Metadata file not found")
        
    with open(meta_path) as f:
        meta_data = json.load(f)
        
    return df, meta_data

def plot_ir_spectrum(spectrum, wavenumbers, title="IR Spectrum"):
    """Plot IR spectrum with inverted x-axis and transmission"""
    plt.figure(figsize=(10, 6))
    
    # Convert absorbance to transmission
    transmission = 1 - np.array(spectrum)
    
    plt.plot(wavenumbers, transmission)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Transmission")
    plt.title(title)
    
    # Invert x-axis
    plt.gca().invert_xaxis()
    
    # Remove y-axis ticks since we care about relative peaks
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_nmr_spectrum(spectrum, ppm_scale, title="NMR Spectrum"):
    """Plot NMR spectrum with inverted x-axis"""
    plt.figure(figsize=(10, 6))
    plt.plot(ppm_scale, spectrum)
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Intensity")
    plt.title(title)
    
    # Invert x-axis
    plt.gca().invert_xaxis()
    
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_hsqc_spectrum(spectrum, h_shifts, c_shifts, title="HSQC Spectrum"):
    """Plot 2D HSQC spectrum"""
    plt.figure(figsize=(10, 8))
    
    try:
        # Convert spectrum to numpy array if it isn't already
        if not isinstance(spectrum, np.ndarray):
            spectrum = np.array(spectrum)
        
        # Handle different possible input formats
        if spectrum.ndim == 1:
            # If flattened, reshape to 512x512
            spectrum = spectrum.reshape((512, 512))
        elif spectrum.ndim > 2:
            # If has extra dimensions, squeeze them out
            spectrum = np.squeeze(spectrum)
            
        if spectrum.shape != (512, 512):
            raise ValueError(f"Unexpected spectrum shape: {spectrum.shape}, expected (512, 512)")
            
        # Create contour plot with fewer levels and custom normalization
        plt.contour(h_shifts, c_shifts, spectrum, 
                   levels=np.linspace(spectrum.min(), spectrum.max(), 20),
                   cmap='viridis')
        
        # Invert both axes
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        
        plt.xlabel("$^1$H Chemical Shift (ppm)")
        plt.ylabel("$^{13}$C Chemical Shift (ppm)")
        plt.title(title)
        
        plt.colorbar(label='Intensity')
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        plt.close()  # Clean up the figure if there's an error
        raise ValueError(f"Error plotting HSQC spectrum: {str(e)}")
    
    plt.show()

def main():
    try:
        # Load data
        print("Loading data...")
        df, meta_data = load_data()
        
        # Get random sample (more efficiently)
        sample_idx = np.random.randint(0, len(df))
        sample = df.iloc[sample_idx]
        
        # Print HSQC information
        hsqc_spectrum = sample["hsqc_nmr_spectrum"]
        if hsqc_spectrum is not None:
            hsqc_array = np.array(hsqc_spectrum)
            print("\nHSQC Spectrum Information:")
            print(f"Shape: {hsqc_array.shape}")
            print("\nFirst few values of HSQC spectrum:")
            print(hsqc_array.flatten()[:10])  # Print first 10 values
            
            # Print min and max values
            print(f"\nMin value: {hsqc_array.min()}")
            print(f"Max value: {hsqc_array.max()}")
        else:
            print("\nHSQC spectrum is None")
            
        # Free up memory
        del df
        
        # Print molecule info
        print("\nMolecule Information:")
        print(f"SMILES: {sample['smiles']}")
        print(f"Molecular Formula: {sample['molecular_formula']}")
        
        # Plot spectra one at a time, clearing memory after each plot
        
        # Plot IR spectrum
        print("\nPlotting IR spectrum...")
        ir_spectrum = np.array(sample["ir_spectra"])
        ir_wavenumbers = meta_data["ir_spectra"]["dimensions"]
        plot_ir_spectrum(ir_spectrum, ir_wavenumbers)
        del ir_spectrum
        
        # Plot 1H NMR spectrum
        print("\nPlotting 1H NMR spectrum...")
        h_nmr_spectrum = np.array(sample["h_nmr_spectra"])
        h_nmr_ppm = meta_data["h_nmr_spectra"]["dimensions"]
        plot_nmr_spectrum(h_nmr_spectrum, h_nmr_ppm, title="1H NMR Spectrum")
        del h_nmr_spectrum
        
        # Plot 13C NMR spectrum
        print("\nPlotting 13C NMR spectrum...")
        c_nmr_spectrum = np.array(sample["c_nmr_spectra"])
        c_nmr_ppm = meta_data["c_nmr_spectra"]["dimensions"]
        plot_nmr_spectrum(c_nmr_spectrum, c_nmr_ppm, title="13C NMR Spectrum")
        del c_nmr_spectrum
        
        # Plot HSQC spectrum
        print("\nPlotting HSQC spectrum...")
        hsqc_spectrum = sample["hsqc_nmr_spectrum"]
        if hsqc_spectrum is None or len(hsqc_spectrum) == 0:
            print("Warning: HSQC spectrum data is empty or None, skipping HSQC plot")
        else:
            h_spec_dim = meta_data['hsqc_nmr_spectrum']['dimensions']['h']
            c_spec_dim = meta_data['hsqc_nmr_spectrum']['dimensions']['c']
            plot_hsqc_spectrum(hsqc_spectrum, h_spec_dim, c_spec_dim)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nPlease check that:")
        print("1. The data directory exists at 'data_extraction/multimodal_spectroscopic_dataset'")
        print("2. The directory contains parquet files")
        print("3. The metadata file exists at 'data_extraction/multimodal_spectroscopic_dataset/meta_data/meta_data_dict.json'")

if __name__ == "__main__":
    main()
