from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path
import re
from tqdm import tqdm
import os
import shutil
import json

def download_dataset_files(
    repo_id: str = "Tomoqt/parquet_spectra",
    local_dir: str = "data_extraction/multimodal_spectroscopic_dataset",
    download_binaries: bool = False,
    binary_repo_id: str = "Tomoqt/binarized_spectral_data",
    binary_dir: str = "training_binaries",
    revision: str = "main",
    token: str = None,
):
    """
    Download dataset files from Huggingface Hub.
    
    Parameters
    ----------
    repo_id : str
        The Huggingface repository ID for parquet files (default: 'Tomoqt/parquet_spectra')
    local_dir : str
        Local directory where parquet files should be downloaded
    download_binaries : bool
        Whether to also download binary files (default: False)
    binary_repo_id : str
        The Huggingface repository ID for binary files
    binary_dir : str
        Local directory where binary files should be downloaded
    revision : str
        The git revision to download from (default: "main")
    token : str, optional
        Huggingface API token for private repositories
    """
    # Create the local directories
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_dir = local_dir / "meta_data"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize the Hugging Face API client
        api = HfApi()
        
        # Download parquet files and metadata
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision, token=token)
        
        # Separate metadata files and parquet files
        metadata_files = [f for f in files if f.endswith('.json')]
        parquet_files = [f for f in files if f.endswith('.parquet')]
        
        print(f"\nFound {len(parquet_files)} parquet files and {len(metadata_files)} metadata files")
        
        # Download and organize metadata files
        for filename in tqdm(metadata_files, desc="Downloading metadata files"):
            try:
                temp_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                    local_dir=local_dir
                )
                # Move metadata files to meta_data directory
                final_path = metadata_dir / os.path.basename(filename)
                shutil.move(temp_path, final_path)
                print(f"\nDownloaded metadata: {filename} -> {final_path}")
            except Exception as e:
                print(f"\nError downloading {filename}: {str(e)}")
        
        # Download parquet files
        for filename in tqdm(parquet_files, desc="Downloading parquet files"):
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                    local_dir=local_dir
                )
                print(f"\nDownloaded: {filename} -> {local_path}")
            except Exception as e:
                print(f"\nError downloading {filename}: {str(e)}")
        
        # Download binary files if requested
        if download_binaries:
            binary_dir = Path(binary_dir)
            binary_dir.mkdir(parents=True, exist_ok=True)
            
            binary_pattern = re.compile(r'^aligned_chunk_\d+_rg\d+\.pt$')
            binary_files = api.list_repo_files(
                repo_id=binary_repo_id, 
                repo_type="dataset", 
                revision=revision, 
                token=token
            )
            binary_files = [f for f in binary_files if binary_pattern.match(os.path.basename(f))]
            
            print(f"\nFound {len(binary_files)} matching binary files")
            for filename in tqdm(binary_files, desc="Downloading binary files"):
                try:
                    local_path = hf_hub_download(
                        repo_id=binary_repo_id,
                        filename=filename,
                        repo_type="dataset",
                        revision=revision,
                        token=token,
                        local_dir=binary_dir
                    )
                    print(f"\nDownloaded: {filename} -> {local_path}")
                except Exception as e:
                    print(f"\nError downloading {filename}: {str(e)}")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    download_dataset_files(
        # For private repositories, you'll need to set your token
        # token="your_token_here"  # or use os.getenv("HF_TOKEN")
        download_binaries=True  # Set to True if you want to download binary files as well
    )
