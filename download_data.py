from huggingface_hub import hf_hub_download
from pathlib import Path
import re
from tqdm import tqdm
import os

def download_dataset_files(
    repo_id: str,
    local_dir: str,
    revision: str = "main",
    token: str = None,
):
    """
    Download binary dataset files from Huggingface Hub.
    
    Parameters
    ----------
    repo_id : str
        The Huggingface repository ID (e.g., 'username/repo-name')
    local_dir : str
        Local directory where files should be downloaded
    revision : str
        The git revision to download from (default: "main")
    token : str, optional
        Huggingface API token for private repositories
    """
    # Create the local directory if it doesn't exist
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the file patterns we're looking for
    # Looking for files like: part-0000_rg0.pt
    binary_pattern = re.compile(r'^aligned_chunk_\d+_rg\d+\.pt$')
    
    try:
        # List all files in the repository
        from huggingface_hub import list_files_info
        files = list_files_info(repo_id=repo_id, repo_type="dataset", revision=revision, token=token)
        
        # Filter and download binary files
        binary_files = [f for f in files if binary_pattern.match(os.path.basename(f.rfilename))]
        
        print(f"\nFound {len(binary_files)} matching binary files")
        for file_info in tqdm(binary_files, desc="Downloading binary files"):
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_info.rfilename,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                    local_dir=local_dir
                )
                print(f"\nDownloaded: {file_info.rfilename} -> {local_path}")
            except Exception as e:
                print(f"\nError downloading {file_info.rfilename}: {str(e)}")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    repo_id = "Tomoqt/binarized_spectral_data"  # Replace with your actual repo
    local_dir = "training_binaries"  # Changed to match the output directory from binaries_faster.py
    
    # For private repositories, you'll need to set your token
    # token = "your_token_here"  # or use os.getenv("HF_TOKEN")
    
    download_dataset_files(
        repo_id=repo_id,
        local_dir=local_dir,
        # token=token  # Uncomment if needed
    ) 