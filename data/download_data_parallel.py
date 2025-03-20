# gen'd in iteration w DS3
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path
import re
from tqdm import tqdm
import os
import shutil
import concurrent.futures


def download_file(repo_id, filename, repo_type, revision, token, local_dir, is_metadata=False):
    """
    Download a single file using hf_hub_download if it doesn't already exist locally.
    """
    try:
        # Determine the final path for the file
        if is_metadata:
            final_dir = local_dir / "meta_data"
        else:
            final_dir = local_dir
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / os.path.basename(filename)

        # Skip download if the file already exists
        if final_path.exists():
            print(f"\nFile already exists: {final_path}")
            return final_path

        # Download the file
        temp_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=local_dir
        )

        # If it's a metadata file, move it to the meta_data directory
        if is_metadata:
            shutil.move(temp_path, final_path)
            return final_path
        else:
            return temp_path
    except Exception as e:
        print(f"\nError downloading {filename}: {str(e)}")
        return None


def download_dataset_files(
        repo_id: str = "Tomoqt/parquet_spectra",
        local_dir: str = "data_extraction/multimodal_spectroscopic_dataset",
        download_binaries: bool = False,
        binary_repo_id: str = "Tomoqt/binarized_spectral_data",
        binary_dir: str = "training_binaries",
        revision: str = "main",
        token: str = None,
        max_workers: int | None = None,  # Number of concurrent downloads
):
    """
    Download dataset files from Huggingface Hub using parallel downloads, skipping files that already exist.

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
    max_workers : int
        Maximum number of concurrent downloads (default: 16)
    """
    # Create the local directories
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = local_dir / "meta_data"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize the Hugging Face API client
        api = HfApi()

        # List files in the repository
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision, token=token)

        # Separate metadata files and parquet files
        metadata_files = [f for f in files if f.endswith('.json')]
        parquet_files = [f for f in files if f.endswith('.parquet')]

        print(f"\nFound {len(parquet_files)} parquet files and {len(metadata_files)} metadata files")

        # Download metadata files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            metadata_futures = [
                executor.submit(
                    download_file,
                    repo_id, filename, "dataset", revision, token, local_dir, is_metadata=True
                )
                for filename in metadata_files
            ]
            # Track progress with tqdm
            for future in tqdm(concurrent.futures.as_completed(metadata_futures), total=len(metadata_files),
                               desc="Downloading metadata files"):
                result = future.result()
                if result:
                    print(f"\nDownloaded metadata: {result}")

        # Download parquet files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            parquet_futures = [
                executor.submit(
                    download_file,
                    repo_id, filename, "dataset", revision, token, local_dir
                )
                for filename in parquet_files
            ]
            # Track progress with tqdm
            for future in tqdm(concurrent.futures.as_completed(parquet_futures), total=len(parquet_files),
                               desc="Downloading parquet files"):
                result = future.result()
                if result:
                    print(f"\nDownloaded: {result}")

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

            # Download binary files in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                binary_futures = [
                    executor.submit(
                        download_file,
                        binary_repo_id, filename, "dataset", revision, token, binary_dir
                    )
                    for filename in binary_files
                ]
                # Track progress with tqdm
                for future in tqdm(concurrent.futures.as_completed(binary_futures), total=len(binary_files),
                                   desc="Downloading binary files"):
                    result = future.result()
                    if result:
                        print(f"\nDownloaded: {result}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Example usage
    download_dataset_files(
        # For private repositories, you'll need to set your token
        # token="your_token_here",  # Replace with your Hugging Face token or use os.getenv("HF_TOKEN")
        download_binaries=True,  # Set to True if you want to download binary files as well
        max_workers=None,  # change to int to limit the num of CPU threads
    )