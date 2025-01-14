import click
from pathlib import Path
import tarfile
from huggingface_hub import hf_hub_download
import shutil

@click.command()
@click.option(
    "--output_path",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("tokenized_baseline"),
    help="Base output path (dataset will be placed in output_path/data/)"
)
@click.option(
    "--repo_id",
    default="Tomoqt/tokenized_NMR_resample1000",
    help="Hugging Face repository ID"
)
@click.option(
    "--filename",
    default="tokenized_dataset.tar.gz",
    help="Name of the archive file in the repository"
)
def main(output_path: Path, repo_id: str, filename: str):
    """
    Download tokenized dataset from Hugging Face and decompress it.
    The dataset will be extracted to output_path/data/ to match
    the organization used in create_tokenized_dataset_faster.py
    """
    print(f"Downloading dataset from {repo_id} (dataset repository)...")
    
    # Create temporary directory for download
    temp_dir = Path("temp_download")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download the file
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=temp_dir,
            repo_type="dataset"
        )
        
        print("Download complete. Extracting...")
        
        # Create temporary extraction directory
        temp_extract = temp_dir / "extract"
        temp_extract.mkdir(exist_ok=True)
        
        # Extract the archive to temporary location first
        with tarfile.open(downloaded_file, "r:gz") as tar:
            tar.extractall(path=temp_extract)
        
        # Ensure final output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Move the data directory to its final location
        # If a data directory already exists, remove it first
        final_data_path = output_path / "data"
        if final_data_path.exists():
            shutil.rmtree(final_data_path)
        
        # Move extracted data to final location
        shutil.move(str(temp_extract / "data"), str(final_data_path))
        
        print(f"Dataset extracted to: {final_data_path}")
        print("\nDirectory structure:")
        for file in sorted(final_data_path.glob("*")):
            print(f"  {file.relative_to(output_path)}")
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("\nCleaned up temporary files")

if __name__ == "__main__":
    main() 