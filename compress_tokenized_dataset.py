import click
from pathlib import Path
import tarfile
import shutil

@click.command()
@click.option(
    "--data_path",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    default=Path("tokenized_baseline"),
    help="Path to the tokenized data directory containing the 'data' folder"
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("compressed_datasets"),
    help="Output path for the compressed archive"
)
def main(data_path: Path, output_path: Path):
    """
    Compress tokenized dataset into a tar.gz archive.
    """
    print(f"Compressing dataset from {data_path}...")
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create archive name
    archive_path = output_path / "tokenized_dataset.tar.gz"
    
    # Create tar.gz archive
    with tarfile.open(archive_path, "w:gz") as tar:
        # Add the data directory to the archive
        tar.add(data_path / "data", arcname="data")
    
    print(f"Dataset compressed and saved to: {archive_path}")
    print(f"Archive size: {archive_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main() 