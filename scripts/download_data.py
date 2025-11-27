"""Download Stanford Dogs dataset."""

import argparse
import sys
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("Error: kagglehub not installed. Install with: pip install kagglehub")
    sys.exit(1)


def download_dataset(output_dir: str = "data") -> str:
    """
    Download Stanford Dogs dataset.

    Args:
        output_dir: Directory to save dataset

    Returns:
        Path to downloaded dataset
    """
    print("Downloading Stanford Dogs dataset...")
    print("This may take a while...")

    try:
        # Download latest version
        path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")
        print(f"Dataset downloaded to: {path}")

        # Optionally move to output_dir
        if output_dir != ".":
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Dataset available at: {path}")
            print(f"To use a custom directory, set output_dir parameter")

        return str(path)

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download Stanford Dogs dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save dataset (default: data)",
    )

    args = parser.parse_args()

    dataset_path = download_dataset(args.output_dir)
    print(f"\nDataset ready at: {dataset_path}")
    print("You can now use this path in your training scripts.")


if __name__ == "__main__":
    main()

