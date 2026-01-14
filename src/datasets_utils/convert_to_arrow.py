import os


os.environ['HF_HOME'] = '/mnt/data/anhvth/tmp/'
import sys
from pathlib import Path

from datasets import load_dataset


def convert_to_arrow(path_to_dir):
    # 1. Setup paths
    input_path = Path(path_to_dir).resolve()
    # Search for all .parquet files in the directory
    parquet_files = [str(f) for f in input_path.glob("*.parquet")]

    if not parquet_files:
        print(f"No parquet files found in {input_path}")
        return

    output_dir = input_path / "hf_dataset_arrow"

    print(f"Found {len(parquet_files)} files. Converting...")

    # 2. Load and Save
    # 'keep_in_memory=False' ensures we don't crash RAM if the files are huge
    n_proc = os.cpu_count() if os.cpu_count() is not None else 1
    dataset = load_dataset("parquet", data_files=parquet_files, split="train", keep_in_memory=True, num_proc=n_proc)

    # This creates the native Arrow format for fast loading
    dataset.save_to_disk(output_dir)

    print(f"✅ Success! Saved to: {output_dir}")
    print("Now load it instantly with: load_from_disk('path')")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_arrow.py <path_to_dir>")
    else:
        convert_to_arrow(sys.argv[1])
