"""
Cosmological Data Loader
Contains real observational data for CGC theory testing
"""
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def list_datasets():
    """List available datasets"""
    datasets = {}
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.txt') or file.endswith('.dat'):
                rel_path = os.path.relpath(os.path.join(root, file), DATA_DIR)
                datasets[file] = rel_path
    return datasets

if __name__ == "__main__":
    print("Available datasets:")
    for file, path in list_datasets().items():
        print(f"  {file:30s} -> {path}")
