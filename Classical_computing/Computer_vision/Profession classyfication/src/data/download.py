"""
Data downloading module for the profession classifier project.
"""

import os
import argparse
import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """
    Download a file from a URL with a progress bar.
    
    Args:
        url (str): URL to download
        output_path (str): Path to save the downloaded file
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, 
                                  reporthook=t.update_to)


def download_and_extract_idenprof(data_dir='data'):
    """
    Download and extract the idenprof dataset.
    
    Args:
        data_dir (str): Directory to save the dataset
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for the idenprof dataset
    url = "https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip"
    zip_path = os.path.join(data_dir, "idenprof-jpg.zip")
    
    # Download the dataset
    if not os.path.exists(zip_path):
        print(f"Downloading idenprof dataset from {url}...")
        download_url(url, zip_path)
    else:
        print(f"Dataset already downloaded at {zip_path}")
    
    # Extract the dataset
    dataset_dir = os.path.join(data_dir, "idenprof")
    if not os.path.exists(dataset_dir):
        print(f"Extracting dataset to {dataset_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete!")
    else:
        print(f"Dataset already extracted at {dataset_dir}")
    
    # Print dataset info
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    professions = os.listdir(train_dir)
    print("\nDataset information:")
    print(f"Number of classes: {len(professions)}")
    print(f"Classes: {', '.join(sorted(professions))}")
    
    train_samples = sum([len(os.listdir(os.path.join(train_dir, p))) for p in professions])
    test_samples = sum([len(os.listdir(os.path.join(test_dir, p))) for p in professions])
    
    print(f"Training samples: {train_samples}")
    print(f"Testing samples: {test_samples}")
    print(f"Total samples: {train_samples + test_samples}")
    
    return dataset_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the idenprof dataset")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory to save the dataset")
    args = parser.parse_args()
    
    download_and_extract_idenprof(args.data_dir)