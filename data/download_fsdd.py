"""
Download the Free Spoken Digit Dataset (FSDD).
MIT License. ~10 MB.

Usage:
    python data/download_fsdd.py

Output:
    data/fsdd/  (raw .wav files organized as {digit}_{speaker}_{rep}.wav)
"""

import os
import urllib.request
import zipfile
import sys

FSDD_URL = (
    "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip"
)
DEST_DIR = os.path.join(os.path.dirname(__file__), "fsdd")
ZIP_PATH = os.path.join(os.path.dirname(__file__), "fsdd-master.zip")


def download(url: str, dest: str) -> None:
    """Download a file from url to dest with a progress indicator."""
    print(f"Downloading FSDD from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")


def extract(zip_path: str, dest_dir: str) -> None:
    """Extract the FSDD zip archive into dest_dir/fsdd/."""
    print(f"Extracting to {dest_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    # The zip extracts to free-spoken-digit-dataset-master/
    extracted = os.path.join(dest_dir, "free-spoken-digit-dataset-master")
    recordings = os.path.join(extracted, "recordings")
    # Move recordings/ contents up to fsdd/
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    for fname in os.listdir(recordings):
        src = os.path.join(recordings, fname)
        dst = os.path.join(dest_dir, fname)
        os.rename(src, dst)
    # Clean up extracted dir and zip
    for root, dirs, files in os.walk(extracted, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(extracted)
    os.remove(zip_path)
    print(f"Extracted to {dest_dir}/ ({len(os.listdir(dest_dir))} files)")


def ensure_fsdd() -> str:
    """Download and extract FSDD if not already present. Returns path to recordings."""
    if not os.path.isdir(DEST_DIR):
        os.makedirs(DEST_DIR, exist_ok=True)
        download(FSDD_URL, ZIP_PATH)
        extract(ZIP_PATH, DEST_DIR)
    else:
        print(f"FSDD already exists at {DEST_DIR}")
    return DEST_DIR


if __name__ == "__main__":
    ensure_fsdd()
