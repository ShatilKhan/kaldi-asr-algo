"""
YesNo dataset (Kaldi egs/yesno, OpenSLR resource 1).

60 recordings of a speaker saying sequences of "ken" (yes) and "lo" (no)
in Hebrew. 8kHz mono WAV, ~6 seconds each, 8 words per recording.
The filename encodes the ground truth: 1_0_1_1_0_1_0_1.wav -> 1=yes, 0=no.

This is the pipeline-validation dataset: with ~3 phones and long utterances
there are ~100+ frames per HMM state, so monophone GMM training converges
easily. If the pipeline can't learn yes/no, the pipeline is wrong.
"""

import os
import tarfile
import urllib.request
from typing import Dict, List, Tuple

from data.reader import read_wav

YESNO_URL = "https://www.openslr.org/resources/1/waves_yesno.tar.gz"


def ensure_yesno(data_dir: str = "data/yesno") -> str:
    """Download and extract the YesNo dataset if needed. Returns wave dir."""
    waves_dir = os.path.join(data_dir, "waves_yesno")
    if os.path.isdir(waves_dir) and len(os.listdir(waves_dir)) >= 60:
        return waves_dir

    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "waves_yesno.tar.gz")
    if not os.path.exists(tar_path):
        print(f"  Downloading {YESNO_URL} ...")
        urllib.request.urlretrieve(YESNO_URL, tar_path)

    print("  Extracting ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)

    return waves_dir


def transcript_from_filename(fname: str) -> str:
    """1_0_1_1_0_1_0_1.wav -> 'yes no yes yes no yes no yes'"""
    stem = os.path.splitext(os.path.basename(fname))[0]
    return " ".join("yes" if c == "1" else "no" for c in stem.split("_"))


def load_yesno(waves_dir: str) -> List[Dict]:
    """Read all recordings. Each record: samples, sample_rate, text, fname."""
    records = []
    for fname in sorted(os.listdir(waves_dir)):
        if not fname.endswith(".wav"):
            continue
        samples, sr = read_wav(os.path.join(waves_dir, fname))
        records.append({
            "samples": samples,
            "sample_rate": sr,
            "text": transcript_from_filename(fname),
            "fname": fname,
        })
    return records


def prepare_yesno(data_dir: str = "data/yesno") -> Tuple[List[Dict], List[Dict]]:
    """
    Load YesNo and split train/test (30/30, interleaved).

    The filenames sort into 0_* (utterances starting with "no") followed by
    1_* (starting with "yes"), so a head/tail split would teach the language
    model that utterances start with "no" and test it on ones that start
    with "yes". Interleaving keeps both halves balanced and deterministic.
    """
    waves_dir = ensure_yesno(data_dir)
    records = load_yesno(waves_dir)
    return records[0::2], records[1::2]


if __name__ == "__main__":
    train, test = prepare_yesno()
    print(f"Train: {len(train)}  Test: {len(test)}")
    r = train[0]
    dur = len(r["samples"]) / r["sample_rate"]
    print(f"First: {r['fname']} ({dur:.1f}s @ {r['sample_rate']}Hz)")
    print(f"  text: {r['text']}")
