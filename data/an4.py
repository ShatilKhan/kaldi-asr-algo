"""
AN4 (CMU Census database): connected small-vocabulary read speech.

Speakers spell out names, addresses and read digit/letter/control-word
strings ("U M N Y H SIX", "ENTER", "GO"). 948 train / 130 test utterances,
16 kHz, speaker-disjoint, with a 99-word vocabulary and its own pronunciation
dictionary. This is the classic task where a monophone GMM-HMM genuinely
works (continuous, real human voice, small enough vocab to be acoustically
separable), unlike open-vocabulary LibriSpeech.

Audio is NIST SPHERE, read directly via soundfile (libsndfile).
"""

import os
import re
import tarfile
import urllib.request
from typing import Dict, List, Tuple

import soundfile as sf

AN4_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz"


def ensure_an4(data_dir: str = "data/an4") -> str:
    """Download + extract AN4 if needed. Returns the an4 root dir."""
    root = os.path.join(data_dir, "an4")
    if os.path.isdir(os.path.join(root, "etc")):
        return root

    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "an4_sphere.tar.gz")
    if not os.path.exists(tar_path):
        print(f"  Downloading {AN4_URL} ...")
        urllib.request.urlretrieve(AN4_URL, tar_path)
    print("  Extracting AN4 ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    return root


def _clean_transcript(line: str) -> Tuple[str, str]:
    """'<s> U M N Y H SIX </s> (an255-fash-b)' -> (uttid, 'u m n y h six')."""
    m = re.search(r"\(([^)]+)\)\s*$", line)
    uttid = m.group(1) if m else None
    text = re.sub(r"\([^)]*\)\s*$", "", line)
    text = text.replace("<s>", "").replace("</s>", "")
    return uttid, " ".join(text.split()).lower()


def load_an4(split: str, data_dir: str = "data/an4") -> List[Dict]:
    """Load AN4 'train' or 'test' split as records (samples loaded lazily)."""
    root = ensure_an4(data_dir)
    fileids = {}
    with open(os.path.join(root, "etc", f"an4_{split}.fileids")) as f:
        for path in f:
            path = path.strip()
            if path:
                fileids[os.path.basename(path)] = path

    records = []
    with open(os.path.join(root, "etc", f"an4_{split}.transcription")) as f:
        for line in f:
            uttid, text = _clean_transcript(line)
            if not uttid or uttid not in fileids or not text:
                continue
            speaker = fileids[uttid].split("/")[1]
            records.append({
                "uttid": uttid,
                "speaker": speaker,
                "text": text,
                "sph": os.path.join(root, "wav", fileids[uttid] + ".sph"),
            })
    return records


def load_audio(record: Dict) -> Tuple["np.ndarray", int]:
    samples, sr = sf.read(record["sph"], dtype="float64")
    return samples, sr


def load_an4_dict(data_dir: str = "data/an4") -> Dict[str, List[str]]:
    """Parse etc/an4.dic into word -> phone list (first pronunciation only)."""
    root = ensure_an4(data_dir)
    prons: Dict[str, List[str]] = {}
    with open(os.path.join(root, "etc", "an4.dic")) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            word = re.sub(r"\(\d+\)$", "", parts[0]).lower()  # drop (2) alternates
            if word in prons:
                continue
            prons[word] = parts[1:]
    return prons


if __name__ == "__main__":
    tr = load_an4("train")
    te = load_an4("test")
    print(f"train: {len(tr)} utts, {len(set(r['speaker'] for r in tr))} speakers")
    print(f"test:  {len(te)} utts, {len(set(r['speaker'] for r in te))} speakers")
    overlap = set(r["speaker"] for r in tr) & set(r["speaker"] for r in te)
    print(f"speaker overlap: {len(overlap)}")
    d = load_an4_dict()
    print(f"dict: {len(d)} words; e.g. six={d.get('six')} go={d.get('go')}")
    s, sr = load_audio(tr[0])
    print(f"audio: {tr[0]['uttid']} ({len(s)/sr:.1f}s @ {sr}Hz): {tr[0]['text']}")
