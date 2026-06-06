"""
Mini LibriSpeech (OpenSLR SLR31): continuous read English speech.

train-clean-5 (~5h) and dev-clean-2 (~2h) are Kaldi's own regression corpus.
Audio is 16kHz FLAC under SPEAKER/CHAPTER/, with transcripts in
SPEAKER-CHAPTER.trans.txt (one line per utterance: "<uttid> THE WORDS").

This is the proper continuous-ASR milestone: real audiobook speech, a real
pronunciation lexicon, and multi-word utterances, so the decoder upgrades
(beam pruning, backoff G) are exercised for real.
"""

import os
import tarfile
import urllib.request
from typing import Dict, List, Tuple

import soundfile as sf

URLS = {
    "train-clean-5": "https://www.openslr.org/resources/31/train-clean-5.tar.gz",
    "dev-clean-2": "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
}


def ensure_libri(subset: str, data_dir: str = "data/librispeech") -> str:
    """Download + extract a LibriSpeech subset if needed. Returns its dir."""
    out_dir = os.path.join(data_dir, "LibriSpeech", subset)
    if os.path.isdir(out_dir):
        return out_dir

    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, f"{subset}.tar.gz")
    if not os.path.exists(tar_path):
        print(f"  Downloading {URLS[subset]} ...")
        urllib.request.urlretrieve(URLS[subset], tar_path)

    print(f"  Extracting {subset} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)

    return out_dir


def load_libri(subset: str, data_dir: str = "data/librispeech") -> List[Dict]:
    """
    Read all utterances of a subset.

    Returns records with keys: uttid, speaker, text (lowercased words),
    flac (path); audio is loaded lazily via load_audio() to keep memory low.
    """
    root = ensure_libri(subset, data_dir)
    records = []
    for spk in sorted(os.listdir(root)):
        spk_dir = os.path.join(root, spk)
        if not os.path.isdir(spk_dir):
            continue
        for chap in sorted(os.listdir(spk_dir)):
            chap_dir = os.path.join(spk_dir, chap)
            trans = os.path.join(chap_dir, f"{spk}-{chap}.trans.txt")
            if not os.path.exists(trans):
                continue
            with open(trans) as f:
                for line in f:
                    uttid, _, text = line.strip().partition(" ")
                    records.append({
                        "uttid": uttid,
                        "speaker": spk,
                        "text": text.lower(),
                        "flac": os.path.join(chap_dir, f"{uttid}.flac"),
                    })
    return records


def load_audio(record: Dict) -> Tuple["np.ndarray", int]:
    """Decode one record's FLAC to (samples float64, sample_rate)."""
    samples, sr = sf.read(record["flac"], dtype="float64")
    return samples, sr


if __name__ == "__main__":
    recs = load_libri("dev-clean-2")
    print(f"dev-clean-2: {len(recs)} utterances, {len(set(r['speaker'] for r in recs))} speakers")
    r = recs[0]
    samples, sr = load_audio(r)
    print(f"  {r['uttid']} ({len(samples)/sr:.1f}s @ {sr}Hz): {r['text'][:60]}...")
