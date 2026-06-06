"""
Google Speech Commands v0.02 (Warden 2018).

105,829 one-second 16kHz WAV clips of 35 spoken words, crowdsourced from
2,618 speakers on consumer microphones. This is the real-life test for the
pipeline: thousands of voices, accents, and noisy recordings.

The official validation_list.txt / testing_list.txt define the splits
(split by speaker hash, so no speaker overlap between train and test).
"""

import os
import random
import tarfile
import urllib.request
from typing import Dict, List, Tuple

from data.reader import read_wav

COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def ensure_speech_commands(data_dir: str = "data/speech_commands") -> str:
    """Download and extract Speech Commands v0.02 if needed (2.4GB)."""
    marker = os.path.join(data_dir, "validation_list.txt")
    if os.path.exists(marker):
        return data_dir

    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "speech_commands_v0.02.tar.gz")
    if not os.path.exists(tar_path):
        print(f"  Downloading {COMMANDS_URL} (2.4GB, this takes a while) ...")
        urllib.request.urlretrieve(COMMANDS_URL, tar_path)

    print("  Extracting ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)

    return data_dir


def _read_list(root: str, name: str) -> set:
    with open(os.path.join(root, name)) as f:
        return {line.strip() for line in f if line.strip()}


def prepare_speech_commands(
    words: List[str],
    data_dir: str = "data/speech_commands",
    train_per_word: int = 300,
    test_per_word: int = 50,
    seed: int = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load a per-word subsample of Speech Commands.

    Train clips come from outside the official validation/testing lists;
    test clips come from the official testing list, so train and test
    speakers never overlap. Subsampling is seeded and deterministic.

    Returns (train_records, test_records); each record has
    samples / sample_rate / text / fname.
    """
    root = ensure_speech_commands(data_dir)
    val_set = _read_list(root, "validation_list.txt")
    test_set = _read_list(root, "testing_list.txt")
    rng = random.Random(seed)

    train_records, test_records = [], []
    for word in words:
        word_dir = os.path.join(root, word)
        files = sorted(f for f in os.listdir(word_dir) if f.endswith(".wav"))

        rel = [f"{word}/{f}" for f in files]
        train_pool = [r for r in rel if r not in val_set and r not in test_set]
        test_pool = [r for r in rel if r in test_set]

        for pool, n, out in [
            (train_pool, train_per_word, train_records),
            (test_pool, test_per_word, test_records),
        ]:
            chosen = rng.sample(pool, min(n, len(pool)))
            for relpath in chosen:
                samples, sr = read_wav(os.path.join(root, relpath))
                out.append({
                    "samples": samples,
                    "sample_rate": sr,
                    "text": word,
                    "fname": relpath,
                })

    return train_records, test_records


if __name__ == "__main__":
    from lexicon import SPEECH_COMMANDS
    tr, te = prepare_speech_commands(SPEECH_COMMANDS.words, train_per_word=5, test_per_word=2)
    print(f"Train: {len(tr)}  Test: {len(te)}")
    r = tr[0]
    print(f"First: {r['fname']} ({len(r['samples'])/r['sample_rate']:.2f}s @ {r['sample_rate']}Hz): {r['text']}")
