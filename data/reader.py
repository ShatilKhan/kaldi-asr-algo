"""
Read Free Spoken Digit Dataset (FSDD) .wav files and metadata.

The dataset has filenames of the form:
    {digit}_{speaker}_{repetition}.wav

This reader:
  - Loads all .wav files from the recordings directory
  - Parses the metadata from filenames
  - Splits into train/test sets by speaker (held-out speakers)
  - Returns raw PCM audio as numpy arrays
"""

import os
import wave
import numpy as np
from typing import List, Tuple, Dict

EXPECTED_SAMPLE_RATE = 8000  # FSDD is 8 kHz


def parse_filename(fname: str) -> Tuple[str, str, int]:
    """
    Parse an FSDD filename into (digit, speaker, repetition).

    Example: '3_jackson_42.wav' → ('3', 'jackson', 42)
    """
    stem = fname.replace(".wav", "")
    parts = stem.split("_")
    digit = parts[0]
    speaker = parts[1] if len(parts) >= 2 else "unknown"
    repetition = int(parts[2]) if len(parts) >= 3 else 0
    return digit, speaker, repetition


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """
    Read a .wav file. Returns (samples, sample_rate).

    samples is a float32 array normalized to [-1, 1].
    """
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    # Convert raw bytes to numpy array
    dtype_map = {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}
    dtype = dtype_map.get(samp_width, np.int16)
    samples = np.frombuffer(raw, dtype=dtype)

    # Handle multi-channel: take first channel
    if n_channels > 1:
        samples = samples[::n_channels]

    # Normalize to float32 in [-1, 1]
    max_val = float(1 << (8 * samp_width - 1))
    samples = samples.astype(np.float32) / max_val

    return samples, sample_rate


def load_fsdd(data_dir: str) -> List[Dict]:
    """
    Load all .wav files from data_dir (the recordings directory).

    Returns a list of dicts:
        {
            "digit": str,
            "speaker": str,
            "repetition": int,
            "samples": np.ndarray (float32),
            "sample_rate": int,
            "path": str
        }
    """
    records = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".wav"):
            continue
        digit, speaker, repetition = parse_filename(fname)
        path = os.path.join(data_dir, fname)
        samples, sr = read_wav(path)
        records.append({
            "digit": digit,
            "speaker": speaker,
            "repetition": repetition,
            "samples": samples,
            "sample_rate": sr,
            "path": path,
        })
    return records


def concat_records(records: List[Dict], group_key: str = "digit") -> List[Dict]:
    """
    Concatenate multiple recordings of the same group into longer utterances.

    This is critical for training: FSDD has ~1-second recordings which give
    only 1-3 frames per HMM state. Concatenating 5 recordings of the same
    digit gives 5× more frames per state, enabling proper GMM training.

    Args:
        records: list of record dicts.
        group_key: field to group by (default "digit").

    Returns:
        List of concatenated record dicts. Each has 'text' set to the group
        value repeated N times (e.g., "0 0 0 0 0" for 5 concatenated "0"s).
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[r[group_key]].append(r)

    concat_records = []
    # Concatenate in groups of CONCAT_SIZE
    CONCAT_SIZE = 4  # 4 utterances → 4× frames → ~12 frames per state
    for key, group in groups.items():
        for i in range(0, len(group), CONCAT_SIZE):
            chunk = group[i:i + CONCAT_SIZE]
            if len(chunk) < 2:
                # Keep single utterances too
                concat_records.append(chunk[0])
                continue
            # Concatenate samples
            all_samples = np.concatenate([r["samples"] for r in chunk])
            all_text = " ".join([r["digit"] for r in chunk])  # e.g., "3 3 3 3"
            # Use the first record's metadata
            rec = dict(chunk[0])
            rec["samples"] = all_samples
            rec["text"] = all_text
            rec["digit"] = key
            concat_records.append(rec)

    return concat_records


def split_by_speaker(
    records: List[Dict], test_speakers: List[str]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split records into train/test by speaker.

    Returns (train_records, test_records).
    """
    train = [r for r in records if r["speaker"] not in test_speakers]
    test = [r for r in records if r["speaker"] in test_speakers]
    return train, test


def get_all_speakers(records: List[Dict]) -> List[str]:
    """Return a sorted list of unique speaker names."""
    return sorted(set(r["speaker"] for r in records))


def default_test_speakers() -> List[str]:
    """
    Return a default set of held-out test speakers for FSDD.
    FSDD has 6 speakers: george, jackson, lucas, nicolas, theo, yweweler
    We hold out theo and yweweler as test speakers (~40% of data).
    """
    return ["theo", "yweweler"]


def prepare_dataset(
    data_dir: str, test_speakers: List[str] = None, concat_training: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Convenience: load FSDD from data_dir and split into train/test.

    When concat_training is True, training recordings of the same digit are
    concatenated into longer sequences (4 utterances per group). This
    significantly improves GMM training by providing more frames per state.

    Returns (train_records, test_records).
    """
    if test_speakers is None:
        test_speakers = default_test_speakers()
    records = load_fsdd(data_dir)
    train, test = split_by_speaker(records, test_speakers)
    if concat_training:
        train = concat_records(train)
    return train, test


if __name__ == "__main__":
    # Test: load FSDD and print summary
    from download_fsdd import ensure_fsdd
    data_dir = ensure_fsdd()
    train, test = prepare_dataset(data_dir)
    print(f"Total: {len(train) + len(test)} recordings")
    print(f"Train: {len(train)} ({len(set(r['speaker'] for r in train))} speakers)")
    print(f"Test:  {len(test)} ({len(set(r['speaker'] for r in test))} speakers)")
    print(f"Speakers: {get_all_speakers(train + test)}")
    print(f"Test speakers: {set(r['speaker'] for r in test)}")
    print(f"Digits: {sorted(set(r['digit'] for r in train))}")
    if train:
        print(f"Example: {train[0]['digit']} by {train[0]['speaker']}, "
              f"{len(train[0]['samples'])} samples at {train[0]['sample_rate']} Hz")
