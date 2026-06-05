"""
Download and read CMU AN4 dataset.

AN4 is a small US English alphanumeric speech dataset with ~1100 utterances
from ~50 speakers. It's the standard "hello world" dataset in Kaldi.

Pipeline:
    1. Download an4_sphere.tar.gz (64 MB) from CMU
    2. Extract SPHERE audio files
    3. Convert to .wav via ffmpeg
    4. Read transcriptions and split train/test

Format:
    - Audio: NIST SPHERE (.sph), 16-bit, 16 kHz
    - Transcriptions: an4_train.transcription, an4_test.transcription
    - Lexicon: an4.dic
"""

import os
import re
import subprocess
import urllib.request
import tarfile
import wave
import struct
import numpy as np
from typing import List, Tuple, Dict

AN4_URL = "http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz"
DEST_DIR = os.path.join(os.path.dirname(__file__), "an4")


def download_and_extract() -> str:
    """Download and extract AN4. Returns path to extracted root."""
    if os.path.isdir(DEST_DIR) and os.listdir(DEST_DIR):
        print(f"AN4 already exists at {DEST_DIR}")
        return DEST_DIR

    os.makedirs(DEST_DIR, exist_ok=True)
    tarball = os.path.join(DEST_DIR, "an4_sphere.tar.gz")

    if not os.path.exists(tarball):
        print(f"Downloading AN4 from {AN4_URL} ...")
        urllib.request.urlretrieve(AN4_URL, tarball)
        print("Download complete.")

    print("Extracting...")
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=DEST_DIR)
    print("Extraction complete.")

    # Remove tarball
    os.remove(tarball)

    # The extracted structure is: data/an4/an4_sphere/
    extracted = os.path.join(DEST_DIR, "an4_sphere")
    if os.path.isdir(extracted):
        # Move contents up
        for item in os.listdir(extracted):
            src = os.path.join(extracted, item)
            dst = os.path.join(DEST_DIR, item)
            os.rename(src, dst)
        os.rmdir(extracted)

    return DEST_DIR


def sph_to_wav(sph_path: str, wav_path: str) -> None:
    """Convert SPHERE audio to WAV using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", sph_path, "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", "16000", wav_path],
        capture_output=True, check=True
    )


def ensure_wav(sph_path: str, wav_dir: str) -> str:
    """Convert .sph to .wav if not already done. Returns path to .wav."""
    basename = os.path.splitext(os.path.basename(sph_path))[0]
    wav_path = os.path.join(wav_dir, basename + ".wav")
    if not os.path.exists(wav_path):
        os.makedirs(wav_dir, exist_ok=True)
        sph_to_wav(sph_path, wav_path)
    return wav_path


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a .wav file, return (samples, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def parse_transcription(line: str) -> Tuple[str, str, str]:
    """
    Parse an AN4 transcription line.

    Format: '<s> WORDS </s> (source_id)'

    Returns (utterance_id, transcript_text, speaker_id).
    """
    line = line.strip()
    if not line:
        return None, None, None

    # Extract words
    words_match = re.search(r'^(.*) \(', line)
    if not words_match:
        return None, None, None
    words = words_match.group(1).strip()
    # Remove <s> and </s> markers
    if words.startswith("<s> "):
        words = words[4:]
    if words.endswith(" </s>"):
        words = words[:-5]

    # Extract source (speaker ID)
    source_match = re.search(r'\((.*)\)', line)
    if not source_match:
        return None, None, None
    source = source_match.group(1)

    # Parse source: prep-mid-last → utterance_id = mid-prep-last
    parts = source.split("-")
    if len(parts) >= 3:
        pre, mid, last = parts[0], parts[1], parts[2]
        utt_id = f"{mid}-{pre}-{last}"
        speaker_id = mid
    else:
        utt_id = source
        speaker_id = source

    return utt_id, words, speaker_id


def load_an4(an4_root: str, wav_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load AN4 dataset.

    Returns (train_records, test_records) where each record is:
        {"utt_id": str, "speaker": str, "text": str, "samples": np.ndarray,
         "sample_rate": int, "path": str}
    """
    etc_dir = os.path.join(an4_root, "etc")
    wav_out_dir = wav_dir

    def load_split(transcript_file: str, sph_subdir: str) -> List[Dict]:
        records = []
        trans_path = os.path.join(etc_dir, transcript_file)
        if not os.path.exists(trans_path):
            print(f"  Warning: {trans_path} not found")
            return records

        with open(trans_path) as f:
            lines = f.readlines()

        for line in lines:
            utt_id, text, speaker = parse_transcription(line)
            if utt_id is None:
                continue

            # Find the .sph file
            # Format: wav/{sph_subdir}/{speaker}/{source}.sph
            # Parse source from utt_id: mid-pre-last → source = pre-mid-last
            parts = utt_id.split("-")
            if len(parts) >= 3:
                source = f"{parts[1]}-{parts[0]}-{parts[2]}"
            else:
                source = utt_id

            sph_path = os.path.join(an4_root, "wav", sph_subdir, speaker, f"{source}.sph")
            if not os.path.exists(sph_path):
                # Try alternate paths
                sph_path = os.path.join(an4_root, "wav", sph_subdir, source + ".sph")

            if not os.path.exists(sph_path):
                # Try finding any .sph with matching name
                for root, dirs, files in os.walk(os.path.join(an4_root, "wav")):
                    for f in files:
                        if f.endswith(".sph") and source in f:
                            sph_path = os.path.join(root, f)
                            break

            if os.path.exists(sph_path):
                try:
                    wav_path = ensure_wav(sph_path, wav_out_dir)
                    samples, sr = read_wav(wav_path)
                    records.append({
                        "utt_id": utt_id,
                        "speaker": speaker,
                        "text": text,
                        "samples": samples,
                        "sample_rate": sr,
                        "path": wav_path,
                    })
                except Exception as e:
                    print(f"  Warning: could not read {sph_path}: {e}")

        return records

    print(f"  Loading training data from {etc_dir}...")
    train = load_split("an4_train.transcription", "an4_clstk")
    print(f"  Loaded {len(train)} training utterances")

    print(f"  Loading test data...")
    test = load_split("an4_test.transcription", "an4test_clstk")
    print(f"  Loaded {len(test)} test utterances")

    return train, test


def prepare_an4() -> Tuple[List[Dict], List[Dict]]:
    """
    Convenience: download and prepare AN4 in one call.

    Returns (train_records, test_records).
    """
    an4_root = download_and_extract()
    wav_dir = os.path.join(an4_root, "wav_converted")
    return load_an4(an4_root, wav_dir)


if __name__ == "__main__":
    train, test = prepare_an4()
    print(f"\nTrain: {len(train)} utterances")
    print(f"Test:  {len(test)} utterances")
    if train:
        speakers = set(r["speaker"] for r in train)
        print(f"Train speakers: {len(speakers)} ({sorted(speakers)[:10]}...)")
        print(f"Example: {train[0]['utt_id']}: \"{train[0]['text']}\"")
    if test:
        print(f"Example test: {test[0]['utt_id']}: \"{test[0]['text']}\"")
