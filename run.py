"""
Kaldi mini-implementation: end-to-end monophone GMM-HMM recognizer.

Pipeline:
  1. Download dataset (FSDD digits or YesNo)
  2. Compute MFCC features
  3. Train monophone GMM-HMM (flat start → align → re-estimate → split)
  4. Build HCLG decoder graph (H ∘ L ∘ G)
  5. Decode test utterances
  6. Evaluate WER/CER/SER with bootstrap CI

Usage:
  uv run python run.py --task yesno
  uv run python run.py --task digits
"""

import argparse
import time

import numpy as np

from feats import extract_mfcc, trim_silence
from hmm import build_all_phone_hmms
from lexicon import DIGITS, YESNO, DIGIT_WORDS
from lm import train_lm
from decoder import decode, assemble_hclg
from train import train
from eval import WERStats


def load_digits():
    """FSDD: spoken digits, one word per test utterance."""
    from data.download_fsdd import ensure_fsdd
    from data.reader import prepare_dataset

    data_dir = ensure_fsdd()
    train_records, test_records = prepare_dataset(data_dir)

    def text_of(r):
        raw = r.get("text", r["digit"])
        return " ".join(DIGIT_WORDS[d] for d in raw.split())

    train_data = [(text_of(r), r["samples"], r["sample_rate"]) for r in train_records]
    test_data = [(text_of(r), r["samples"], r["sample_rate"]) for r in test_records]
    return train_data, test_data


def load_yesno():
    """YesNo: 8-word yes/no sequences, 31 train / 29 test."""
    from data.yesno import prepare_yesno

    train_records, test_records = prepare_yesno()
    train_data = [(r["text"], r["samples"], r["sample_rate"]) for r in train_records]
    test_data = [(r["text"], r["samples"], r["sample_rate"]) for r in test_records]
    return train_data, test_data


# Per-task settings. word_penalty and acoustic_scale were tuned on the
# yesno dev split; trim applies energy endpointing (see feats.trim_silence).
TASKS = {
    "digits": {
        "loader": load_digits, "lexicon": DIGITS, "sil_between": False,
        "trim": False, "word_penalty": 0.0, "acoustic_scale": 0.0833,
        "component_levels": [1, 2, 4],
    },
    "yesno": {
        "loader": load_yesno, "lexicon": YESNO, "sil_between": True,
        "trim": True, "word_penalty": 6.0, "acoustic_scale": 0.05,
        "component_levels": [1, 2, 4, 8],
    },
}


def main():
    parser = argparse.ArgumentParser(description="Mini Kaldi monophone recognizer")
    parser.add_argument("--task", choices=sorted(TASKS.keys()), default="yesno")
    args = parser.parse_args()

    np.random.seed(0)  # reproducible EM runs

    task = TASKS[args.task]
    lex = task["lexicon"]

    print("=" * 60)
    print(f"  Kaldi Mini-Implementation — task: {args.task}")
    print("  Povey et al., 'The Kaldi Speech Recognition Toolkit' (2011)")
    print("=" * 60)

    # ---- Step 1: Data ----
    print("\n[1/5] Loading data...")
    train_data, test_data = task["loader"]()
    print(f"  Train: {len(train_data)} utterances")
    print(f"  Test:  {len(test_data)} utterances")

    # ---- Step 2: Feature extraction ----
    print("\n[2/5] Extracting MFCC features...")
    t0 = time.time()

    def feats_of(samples, sr):
        if task["trim"]:
            samples = trim_silence(samples, sr)
        return extract_mfcc(samples, sr)

    train_set = [(text, feats_of(samples, sr)) for text, samples, sr in train_data]
    test_set = [(text, feats_of(samples, sr)) for text, samples, sr in test_data]
    train_set = [(t, f) for t, f in train_set if f.shape[0] > 0]
    test_set = [(t, f) for t, f in test_set if f.shape[0] > 0]

    train_transcripts = [t for t, _ in train_set]
    train_frames = [f for _, f in train_set]
    test_transcripts = [t for t, _ in test_set]
    test_frames = [f for _, f in test_set]

    print(f"  Train: {len(train_frames)} utterances, {sum(f.shape[0] for f in train_frames)} frames")
    print(f"  Test:  {len(test_frames)} utterances, {sum(f.shape[0] for f in test_frames)} frames")
    print(f"  Time:  {time.time() - t0:.1f}s")

    # ---- Step 3: Train monophone GMM-HMM ----
    print("\n[3/5] Training monophone GMM-HMM...")
    t0 = time.time()
    gmms = train(train_transcripts, train_frames, lex,
                 component_levels=task["component_levels"],
                 sil_between=task["sil_between"], verbose=True)
    print(f"  Training time: {(time.time() - t0)/60:.1f} min")

    # ---- Step 4: Build decoder graph ----
    print("\n[4/5] Building HCLG decoder graph (H ∘ L ∘ G)...")
    t0 = time.time()
    phone_hmms = build_all_phone_hmms(lex.num_phones)
    lm = train_lm(train_transcripts, lex)
    print(f"  LM: {lm}")
    hclg = assemble_hclg(phone_hmms, lm, lex, word_penalty=task["word_penalty"])
    print(f"  Build time: {time.time() - t0:.1f}s")

    # ---- Step 5: Decode and evaluate ----
    print("\n[5/5] Decoding test utterances...")
    t0 = time.time()

    stats = WERStats()
    refs_list = []
    hyps_list = []

    for i, (frames, ref_text) in enumerate(zip(test_frames, test_transcripts)):
        word_ids = decode(frames, gmms, hclg, lex.num_words,
                          acoustic_scale=task["acoustic_scale"])
        hyp_words = [lex.word_ids[wid] for wid in word_ids if wid in lex.word_ids]
        ref_words = ref_text.split()

        refs_list.append(ref_words)
        hyps_list.append(hyp_words)
        stats.add(ref_words, hyp_words)

        if (i + 1) % 10 == 0:
            print(f"    Decoded {i + 1}/{len(test_frames)}...")

    print(f"  Decoded {len(test_frames)} utterances in {time.time() - t0:.1f}s")

    # Results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print()
    print(stats.report())
    print(stats.sample_outputs(refs_list, hyps_list, n=10))


if __name__ == "__main__":
    main()
