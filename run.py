"""
Kaldi mini-implementation: end-to-end monophone GMM-HMM digit recognizer.

Pipeline:
  1. Download FSDD dataset
  2. Compute MFCC features
  3. Train monophone GMM-HMM (flat start → align → re-estimate → split)
  4. Build HCLG decoder graph (H ∘ L ∘ G)
  5. Decode test utterances
  6. Evaluate WER/CER/SER with bootstrap CI
  7. Show sample outputs
"""

import sys
import os
import time
import random
import numpy as np

# Local imports
from data.download_fsdd import ensure_fsdd
from data.reader import prepare_dataset
from feats import extract_mfcc
from hmm import build_all_phone_hmms, total_pdfs
from gmm import DiagGmm
from lexicon import LEXICON, WORD_MAP, WORD_IDS, WORDS, NUM_PHONES
from lm import train_lm
from fst import compose, best_path
from decoder import build_h_fst, build_l_fst, build_g_fst, decode, assemble_hclg
from train import train, save_model, load_model, build_phone_sequence
from eval import WERStats


def main():
    print("=" * 60)
    print("  Kaldi Mini-Implementation: Digit Speech Recognizer")
    print("  Povey et al., 'The Kaldi Speech Recognition Toolkit' (2011)")
    print("=" * 60)

    # ---- Step 1: Data ----
    print("\n[1/5] Loading data...")
    data_dir = ensure_fsdd()
    train_records, test_records = prepare_dataset(data_dir)
    print(f"  Train: {len(train_records)} utterances ({len(set(r['speaker'] for r in train_records))} speakers)")
    print(f"  Test:  {len(test_records)} utterances ({len(set(r['speaker'] for r in test_records))} speakers)")

    # ---- Step 2: Feature extraction ----
    print("\n[2/5] Extracting MFCC features...")
    t0 = time.time()

    train_transcripts = [r["digit"] for r in train_records]
    test_transcripts = [r["digit"] for r in test_records]

    train_frames = [extract_mfcc(r["samples"], r["sample_rate"]) for r in train_records]
    test_frames = [extract_mfcc(r["samples"], r["sample_rate"]) for r in test_records]

    # Filter empty
    train_valid = [(t, f) for t, f in zip(train_transcripts, train_frames) if f.shape[0] > 0]
    test_valid = [(t, f) for t, f in zip(test_transcripts, test_frames) if f.shape[0] > 0]
    train_transcripts, train_frames = zip(*train_valid)
    test_transcripts, test_frames = zip(*test_valid)

    t1 = time.time()
    print(f"  Train: {len(train_frames)} utterances, {sum(f.shape[0] for f in train_frames)} frames")
    print(f"  Test:  {len(test_frames)} utterances, {sum(f.shape[0] for f in test_frames)} frames")
    print(f"  Time:  {t1 - t0:.1f}s")

    # ---- Step 3: Train monophone GMM-HMM ----
    print("\n[3/5] Training monophone GMM-HMM...")
    t0 = time.time()

    train_transcripts_list = list(train_transcripts)
    train_frames_list = list(train_frames)

    gmms = train(train_transcripts_list, train_frames_list, NUM_PHONES, verbose=True)

    t1 = time.time()
    print(f"  Training time: {(t1 - t0)/60:.1f} min")

    # ---- Step 4: Build decoder graph ----
    print("\n[4/5] Building HCLG decoder graph (H ∘ L ∘ G)...")
    t0 = time.time()

    phone_hmms = build_all_phone_hmms(NUM_PHONES)

    # Train LM on training transcripts
    lm = train_lm(train_transcripts_list)
    print(f"  LM: {lm}")

    # Build HCLG
    hclg = assemble_hclg(phone_hmms, lm)

    t1 = time.time()
    print(f"  Decoder graph: {hclg.num_states} states, {hclg.num_arcs} arcs")
    print(f"  Build time: {t1 - t0:.1f}s")

    # ---- Step 5: Decode and evaluate ----
    print("\n[5/5] Decoding test utterances...")
    t0 = time.time()

    stats = WERStats()
    refs_list = []
    hyps_list = []

    for i, (frames, ref_text) in enumerate(zip(test_frames, test_transcripts)):
        # Decode
        word_ids = decode(frames, gmms, hclg)
        hyp_words = [WORD_IDS[wid] for wid in word_ids if wid in WORD_IDS]

        ref_words = [ref_text]  # single word per utterance in FSDD

        refs_list.append(ref_words)
        hyps_list.append(hyp_words)

        stats.add(ref_words, hyp_words)

        if (i + 1) % 100 == 0:
            print(f"    Decoded {i + 1}/{len(test_frames)}...")

    t1 = time.time()
    print(f"  Decoded {len(test_frames)} utterances in {t1 - t0:.1f}s")

    # Results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print()
    print(stats.report())
    print(stats.sample_outputs(refs_list, hyps_list, n=10))
    print()

    # Per-digit WER breakdown
    print("--- Per-digit accuracy ---")
    digit_correct = {d: 0 for d in WORDS}
    digit_total = {d: 0 for d in WORDS}
    for ref, hyp in zip(refs_list, hyps_list):
        digit = ref[0]
        digit_total[digit] = digit_total.get(digit, 0) + 1
        if ref == hyp:
            digit_correct[digit] = digit_correct.get(digit, 0) + 1

    for digit in sorted(digit_total.keys()):
        total = digit_total[digit]
        correct = digit_correct.get(digit, 0)
        pct = 100.0 * correct / total if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {digit:8s}: {bar} {pct:.0f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
