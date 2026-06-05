"""
Monophone GMM-HMM training pipeline (paper Section IV, Kaldi recipe).

Implements the flat-start monophone training described in steps/train_mono.sh:
  1. Compute MFCCs for all training audio
  2. Flat-start: initialize all phone-state GMMs to global mean/variance
  3. Align: Viterbi alignment on known phone sequence → state per frame
  4. Re-estimate: update GMM parameters from alignment
  5. Repeat 3-4 for N iterations
  6. Split: double GMM components
  7. Repeat 3-4 for N iterations
  8. Split again
  9. Final iterations

Output: trained GMMs (one per pdf-id), ready for decoding.
"""

import numpy as np
import json
import os
import sys
from typing import List, Tuple, Optional
from collections import defaultdict

# Local imports
from data.download_fsdd import ensure_fsdd
from data.reader import prepare_dataset
from feats import extract_mfcc
from hmm import build_all_phone_hmms, build_utterance_hmm, total_pdfs
from gmm import DiagGmm, train_gmm
from lexicon import LEXICON, WORD_MAP, PHONE_MAP, NUM_PHONES, SIL_PHONE


# Training constants
N_ITERS_INITIAL = 3       # iterations before first split
N_ITERS_AFTER_SPLIT = 3   # iterations per split level
N_COMPONENTS = [1, 2, 4]  # GMM component counts at each stage
EM_ITERS = 15             # EM iterations for each training call


# Map FSDD digit strings (0-9) to lexicon word names
_DIGIT_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def build_phone_sequence(transcript: str) -> List[int]:
    """
    Convert a transcript (e.g., "three five seven" or "3 5 7") to a flat
    list of phone IDs. Includes silence at start and end.
    """
    words = transcript.strip().lower().split()
    phones = [SIL_PHONE]
    for word in words:
        # Map digit strings to word names (FSDD stores digits like '0', not 'zero')
        mapped = _DIGIT_MAP.get(word, word)
        if mapped in LEXICON:
            phones.extend(LEXICON[mapped])
    phones.append(SIL_PHONE)
    return phones


def flat_start_initialize(
    frames_list: List[np.ndarray],
    num_pdfs: int,
    n_components: int = 1,
) -> List[DiagGmm]:
    """
    Initialize all phone-state GMMs from the global statistics.

    All pdf-ids start with the same GMM (global mean/variance).
    This is the standard Kaldi "flat-start" approach.

    Args:
        frames_list: list of (N_frames, D) arrays for each training utterance.
        num_pdfs: total number of pdf-ids (3 per phone).
        n_components: initial number of Gaussian components.

    Returns:
        List of DiagGmm, one per pdf-id.
    """
    # Concatenate all frames
    all_frames = np.vstack(frames_list) if frames_list else np.zeros((1, 39))
    D = all_frames.shape[1]

    global_mean = np.mean(all_frames, axis=0)
    global_var = np.var(all_frames, axis=0) + 1e-4

    # Create GMMs for all pdf-ids with slight perturbations for diversity
    rng = np.random.RandomState(42)
    gmms = []
    for p in range(num_pdfs):
        # Perturb each GMM slightly so they're not identical
        # This ensures Viterbi alignment can differentiate states
        mean_perturb = rng.randn(39) * np.sqrt(global_var) * 0.1
        var_perturb = 1.0 + rng.randn(39) * 0.05

        if n_components == 1:
            means = (global_mean + mean_perturb).reshape(1, -1)
            vars_ = (global_var * var_perturb + 1e-4).reshape(1, -1)
            weights = np.ones(1)
        else:
            subset = all_frames[rng.choice(len(all_frames), min(1000, len(all_frames)), replace=False)]
            gmm = train_gmm(subset, n_components=n_components, n_iter=EM_ITERS)
            means, vars_, weights = gmm.means, gmm.vars, gmm.weights

        gmms.append(DiagGmm(means.copy(), vars_.copy(), weights.copy()))

    return gmms


def viterbi_align(
    frames: np.ndarray,
    phone_ids: List[int],
    phone_hmms: list,
    gmms: List[DiagGmm],
) -> np.ndarray:
    """
    Viterbi alignment for one utterance with known phone sequence.

    Given the known phone sequence (from the transcript), concatenate the
    corresponding HMM topologies and find the most likely state sequence.

    Args:
        frames: (num_frames, D) MFCC features.
        phone_ids: list of phone IDs from build_phone_sequence().
        phone_hmms: list of all phone HMMs from build_all_phone_hmms().
        gmms: list of DiagGmm per pdf-id.

    Returns:
        (num_frames,) array of pdf-id assignments for each frame.
    """
    num_frames = frames.shape[0]
    num_pdfs = len(gmms)

    if num_frames == 0:
        return np.array([], dtype=int)

    # Build concatenated HMM for this phone sequence
    states, npdf = build_utterance_hmm(phone_ids, phone_hmms)
    num_states = len(states)

    # Precompute log-likelihoods for each (frame, pdf-id) using vectorized scoring
    from gmm import DiagGmm
    log_likes = DiagGmm.score_batch_all(gmms, frames)  # (num_frames, num_pdfs)

    # Viterbi DP on the HMM
    dp = np.full((num_frames, num_states), -1e30)
    back = np.zeros((num_frames, num_states), dtype=int)

    # First frame
    pdf_0 = states[0]["pdf_id"]
    dp[0, 0] = log_likes[0, pdf_0]

    # Pre-compute transition log-probs for vectorization
    self_loop = np.array([s["self_loop_logp"] for s in states])
    forward = np.array([s["forward_logp"] for s in states])
    pdf_ids = np.array([s["pdf_id"] for s in states])

    # Fill DP table (vectorized inner loop)
    for t in range(1, num_frames):
        # Self-loop: from same state
        dp_candidates = dp[t - 1] + self_loop

        # Forward: from previous state (state s-1)
        forward_candidates = np.full(num_states, -1e30)
        forward_candidates[1:] = dp[t - 1, :-1] + forward[:-1]

        # Take the better of self-loop and forward
        best = np.maximum(dp_candidates, forward_candidates)
        back[t] = np.where(dp_candidates >= forward_candidates, np.arange(num_states), np.arange(num_states) - 1)
        # Fix: np.where approach is tricky. Use argmax approach instead:
        stacked = np.column_stack([dp_candidates, forward_candidates])
        best = np.max(stacked, axis=1)
        back[t] = np.argmax(stacked, axis=1)

        # Add emission score
        dp[t] = best + log_likes[t, pdf_ids]

    # Traceback
    best_final = int(np.argmax(dp[-1]))
    align = np.zeros(num_frames, dtype=int)
    align[-1] = pdf_ids[best_final]
    prev_state = best_final
    for t in range(num_frames - 2, -1, -1):
        prev_state = back[t + 1, prev_state]
        align[t] = pdf_ids[prev_state]

    return align


def reestimate_gmms(
    alignments: List[np.ndarray],
    frames_list: List[np.ndarray],
    old_gmms: List[DiagGmm],
    n_components: int,
) -> List[DiagGmm]:
    """
    Re-estimate GMM parameters from aligned frame-to-state assignments.

    For each pdf-id, collect all frames that were assigned to it,
    then train a new GMM on those frames.

    Args:
        alignments: list of (num_frames,) arrays (one per utterance).
        frames_list: list of (num_frames, D) arrays.
        old_gmms: current GMMs (used if no frames assigned to a pdf-id).
        n_components: number of Gaussian components.

    Returns:
        Updated list of DiagGmm.
    """
    num_pdfs = len(old_gmms)
    D = frames_list[0].shape[1]

    # Collect frames per pdf-id
    pdf_frames = [[] for _ in range(num_pdfs)]

    for frames, align in zip(frames_list, alignments):
        for t, pdf_id in enumerate(align):
            if 0 <= pdf_id < num_pdfs:
                pdf_frames[pdf_id].append(frames[t])

    # Train new GMMs
    new_gmms = []
    for p in range(num_pdfs):
        if len(pdf_frames[p]) >= n_components * 5:
            train_frames = np.array(pdf_frames[p])
            gmm = train_gmm(train_frames, n_components=n_components, n_iter=EM_ITERS)
        else:
            # Not enough data: keep old GMM or create default
            gmm = old_gmms[p]
            if n_components != gmm.K:
                # Need to adjust component count
                if n_components > gmm.K:
                    gmm = gmm.split()
                    while gmm.K < n_components:
                        gmm = gmm.split()
                else:
                    # Keep old for now
                    pass
        new_gmms.append(gmm)

    return new_gmms


def compute_total_log_likelihood(
    frames_list: List[np.ndarray],
    alignments: List[np.ndarray],
    gmms: List[DiagGmm],
) -> float:
    """
    Compute total log-likelihood of all training data under the current model.
    Used to track training progress.
    """
    total = 0.0
    for frames, align in zip(frames_list, alignments):
        for t, pdf_id in enumerate(align):
            if 0 <= pdf_id < len(gmms):
                total += gmms[pdf_id].log_likelihood(frames[t])
    return total


def train(
    transcripts: List[str],
    frames_list: List[np.ndarray],
    num_phones: int = NUM_PHONES,
    component_levels: List[int] = None,
    iters_per_level: int = N_ITERS_AFTER_SPLIT,
    verbose: bool = True,
) -> List[DiagGmm]:
    """
    Full monophone training pipeline.

    Args:
        transcripts: list of text transcripts.
        frames_list: list of (num_frames, D) MFCC arrays.
        num_phones: number of phones.
        component_levels: list of GMM component counts [1, 2, 4].
        iters_per_level: iterations per component level.
        verbose: if True, print progress.

    Returns:
        List of trained DiagGmm, one per pdf-id.
    """
    if component_levels is None:
        component_levels = N_COMPONENTS

    # Build phone HMMs
    phone_hmms = build_all_phone_hmms(num_phones)
    npdfs = total_pdfs(phone_hmms)
    D = frames_list[0].shape[1]

    if verbose:
        print(f"Training {num_phones} phones, {npdfs} pdf-ids, {D}-dim features")
        print(f"  {len(frames_list)} utterances")

    # Build phone sequences for all utterances
    all_phone_seqs = [build_phone_sequence(t) for t in transcripts]

    # Initialize GMMs
    gmms = flat_start_initialize(frames_list, npdfs, n_components=component_levels[0])

    # Training loop
    for level_idx, n_comp in enumerate(component_levels):
        if verbose:
            print(f"\n--- Level {level_idx + 1}: {n_comp} Gaussians ---")

        # Adjust GMMs to the right component count
        if level_idx > 0:
            if verbose:
                print(f"  Splitting from {component_levels[level_idx - 1]} to {n_comp} Gaussians")
            # Split each GMM to reach target component count
            gmms = [gmm.split() for gmm in gmms]

        for it in range(iters_per_level):
            # Align all utterances
            alignments = []
            for i, (frames, phones) in enumerate(zip(frames_list, all_phone_seqs)):
                if npdfs > 0:
                    align = viterbi_align(frames, phones, phone_hmms, gmms)
                    alignments.append(align)

            # Re-estimate
            gmms = reestimate_gmms(alignments, frames_list, gmms, n_comp)

            if verbose:
                ll = compute_total_log_likelihood(frames_list, alignments, gmms)
                print(f"  Iter {it + 1}/{iters_per_level}: log-likelihood = {ll:.1f}")

    if verbose:
        print("\nTraining complete.")
        print(f"  Final model: {len(gmms)} pdf-ids, "
              f"{sum(gmm.K for gmm in gmms)} total Gaussians")

    return gmms


def save_model(gmms: List[DiagGmm], path: str) -> None:
    """Save trained GMMs to a JSON file."""
    data = {
        "num_pdfs": len(gmms),
        "feature_dim": gmms[0].D,
        "gmms": [g.to_dict() for g in gmms],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_model(path: str) -> List[DiagGmm]:
    """Load trained GMMs from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [DiagGmm.from_dict(g) for g in data["gmms"]]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Load FSDD
    data_dir = ensure_fsdd()
    train_records, test_records = prepare_dataset(data_dir)

    print(f"Loaded {len(train_records)} training, {len(test_records)} test recordings")

    # Extract transcripts and MFCCs for training
    transcripts = [r["digit"] for r in train_records]
    frames_list = [extract_mfcc(r["samples"], r["sample_rate"]) for r in train_records]

    # Filter out empty frames
    valid = [(t, f) for t, f in zip(transcripts, frames_list) if f.shape[0] > 0]
    if len(valid) < len(transcripts):
        print(f"Filtered out {len(transcripts) - len(valid)} empty utterances")
    transcripts, frames_list = zip(*valid) if valid else ([], [])
    transcripts = list(transcripts)
    frames_list = list(frames_list)

    # Train
    gmms = train(transcripts, frames_list, NUM_PHONES, verbose=True)

    # Save model
    os.makedirs("models", exist_ok=True)
    save_model(gmms, "models/trained_gmms.json")
    print(f"\nModel saved to models/trained_gmms.json")

    # Quick test: score first few frames of first test utterance
    if test_records:
        test_feats = extract_mfcc(test_records[0]["samples"], test_records[0]["sample_rate"])
        if test_feats.shape[0] > 0:
            scores = [gmms[p].log_likelihood(test_feats[0]) for p in range(min(5, len(gmms)))]
            print(f"  Test utterance '{test_records[0]['digit']}':")
            print(f"    First frame scores (first 5 pdf-ids): {[f'{s:.2f}' for s in scores]}")
