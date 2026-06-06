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
from lexicon import Lexicon


# Training constants
N_ITERS_INITIAL = 5       # iterations before first split
N_ITERS_AFTER_SPLIT = 5   # iterations per split level
N_COMPONENTS = [1, 2, 4]  # GMM component counts at each stage
EM_ITERS = 20             # EM iterations for each training call
MIN_OCCUPANCY = 1         # minimum frames per pdf-id to update GMM


def build_phone_sequence(transcript: str, lex: Lexicon, sil_between: bool = False) -> List[int]:
    """
    Convert a transcript (e.g., "three five seven") to a flat list of phone
    IDs. Includes silence at start and end, and optionally between words
    (useful for datasets with real pauses, like yesno).
    """
    words = transcript.strip().lower().split()
    phones = [lex.sil_phone]
    for i, word in enumerate(words):
        if word in lex.lexicon:
            if sil_between and i > 0:
                phones.append(lex.sil_phone)
            phones.extend(lex.lexicon[word])
    phones.append(lex.sil_phone)
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

    # Kaldi flat-start: ALL pdf-ids get the SAME GMM (global stats).
    # Diversity comes from equal alignment pass, not from perturbed means.
    # Each state sees different frames from the equal alignment → different GMMs
    # after the first re-estimation.
    gmms = []
    for _ in range(num_pdfs):
        if n_components == 1:
            means = global_mean.reshape(1, -1)
            vars_ = global_var.reshape(1, -1)
            weights = np.ones(1)
        else:
            subset = all_frames[np.random.choice(len(all_frames), min(1000, len(all_frames)), replace=False)]
            gmm = train_gmm(subset, n_components=n_components, n_iter=EM_ITERS)
            means, vars_, weights = gmm.means, gmm.vars, gmm.weights

        gmms.append(DiagGmm(means.copy(), vars_.copy(), weights.copy()))

    return gmms


def viterbi_align(
    frames: np.ndarray,
    phone_ids: List[int],
    phone_hmms: list,
    gmms: List[DiagGmm],
    acoustic_scale: float = 0.1,
    transition_scale: float = 1.0,
    self_loop_scale: float = 0.1,
    silence_boost: float = 1.0,
) -> np.ndarray:
    """
    Viterbi alignment for one utterance with known phone sequence.

    Given the known phone sequence (from the transcript), concatenate the
    corresponding HMM topologies and find the most likely state sequence.

    Matches Kaldi's gmm-align-compiled with --scale-opts options.

    Args:
        frames: (num_frames, D) MFCC features.
        phone_ids: list of phone IDs from build_phone_sequence().
        phone_hmms: list of all phone HMMs from build_all_phone_hmms().
        gmms: list of DiagGmm per pdf-id.
        acoustic_scale: weight for GMM emission scores (default 0.1).
        transition_scale: weight for transition probabilities (default 1.0).
        self_loop_scale: weight for self-loop probabilities (default 0.1).
        silence_boost: boost silence state scores (default 1.0 = no boost).

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

    # Precompute log-likelihoods
    from gmm import DiagGmm
    log_likes = DiagGmm.score_batch_all(gmms, frames)

    # Apply silence boost: multiply silence (phone_id=0) state scores by silence_boost
    # In log space: multiply = add log(factor)
    if silence_boost != 1.0:
        for s_idx, state in enumerate(states):
            # Find which phone this state belongs to by checking its pdf_id
            phone_for_state = phone_ids[min(s_idx // 3, len(phone_ids) - 1)]
            if phone_for_state == 0:  # SIL phone
                pdf_id = state["pdf_id"]
                log_likes[:, pdf_id] += np.log(silence_boost)

    # Pre-compute transition log-probs with scaling
    self_loop = np.array([s["self_loop_logp"] for s in states]) * self_loop_scale
    forward = np.array([s["forward_logp"] for s in states]) * transition_scale
    pdf_ids = np.array([s["pdf_id"] for s in states])

    # Viterbi DP on the HMM
    dp = np.full((num_frames, num_states), -1e30)
    back = np.zeros((num_frames, num_states), dtype=int)

    # First frame
    dp[0, 0] = log_likes[0, pdf_ids[0]] * acoustic_scale

    # Fill DP table
    for t in range(1, num_frames):
        # Self-loop and forward candidates
        dp_self = dp[t - 1] + self_loop
        dp_forward = np.full(num_states, -1e30)
        dp_forward[1:] = dp[t - 1, :-1] + forward[:-1]

        # Pick best predecessor
        stacked = np.column_stack([dp_self, dp_forward])
        best = np.max(stacked, axis=1)
        back[t] = np.argmax(stacked, axis=1)

        # Add emission score
        dp[t] = best + log_likes[t, pdf_ids] * acoustic_scale

    # Traceback. This is FORCED alignment: the path must end in the LAST
    # HMM state of the sequence, otherwise later phones get no frames and
    # their GMMs never move off the flat start.
    best_final = num_states - 1
    if dp[-1, best_final] <= -1e29:
        # Last state unreachable (utterance shorter than the state chain);
        # fall back to the best reachable state.
        best_final = int(np.argmax(dp[-1]))
    # back[t, s] is a CHOICE flag from argmax([self, forward]):
    #   0 = predecessor is s itself (self-loop), 1 = predecessor is s - 1.
    # Convert flag -> predecessor state by subtracting it.
    align = np.zeros(num_frames, dtype=int)
    align[-1] = pdf_ids[best_final]
    prev_state = best_final
    for t in range(num_frames - 2, -1, -1):
        prev_state = prev_state - back[t + 1, prev_state]
        align[t] = pdf_ids[prev_state]

    return align


def equal_align(
    frames: np.ndarray,
    phone_ids: List[int],
    phone_hmms: list,
) -> np.ndarray:
    """
    Equal alignment pass 0 (Kaldi's align-equal).

    Divides frames equally among all HMM states in the known phone sequence.
    No GMM scores used — just uniform distribution.

    This is the critical bootstrap step: it gives every state some training
    data for the first GMM update, breaking the "all frames go to silence" trap.

    Args:
        frames: (num_frames, D) MFCC features.
        phone_ids: list of phone IDs.
        phone_hmms: list of all phone HMMs.

    Returns:
        (num_frames,) array of pdf-id assignments (one per frame), equally distributed.
    """
    num_frames = frames.shape[0]
    if num_frames == 0:
        return np.array([], dtype=int)

    states, _ = build_utterance_hmm(phone_ids, phone_hmms)
    num_states = len(states)

    # Equal division: some get floor, some get ceil
    base = num_frames // num_states
    extra = num_frames % num_states

    align = np.zeros(num_frames, dtype=int)
    idx = 0
    for s in range(num_states):
        count = base + (1 if s < extra else 0)
        align[idx: idx + count] = states[s]["pdf_id"]
        idx += count

    return align


def reestimate_gmms(
    alignments: List[np.ndarray],
    frames_list: List[np.ndarray],
    old_gmms: List[DiagGmm],
    n_components: int,
    min_occupancy: int = MIN_OCCUPANCY,
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

    # Stack everything once, then select per pdf-id with a boolean mask.
    # Much lighter than appending 100k+ frame rows into Python lists.
    all_frames = np.vstack(frames_list)
    all_align = np.concatenate(alignments).astype(int)

    new_gmms = []
    for p in range(num_pdfs):
        rows = all_frames[all_align == p]
        if len(rows) >= min_occupancy:
            actual_k = min(len(rows), n_components)
            gmm = train_gmm(rows, n_components=actual_k, n_iter=EM_ITERS)
        else:
            # Keep old GMM at whatever K it had
            gmm = old_gmms[p]
        new_gmms.append(gmm)

    # Step 2: ensure ALL GMMs have exactly n_components by splitting
    for p in range(num_pdfs):
        while new_gmms[p].K < n_components:
            new_gmms[p] = new_gmms[p].split()

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
    from gmm import DiagGmm
    total = 0.0
    for frames, align in zip(frames_list, alignments):
        # Log-likelihood of assigned frames under their GMMs
        valid = np.array([(p >= 0 and p < len(gmms)) for p in align])
        if not np.any(valid):
            continue
        pdf_ids = align[valid].astype(int)
        # Score specific pdf-ids for specific frames using individual GMM scoring
        for idx in np.where(valid)[0]:
            pdf_id = align[idx]
            if 0 <= pdf_id < len(gmms):
                total += gmms[pdf_id].log_likelihood(frames[idx])
    return total


def train(
    transcripts: List[str],
    frames_list: List[np.ndarray],
    lex: Lexicon,
    component_levels: List[int] = None,
    iters_per_level: int = N_ITERS_AFTER_SPLIT,
    sil_between: bool = False,
    verbose: bool = True,
) -> List[DiagGmm]:
    """
    Full monophone training pipeline.

    Args:
        transcripts: list of text transcripts.
        frames_list: list of (num_frames, D) MFCC arrays.
        lex: the task Lexicon (phone set + pronunciations).
        component_levels: list of GMM component counts [1, 2, 4].
        iters_per_level: iterations per component level.
        sil_between: insert silence between words in training phone sequences.
        verbose: if True, print progress.

    Returns:
        List of trained DiagGmm, one per pdf-id.
    """
    if component_levels is None:
        component_levels = N_COMPONENTS

    # Build phone HMMs
    phone_hmms = build_all_phone_hmms(lex.num_phones)
    npdfs = total_pdfs(phone_hmms)
    D = frames_list[0].shape[1]

    if verbose:
        print(f"Training {lex.num_phones} phones, {npdfs} pdf-ids, {D}-dim features")
        print(f"  {len(frames_list)} utterances")

    # Build phone sequences for all utterances
    all_phone_seqs = [build_phone_sequence(t, lex, sil_between=sil_between) for t in transcripts]

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
            gmms = [gmm.split() for gmm in gmms]

        # Keep previous alignment for iterations without realignment
        prev_alignments = None

        for it in range(iters_per_level):
            # Determine whether to realign this iteration
            do_align = (level_idx == 0 and it == 0)  # Pass 0: always equal-align
            if not do_align:
                # For all levels: realign at specific iterations
                # Level 1 (1-GMM): realign every iteration
                # Level 2+ (2+ GMMs): realign every other iteration
                if level_idx == 0:
                    do_align = True  # every iteration at level 1
                else:
                    do_align = (it % 2 == 0)  # every other for higher levels

            if do_align:
                alignments = []
                if level_idx == 0 and it == 0:
                    # Pass 0: equal alignment (Kaldi's align-equal)
                    for frames, phones in zip(frames_list, all_phone_seqs):
                        alignments.append(equal_align(frames, phones, phone_hmms))
                    if verbose:
                        print(f"  Pass 0 (equal alignment): distributing frames equally")
                else:
                    # Viterbi alignment with Kaldi scale factors
                    for frames, phones in zip(frames_list, all_phone_seqs):
                        alignments.append(viterbi_align(
                            frames, phones, phone_hmms, gmms,
                            acoustic_scale=1.0,
                            transition_scale=1.0,
                            self_loop_scale=1.0,
                        ))
                prev_alignments = alignments
                realign_msg = "realign" if level_idx > 0 or it > 0 else "equal"
            else:
                # No realignment this iteration: reuse previous alignment
                alignments = prev_alignments
                realign_msg = "no realign"

            if alignments is None:
                continue

            # Re-estimate GMMs from alignment
            gmms = reestimate_gmms(alignments, frames_list, gmms, n_comp)

            if verbose:
                ll = compute_total_log_likelihood(frames_list, alignments, gmms)
                print(f"  Iter {it + 1}/{iters_per_level} ({realign_msg}): log-likelihood = {ll:.1f}")

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
    from lexicon import DIGITS, DIGIT_WORDS
    transcripts = [" ".join(DIGIT_WORDS[d] for d in r["digit"].split()) for r in train_records]
    frames_list = [extract_mfcc(r["samples"], r["sample_rate"]) for r in train_records]

    # Filter out empty frames
    valid = [(t, f) for t, f in zip(transcripts, frames_list) if f.shape[0] > 0]
    if len(valid) < len(transcripts):
        print(f"Filtered out {len(transcripts) - len(valid)} empty utterances")
    transcripts, frames_list = zip(*valid) if valid else ([], [])
    transcripts = list(transcripts)
    frames_list = list(frames_list)

    # Train
    gmms = train(transcripts, frames_list, DIGITS, verbose=True)

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
