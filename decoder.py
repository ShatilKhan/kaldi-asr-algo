"""
Decoder: HCLG graph assembly and Viterbi search (paper Sections VII, VIII).

Assembles the composed HCLG = H ∘ L ∘ G graph and runs token-passing
Viterbi decoding to find the most likely word sequence.

The decoding process:
  1. Build H (HMM topology) as an FST
  2. Build L (lexicon) as an FST
  3. Build G (language model) as an FST
  4. Compose: HL = H ∘ L, then HCLG = HL ∘ G
  5. For each audio utterance:
     a. Score each frame against all GMMs → score table (num_frames × num_pdfs)
     b. Run token-passing Viterbi on HCLG
     c. Extract word sequence from best path
"""

import numpy as np
from typing import List, Tuple, Optional
from fst import FST, Arc, EPS, compose, best_path
from hmm import build_all_phone_hmms, total_pdfs
from lexicon import (
    NUM_PHONES, WORDS, WORD_MAP, PHONE_IDS,
    LEXICON, SIL_PHONE, START_WORD, END_WORD, WORD_IDS,
)
from gmm import DiagGmm


def build_h_fst(phone_hmms: list) -> FST:
    """
    Build the H FST (HMM topology, paper Section IV-C).

    H models the 3-state left-to-right HMM for each phone.
    Input labels = pdf-ids (GMM indices).
    Output labels = transition-ids (unique per HMM state, used for composing with C/L).

    Each phone's 3 HMM states have transition-ids that are used in sequence
    by the L FST. L reads sequences of transition-ids and maps them to words.

    Args:
        phone_hmms: list of phone HMM dicts from hmm.build_all_phone_hmms()

    Returns:
        H FST.
    """
    h = FST()
    start = h.add_state()
    h.set_start(start)

    # Shared "loop-back" state: after finishing a phone, go here to start any phone
    loop_state = h.add_state()
    h.set_final(loop_state)  # allow ending here too

    for phmm in phone_hmms:
        pid = phmm["phone_id"]

        # Create 3 HMM states for this phone
        hmm_states = []
        for i, s in enumerate(phmm["states"]):
            hs = h.add_state()
            hmm_states.append(hs)

        # Epsilon from loop-back to first HMM state of this phone
        h.add_arc(loop_state, Arc(hmm_states[0], EPS, EPS, 0.0))
        # Also epsilon from start to this phone (for sentence start)
        h.add_arc(start, Arc(hmm_states[0], EPS, EPS, 0.0))

        # Add arcs for each HMM state
        for i, s in enumerate(phmm["states"]):
            hs = hmm_states[i]
            pdf_id = s["pdf_id"]
            tid = s["pdf_id"]  # Use pdf-id as transition-id (unique per state)

            # Self-loop: stay in this state
            h.add_arc(hs, Arc(hs, pdf_id, tid, -s["self_loop_logp"]))

            # Forward: go to next state
            if i < 2:
                next_hs = hmm_states[i + 1]
                h.add_arc(hs, Arc(next_hs, pdf_id, tid, -s["forward_logp"]))
            else:
                # From 3rd state, exit to loop-back state
                h.add_arc(hs, Arc(loop_state, pdf_id, tid, -phmm["exit_logp"]))

    return h


def build_l_fst() -> FST:
    """
    Build the L FST (lexicon, paper Section V).

    L maps transition-id sequences to words.
    Each word is modeled as: ε:ε → tid1:ε → tid2:ε → ... → tidN:word → final

    H now outputs transition-ids (same as pdf-ids), and L consumes them.
    The output is the word ID on the final arc of the phone sequence.
    """
    l = FST()
    start = l.add_state()
    l.set_start(start)

    for word in WORDS:
        word_id = WORD_MAP[word]
        phones = LEXICON[word]

        # Build transition-id sequence for this word: 3 tids per phone
        # Each phone has 3 states with pdf-ids = phone_offset*3 + 0/1/2
        seq = []
        for phone_id in phones:
            seq.extend([phone_id * 3, phone_id * 3 + 1, phone_id * 3 + 2])

        # Entry state for this word
        entry = l.add_state()
        l.add_arc(start, Arc(entry, EPS, EPS, 0.0))

        prev = entry
        for i, tid in enumerate(seq):
            cur = l.add_state()
            olabel = word_id if i == len(seq) - 1 else EPS
            l.add_arc(prev, Arc(cur, tid, olabel, 0.0))
            prev = cur

        # Final state for this pronunciation
        l.set_final(prev)

    return l


def build_g_fst(lm) -> FST:
    """
    Build the G FST (language model, paper Section VI).

    G encodes word-to-word transition probabilities.
    Input = word_id, Output = word_id (identity mapping).

    Args:
        lm: LanguageModel from lm.py.

    Returns:
        G FST.
    """
    g = FST()
    start = g.add_state()
    g.set_start(start)

    # One state per word + start state
    word_states = {}
    for w in range(len(WORDS)):
        word_states[w] = g.add_state()
        g.set_final(word_states[w])

    # start -> first word (unigram)
    for word_id in range(len(WORDS)):
        logp = lm.bigram_prob(START_WORD, word_id)
        w = -logp  # convert to cost
        g.add_arc(start, Arc(word_states[word_id], word_id, word_id, w))

    # word -> word (bigram transitions)
    for prev_id in range(len(WORDS)):
        for word_id in range(len(WORDS)):
            logp = lm.bigram_prob(prev_id, word_id)
            w = -logp
            g.add_arc(word_states[prev_id], Arc(word_states[word_id], word_id, word_id, w))

    return g


def decode(
    frames: np.ndarray,
    gmms: List[DiagGmm],
    hclg: FST,
    beam: float = float("inf"),
) -> List[int]:
    """
    Decode an utterance: Viterbi token-passing on HCLG (paper Section VIII).

    Args:
        frames: (num_frames, D) MFCC feature matrix.
        gmms: list of DiagGmm, one per pdf-id.
        hclg: composed HCLG FST.
        beam: beam width for pruning (default: no pruning).

    Returns:
        List of word IDs (the hypothesized word sequence).
    """
    num_frames = frames.shape[0]
    num_pdfs = len(gmms)

    if num_frames == 0 or num_pdfs == 0:
        return []

    # --- Step 1: Score all frames against all GMMs ---
    # scores[t][p] = log-likelihood of frame t under pdf-id p
    scores = np.zeros((num_frames, num_pdfs))
    for p, gmm in enumerate(gmms):
        for t in range(num_frames):
            scores[t, p] = gmm.log_likelihood(frames[t])

    # --- Step 2: Token-passing Viterbi ---
    # Token: (cumulative_score, history_of_olabels)
    # We track best token per HCLG state
    tokens = {hclg.start_state: (0.0, [])}
    best_score_at_t = 0.0

    for t in range(num_frames):
        new_tokens = {}

        for state, (score, out_seq) in tokens.items():
            for arc in hclg.arcs[state]:
                # GMM score contribution depends on whether this is an epsilon arc
                if arc.ilabel == EPS:
                    gmm_score = 0.0
                elif 0 <= arc.ilabel < num_pdfs:
                    gmm_score = -scores[t, arc.ilabel]  # negative because lower=cost
                else:
                    continue  # invalid pdf-id

                new_score = score + arc.weight + gmm_score
                new_out = out_seq + ([arc.olabel] if arc.olabel != EPS and arc.olabel < len(WORDS) else [])

                if arc.next_state not in new_tokens or new_score < new_tokens[arc.next_state][0]:
                    new_tokens[arc.next_state] = (new_score, new_out)

        # --- Prune (if beam is set) ---
        if beam < float("inf") and new_tokens:
            best_local = min(s for s, _ in new_tokens.values())
            new_tokens = {
                s: (sc, seq)
                for s, (sc, seq) in new_tokens.items()
                if sc <= best_local + beam
            }

        # --- ε-closure after matching this frame ---
        # Propagate tokens on epsilon arcs
        changed = True
        while changed:
            changed = False
            for state, (score, out_seq) in list(new_tokens.items()):
                for arc in hclg.arcs[state]:
                    if arc.ilabel == EPS:
                        new_score = score + arc.weight
                        new_out = out_seq + ([arc.olabel] if arc.olabel != EPS and arc.olabel < len(WORDS) else [])
                        if arc.next_state not in new_tokens or new_score < new_tokens[arc.next_state][0]:
                            new_tokens[arc.next_state] = (new_score, new_out)
                            changed = True

        tokens = new_tokens
        if not tokens:
            return []

    # --- Step 3: Pick best final path ---
    best_score = float("inf")
    best_seq = []
    for state, (score, out_seq) in tokens.items():
        if state in hclg.final_states:
            final_score = score + hclg.final_weights.get(state, 0.0)
            if final_score < best_score:
                best_score = final_score
                best_seq = out_seq

    # Fallback: no final state reached, take best non-final
    if best_score == float("inf") and tokens:
        best_state = min(tokens.keys(), key=lambda s: tokens[s][0])
        best_seq = tokens[best_state][1]

    return best_seq


def assemble_hclg(phone_hmms: list, lm) -> FST:
    """
    Build the full HCLG = H ∘ L ∘ G decoder graph (paper Section VII).

    Args:
        phone_hmms: list of phone HMM dicts.
        lm: LanguageModel from lm.py.

    Returns:
        Composed HCLG FST.
    """
    from fst import compose

    print("  Building H FST...")
    h = build_h_fst(phone_hmms)
    print(f"    H: {h.print_stats()}")

    print("  Building L FST...")
    l = build_l_fst()
    print(f"    L: {l.print_stats()}")

    print("  Building G FST...")
    g = build_g_fst(lm)
    print(f"    G: {g.print_stats()}")

    print("  Composing H ∘ L...")
    hl = compose(h, l)
    print(f"    H∘L: {hl.print_stats()}")

    print("  Composing (H∘L) ∘ G...")
    hclg = compose(hl, g)
    print(f"    HCLG: {hclg.print_stats()}")

    return hclg


if __name__ == "__main__":
    from lexicon import WORDS, LEXICON, PHONE_IDS
    from lm import train_lm

    # Build phone HMMs
    num_phones = 22  # from lexicon
    phone_hmms = build_all_phone_hmms(num_phones)
    npdfs = total_pdfs(phone_hmms)
    print(f"Phone HMMs: {len(phone_hmms)} phones, {npdfs} pdf-ids")

    # Train a simple LM
    sample_transcripts = [
        "zero one two three four five six seven eight nine",
        "one two three four five six seven eight nine zero",
        "two three four five six seven eight nine zero one",
    ]
    lm = train_lm(sample_transcripts)
    print(f"LM: {lm}")

    # Test H FST
    h = build_h_fst(phone_hmms)
    print(f"\nH FST: {h.print_stats()}")

    # Test L FST
    l = build_l_fst()
    print(f"L FST: {l.print_stats()}")

    # Test G FST
    g = build_g_fst(lm)
    print(f"G FST: {g.print_stats()}")

    # Compose and test
    print("\n--- Composing HCLG ---")
    hclg = assemble_hclg(phone_hmms, lm)

    # Check HCLG sanity
    print(f"\nComposed HCLG: {hclg.print_stats()}")
