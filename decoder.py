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

from collections import deque

import numpy as np
from typing import List
from fst import FST, Arc, EPS, compose
from lexicon import Lexicon
from gmm import DiagGmm


def build_h_fst(phone_hmms: list) -> FST:
    """
    Build the H FST (HMM topology, paper Section IV-C).

    Input labels = pdf-ids (GMM indices). Output labels = transition-ids.

    Each HMM state emits its transition-id exactly ONCE, on the arc that
    leaves the state (forward/exit). Self-loops consume a frame but output
    epsilon. This mirrors Kaldi's add-self-loops-after-composition design:
    if self-loops also emitted the tid, composing with L would only accept
    phones that last exactly 3 frames, since L expects each tid once.

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
        # Create 3 HMM states for this phone
        hmm_states = [h.add_state() for _ in phmm["states"]]

        # Epsilon from loop-back to first HMM state of this phone
        h.add_arc(loop_state, Arc(hmm_states[0], EPS, EPS, 0.0))
        # Also epsilon from start to this phone (for sentence start)
        h.add_arc(start, Arc(hmm_states[0], EPS, EPS, 0.0))

        # Add arcs for each HMM state
        for i, s in enumerate(phmm["states"]):
            hs = hmm_states[i]
            pdf_id = s["pdf_id"]
            tid = s["pdf_id"]  # use pdf-id as transition-id (unique per state)

            # Self-loop: consume a frame, emit nothing
            h.add_arc(hs, Arc(hs, pdf_id, EPS, -s["self_loop_logp"]))

            # Leaving the state emits its tid exactly once
            if i < 2:
                next_hs = hmm_states[i + 1]
                h.add_arc(hs, Arc(next_hs, pdf_id, tid, -s["forward_logp"]))
            else:
                # From 3rd state, exit to loop-back state
                h.add_arc(hs, Arc(loop_state, pdf_id, tid, -phmm["exit_logp"]))

    return h


def build_l_fst(lex: Lexicon) -> FST:
    """
    Build the L FST (lexicon, paper Section V).

    L maps transition-id sequences to words, as a closed loop so it accepts
    any number of words per utterance:

        start ──(word tids, word olabel on last arc)──▶ back to start
        start ──(SIL tids, all eps)──▶ back to start   (optional silence)

    The start state is final, so the utterance can end after any word or
    silence. Optional silence at the start, between words, and at the end
    falls out of the loop structure for free.
    """
    l = FST()
    start = l.add_state()
    l.set_start(start)
    l.set_final(start)

    def add_path(tids: List[int], olabel_last: int):
        """Add a tid path from start back to start."""
        prev = start
        for i, tid in enumerate(tids):
            last = i == len(tids) - 1
            cur = start if last else l.add_state()
            olabel = olabel_last if last else EPS
            l.add_arc(prev, Arc(cur, tid, olabel, 0.0))
            prev = cur

    # Optional silence loop (epsilon output)
    sil = lex.sil_phone
    add_path([sil * 3, sil * 3 + 1, sil * 3 + 2], EPS)

    # One loop per word; the word id comes out on the final arc
    for word in lex.words:
        word_id = lex.word_map[word]
        seq = []
        for phone_id in lex.lexicon[word]:
            seq.extend([phone_id * 3, phone_id * 3 + 1, phone_id * 3 + 2])
        add_path(seq, word_id)

    return l


def build_g_fst(lm, lex: Lexicon, word_penalty: float = 0.0) -> FST:
    """
    Build the G FST (language model, paper Section VI).

    G encodes word-to-word transition probabilities.
    Input = word_id, Output = word_id (identity mapping).

    word_penalty is a fixed extra cost per word arc (the classic word
    insertion penalty): without it, short spurious words are nearly free
    and the decoder happily inserts them over noisy frames.
    """
    g = FST()
    start = g.add_state()
    g.set_start(start)
    # Empty utterance = <s> followed directly by </s>
    g.set_final(start, -lm.bigram_prob(lex.start_word, lex.end_word))

    # One state per word + start state. Ending after word w costs the LM's
    # P(</s> | w), exactly like Kaldi bakes the sentence-end probability
    # into G's final weights.
    word_states = {}
    for w in range(lex.num_words):
        word_states[w] = g.add_state()
        g.set_final(word_states[w], -lm.bigram_prob(w, lex.end_word))

    # start -> first word
    for word_id in range(lex.num_words):
        logp = lm.bigram_prob(lex.start_word, word_id)
        g.add_arc(start, Arc(word_states[word_id], word_id, word_id, -logp + word_penalty))

    # word -> word (bigram transitions)
    for prev_id in range(lex.num_words):
        for word_id in range(lex.num_words):
            logp = lm.bigram_prob(prev_id, word_id)
            g.add_arc(word_states[prev_id],
                      Arc(word_states[word_id], word_id, word_id, -logp + word_penalty))

    return g


def _prune(tokens: dict, beam: float) -> dict:
    """Drop tokens whose cost is worse than (best cost + beam)."""
    if beam == float("inf") or not tokens:
        return tokens
    best = min(s for s, _ in tokens.values())
    cutoff = best + beam
    return {st: v for st, v in tokens.items() if v[0] <= cutoff}


def build_unigram_g(lm, lex: Lexicon, word_penalty: float = 0.0,
                    include_lm: bool = True) -> FST:
    """
    Build a single-state word-loop G.

    For isolated-word tasks a bigram adds no context (one word per utterance)
    but its V states multiply the H∘L state count during composition (the
    epsilon filter copies every H∘L state per G state), blowing HCLG up by V×.
    A one-state loop keeps HCLG ≈ H∘L.

    include_lm=True bakes the unigram log-prob into each word arc. With
    include_lm=False the arc carries only the word penalty, and the language
    model is applied on the fly in the decoder (so a bigram can be used
    without the V× determinization blowup of a multi-state G).
    """
    g = FST()
    s = g.add_state()
    g.set_start(s)
    g.set_final(s, 0.0)
    for w in range(lex.num_words):
        weight = word_penalty + (-lm.unigram_prob(w) if include_lm else 0.0)
        g.add_arc(s, Arc(s, w, w, weight))
    return g


def decode(
    frames: np.ndarray,
    gmms: List[DiagGmm],
    hclg: FST,
    num_words: int,
    acoustic_scale: float = 0.0833,
    beam: float = float("inf"),
    lm=None,
) -> List[int]:
    """
    Decode an utterance: Viterbi token-passing on HCLG (paper Section VIII).

    Fast token passing: a worklist epsilon-closure (no O(n^2) re-scan) and
    backpointers (no per-token list copies).

    Args:
        frames: (num_frames, D) MFCC feature matrix.
        gmms: list of DiagGmm, one per pdf-id.
        hclg: composed HCLG FST.
        num_words: vocabulary size (olabels >= num_words are not words).
        acoustic_scale: weight on acoustic scores vs graph scores.
        beam: prune tokens with cost > best + beam each frame (inf = no prune).
        lm: if given, apply its bigram cost on the fly when a word is emitted,
            keying tokens by (state, last_word). Lets a compact unigram-shaped
            graph carry real bigram context. None = use the graph weights as-is.

    Returns:
        List of word IDs (the hypothesized word sequence).
    """
    num_frames = frames.shape[0]
    num_pdfs = len(gmms)
    if num_frames == 0 or num_pdfs == 0:
        return []

    scores = DiagGmm.score_batch_all(gmms, frames)  # (num_frames, num_pdfs)
    arcs = hclg.arcs
    use_lm = lm is not None
    start_word = lm.start_word if use_lm else -1
    end_word = lm.end_word if use_lm else -1

    # Backpointers: a new entry is created only when a word is emitted, so the
    # table stays small. bp index -1 is the root (empty history).
    bp_prev = []
    bp_word = []

    def emit_bp(prev_bp, word):
        bp_prev.append(prev_bp)
        bp_word.append(word)
        return len(bp_prev) - 1

    def state_of(key):
        return key[0] if use_lm else key

    def lastword_of(key):
        return key[1] if use_lm else None

    def close(tokens):
        """Relax all epsilon-input arcs to a fixed point via a worklist."""
        work = deque(tokens.keys())
        while work:
            key = work.popleft()
            cost, bp = tokens[key]
            st, lw = state_of(key), lastword_of(key)
            for arc in arcs[st]:
                if arc.ilabel != EPS:
                    continue
                w = arc.olabel if (arc.olabel != EPS and arc.olabel < num_words) else None
                ncost = cost + arc.weight
                nlw, nbp = lw, bp
                if w is not None:
                    if use_lm:
                        ncost += -lm.bigram_prob(lw, w)
                        nlw = w
                    nbp = emit_bp(bp, w)
                nkey = (arc.next_state, nlw) if use_lm else arc.next_state
                if nkey not in tokens or ncost < tokens[nkey][0]:
                    tokens[nkey] = (ncost, nbp)
                    work.append(nkey)
        return tokens

    start_key = (hclg.start_state, start_word) if use_lm else hclg.start_state
    tokens = close({start_key: (0.0, -1)})

    for t in range(num_frames):
        row = scores[t]
        new_tokens = {}
        for key, (cost, bp) in tokens.items():
            st, lw = state_of(key), lastword_of(key)
            for arc in arcs[st]:
                il = arc.ilabel
                if not (0 <= il < num_pdfs):
                    continue
                ncost = cost + arc.weight - row[il] * acoustic_scale
                w = arc.olabel if (arc.olabel != EPS and arc.olabel < num_words) else None
                nlw, nbp = lw, bp
                if w is not None:
                    if use_lm:
                        ncost += -lm.bigram_prob(lw, w)
                        nlw = w
                    nbp = emit_bp(bp, w)
                nkey = (arc.next_state, nlw) if use_lm else arc.next_state
                cur = new_tokens.get(nkey)
                if cur is None or ncost < cur[0]:
                    new_tokens[nkey] = (ncost, nbp)

        tokens = _prune(close(new_tokens), beam)
        if not tokens:
            return []

    # --- Pick best final path ---
    best_cost = float("inf")
    best_bp = -1
    for key, (cost, bp) in tokens.items():
        st = state_of(key)
        if st in hclg.final_states:
            fc = cost + hclg.final_weights.get(st, 0.0)
            if use_lm:
                fc += -lm.bigram_prob(lastword_of(key), end_word)
            if fc < best_cost:
                best_cost, best_bp = fc, bp

    if best_cost == float("inf") and tokens:  # no final reached: best effort
        key = min(tokens, key=lambda k: tokens[k][0])
        best_bp = tokens[key][1]

    # Reconstruct the word sequence from backpointers.
    words = []
    i = best_bp
    while i != -1:
        words.append(bp_word[i])
        i = bp_prev[i]
    words.reverse()
    return words


def assemble_hclg(phone_hmms: list, lm, lex: Lexicon,
                  word_penalty: float = 0.0, g_mode: str = "bigram") -> FST:
    """
    Build the full HCLG = H ∘ L ∘ G decoder graph (paper Section VII).

    g_mode "bigram" uses the dense bigram G (fine for small vocab: yesno,
    digits); "unigram" uses a one-state unigram G to keep HCLG compact for
    larger isolated-word vocab (speech commands).
    """
    print("  Building H FST...")
    h = build_h_fst(phone_hmms)
    print(f"    H: {h.print_stats()}")

    print("  Building L FST...")
    l = build_l_fst(lex)
    print(f"    L: {l.print_stats()}")

    print(f"  Building G FST ({g_mode})...")
    if g_mode == "wordloop":
        # Penalty-only word loop; the LM is applied on the fly in the decoder
        # (lets a bigram run without the V× determinization blowup).
        g = build_unigram_g(lm, lex, word_penalty=word_penalty, include_lm=False)
    elif g_mode == "unigram":
        g = build_unigram_g(lm, lex, word_penalty=word_penalty)
    else:
        g = build_g_fst(lm, lex, word_penalty=word_penalty)
    print(f"    G: {g.print_stats()}")

    print("  Composing H ∘ L...")
    hl = compose(h, l)
    print(f"    H∘L: {hl.print_stats()}")

    print("  Composing (H∘L) ∘ G...")
    hclg = compose(hl, g)
    print(f"    HCLG: {hclg.print_stats()}")

    return hclg


if __name__ == "__main__":
    from lexicon import YESNO
    from lm import train_lm
    from hmm import build_all_phone_hmms, total_pdfs

    lex = YESNO
    phone_hmms = build_all_phone_hmms(lex.num_phones)
    print(f"Phone HMMs: {len(phone_hmms)} phones, {total_pdfs(phone_hmms)} pdf-ids")

    lm = train_lm(["yes no yes yes no", "no no yes no"], lex)
    print(f"LM: {lm}")

    hclg = assemble_hclg(phone_hmms, lm, lex)
    print(f"\nComposed HCLG: {hclg.print_stats()}")
