"""
Language model for digit recognition (paper Section VI).

Simple n-gram model built from training transcripts.
Used to build the G FST for HCLG composition.

The model assigns probabilities to word sequences:
    P(word_i | word_{i-1}, word_{i-2}, ...)

We implement a bigram model (unigram fallback). Probabilities are stored
as negative log probabilities (costs) for integration with the WFST decoder.
"""

import math
from typing import Dict, Tuple, Optional
import numpy as np
from lexicon import WORDS, WORD_MAP, START_WORD, END_WORD, NUM_WORDS


class LanguageModel:
    """
    Bigram language model with unigram backoff.

    Attributes:
        unigrams: dict[word_id] -> log_prob
        bigrams: dict[(prev_id, word_id)] -> log_prob
        backoff: dict[prev_id] -> backoff_weight (log)
        order: 1 for unigram, 2 for bigram
    """

    def __init__(self, order: int = 2):
        self.order = order
        self.unigrams: Dict[int, float] = {}
        self.bigrams: Dict[Tuple[int, int], float] = {}
        self.backoff: Dict[int, float] = {}

    def unigram_prob(self, word_id: int) -> float:
        """Get the log-probability of a word (unigram)."""
        return self.unigrams.get(word_id, -float("inf"))

    def bigram_prob(self, prev_id: int, word_id: int) -> float:
        """Get the log-probability of word_id given prev_id."""
        key = (prev_id, word_id)
        if key in self.bigrams:
            return self.bigrams[key]
        # Backoff to unigram
        bo = self.backoff.get(prev_id, 0.0)
        return bo + self.unigram_prob(word_id)

    def score(self, prev_id: int, word_id: int) -> float:
        """Log-probability of word_id given prev_id."""
        if self.order >= 2:
            return self.bigram_prob(prev_id, word_id)
        return self.unigram_prob(word_id)

    def sentence_log_prob(self, word_ids: list) -> float:
        """Compute log-probability of a full sentence (with <s> and </s>)."""
        ids = [START_WORD] + word_ids + [END_WORD]
        logp = 0.0
        for i in range(1, len(ids)):
            logp += self.score(ids[i - 1], ids[i])
        return logp

    def perplexity(self, word_ids: list) -> float:
        """Compute perplexity of a sentence: 2^(-avg log_prob per word)."""
        logp = self.sentence_log_prob(word_ids)
        # Number of word predictions = len(word_ids) + 1 (the final </s>)
        n = len(word_ids) + 1
        if n == 0:
            return float("inf")
        avg_logp = logp / n
        return math.exp(-avg_logp)

    def __repr__(self) -> str:
        return f"LanguageModel(order={self.order}, unigrams={len(self.unigrams)}, bigrams={len(self.bigrams)})"


def train_lm(
    transcripts: list,
    order: int = 2,
    add_k: float = 1.0,
) -> LanguageModel:
    """
    Train an n-gram language model from a list of transcript texts.

    Uses add-k smoothing for unseen bigrams.

    Args:
        transcripts: list of strings, each a sentence like "three five seven"
        order: 1 (unigram) or 2 (bigram)
        add_k: smoothing constant for add-k smoothing

    Returns:
        LanguageModel with log-probabilities.
    """
    lm = LanguageModel(order=order)

    # Count occurrences
    unigram_counts: Dict[int, float] = {}
    bigram_counts: Dict[Tuple[int, int], float] = {}
    start_counts: Dict[int, float] = {}  # words at sentence start

    for trans in transcripts:
        words = trans.strip().lower().split()
        word_ids = [WORD_MAP[w] for w in words if w in WORD_MAP]

        # <s> first_word_id
        if word_ids:
            start_counts[word_ids[0]] = start_counts.get(word_ids[0], 0) + 1

        # Count unigrams and bigrams
        all_ids = [START_WORD] + word_ids + [END_WORD]
        for w in all_ids:
            unigram_counts[w] = unigram_counts.get(w, 0) + 1
        for i in range(1, len(all_ids)):
            key = (all_ids[i - 1], all_ids[i])
            bigram_counts[key] = bigram_counts.get(key, 0) + 1

    # Total unigram count
    total_unigrams = sum(unigram_counts.values())

    # Compute unigram probabilities (with smoothing)
    vocab_size = NUM_WORDS + 2  # words + <s> + </s>
    for w in range(NUM_WORDS + 2):
        cnt = unigram_counts.get(w, 0)
        lm.unigrams[w] = math.log((cnt + add_k) / (total_unigrams + add_k * vocab_size))

    # Compute bigram probabilities (with smoothing)
    if order >= 2:
        for prev_id in range(NUM_WORDS + 2):
            prev_count = unigram_counts.get(prev_id, 0)
            lm.backoff[prev_id] = math.log(
                prev_count / (prev_count + add_k * vocab_size)
                if prev_count > 0
                else 1.0
            )
            for word_id in range(NUM_WORDS + 2):
                cnt_prev_word = prev_count
                cnt = bigram_counts.get((prev_id, word_id), 0)
                if cnt_prev_word > 0:
                    prob = (cnt + add_k) / (cnt_prev_word + add_k * vocab_size)
                else:
                    prob = 1.0 / vocab_size
                lm.bigrams[(prev_id, word_id)] = math.log(prob)

    return lm


def build_g_fst(lm: LanguageModel) -> "FST":
    """
    Build the G WFST from a trained language model.

    The G FST encodes word-level transition probabilities.
    Input = output = word_id (identity mapping on each arc).
    Weight = negative log probability from the LM.

    Returns:
        FST for G, ready to be composed with L.
    """
    from fst import FST, Arc, EPS

    fst = FST()

    # One state per context word (plus start state)
    # State 0 = start (<s> context), state 1...N = word-specific states
    # Actually simpler: one big state per unique word context
    # For bigram, we need states for each possible previous word

    # Start state
    start = fst.add_state()
    fst.set_start(start)

    # We'll create a state per word for the bigram context
    # State for each word + start + end
    word_states = {}
    for w in range(NUM_WORDS):
        word_states[w] = fst.add_state()
        fst.set_final(word_states[w])  # allow ending at any word

    # Add arcs for start -> first word
    for word_id in range(NUM_WORDS):
        logp = lm.bigram_prob(START_WORD, word_id)
        fst.add_arc(start, Arc(word_states[word_id], word_id, word_id, -logp))

    # Add arcs for word -> word transitions
    for prev_id in range(NUM_WORDS):
        for word_id in range(NUM_WORDS):
            logp = lm.bigram_prob(prev_id, word_id)
            fst.add_arc(
                word_states[prev_id],
                Arc(word_states[word_id], word_id, word_id, -logp),
            )

    # Add arcs to end (sentence end)
    for prev_id in range(NUM_WORDS):
        logp = lm.bigram_prob(prev_id, END_WORD)
        fst.add_arc(
            word_states[prev_id],
            Arc(fst.add_state(), END_WORD, END_WORD, -logp),
        )

    # Set the last state as final
    final_state = fst.num_states - 1
    fst.set_final(final_state)

    return fst


if __name__ == "__main__":
    # Test LM with sample transcripts
    sample = ["one two three", "zero one two", "three four five", "six seven eight", "nine zero one"]
    lm = train_lm(sample)
    print(f"Trained {lm}")

    # Test probabilities
    for w in WORDS:
        wid = WORD_MAP[w]
        print(f"  P({w}) = {math.exp(lm.unigram_prob(wid)):.4f}")

    # Test bigram
    p1_wid = WORD_MAP["one"]
    p2_wid = WORD_MAP["two"]
    bp = lm.bigram_prob(p1_wid, p2_wid)
    print(f"\n  P(two | one) = {math.exp(bp):.4f} (log = {bp:.4f})")

    # Test perplexity
    test = [WORD_MAP[w] for w in "three four five".split()]
    pp = lm.perplexity(test)
    print(f"\n  Perplexity of 'three four five': {pp:.2f}")

    # Build G FST
    g = build_g_fst(lm)
    print(f"\n  G FST: {g.print_stats()}")
