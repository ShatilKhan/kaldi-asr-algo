"""
Language model (paper Section VI).

Simple n-gram model built from training transcripts.
Used to build the G FST for HCLG composition.

The model assigns probabilities to word sequences:
    P(word_i | word_{i-1}, word_{i-2}, ...)

We implement a bigram model (unigram fallback). Probabilities are stored
as negative log probabilities (costs) for integration with the WFST decoder.
"""

import math
from typing import Dict, Tuple

from lexicon import Lexicon


class LanguageModel:
    """
    Bigram language model with unigram backoff.

    Attributes:
        unigrams: dict[word_id] -> log_prob
        bigrams: dict[(prev_id, word_id)] -> log_prob
        backoff: dict[prev_id] -> backoff_weight (log)
        order: 1 for unigram, 2 for bigram
        num_words, start_word, end_word: vocabulary layout from the Lexicon
    """

    def __init__(self, order: int = 2, num_words: int = 0,
                 start_word: int = 0, end_word: int = 0):
        self.order = order
        self.num_words = num_words
        self.start_word = start_word
        self.end_word = end_word
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
        ids = [self.start_word] + word_ids + [self.end_word]
        logp = 0.0
        for i in range(1, len(ids)):
            logp += self.score(ids[i - 1], ids[i])
        return logp

    def perplexity(self, word_ids: list) -> float:
        """Compute perplexity of a sentence: exp(-avg log_prob per word)."""
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
    lex: Lexicon,
    order: int = 2,
    add_k: float = 1.0,
) -> LanguageModel:
    """
    Train an n-gram language model from a list of transcript texts.

    Uses add-k smoothing for unseen bigrams.

    Args:
        transcripts: list of strings, each a sentence like "three five seven"
        lex: the task Lexicon (provides word ids and sentence markers)
        order: 1 (unigram) or 2 (bigram)
        add_k: smoothing constant for add-k smoothing

    Returns:
        LanguageModel with log-probabilities.
    """
    lm = LanguageModel(order=order, num_words=lex.num_words,
                       start_word=lex.start_word, end_word=lex.end_word)

    # Count occurrences
    unigram_counts: Dict[int, float] = {}
    bigram_counts: Dict[Tuple[int, int], float] = {}

    for trans in transcripts:
        words = trans.strip().lower().split()
        word_ids = [lex.word_map[w] for w in words if w in lex.word_map]

        # Count unigrams and bigrams over <s> w1 ... wN </s>
        all_ids = [lex.start_word] + word_ids + [lex.end_word]
        for w in all_ids:
            unigram_counts[w] = unigram_counts.get(w, 0) + 1
        for i in range(1, len(all_ids)):
            key = (all_ids[i - 1], all_ids[i])
            bigram_counts[key] = bigram_counts.get(key, 0) + 1

    # Total unigram count
    total_unigrams = sum(unigram_counts.values())

    # Compute unigram probabilities (with smoothing)
    vocab_size = lex.num_words + 2  # words + <s> + </s>
    for w in range(vocab_size):
        cnt = unigram_counts.get(w, 0)
        lm.unigrams[w] = math.log((cnt + add_k) / (total_unigrams + add_k * vocab_size))

    # Compute bigram probabilities (with smoothing)
    if order >= 2:
        for prev_id in range(vocab_size):
            prev_count = unigram_counts.get(prev_id, 0)
            lm.backoff[prev_id] = math.log(
                prev_count / (prev_count + add_k * vocab_size)
                if prev_count > 0
                else 1.0
            )
            for word_id in range(vocab_size):
                cnt = bigram_counts.get((prev_id, word_id), 0)
                if prev_count > 0:
                    prob = (cnt + add_k) / (prev_count + add_k * vocab_size)
                else:
                    prob = 1.0 / vocab_size
                lm.bigrams[(prev_id, word_id)] = math.log(prob)

    return lm


if __name__ == "__main__":
    from lexicon import DIGITS

    sample = ["one two three", "zero one two", "three four five", "six seven eight", "nine zero one"]
    lm = train_lm(sample, DIGITS)
    print(f"Trained {lm}")

    for w in DIGITS.words:
        wid = DIGITS.word_map[w]
        print(f"  P({w}) = {math.exp(lm.unigram_prob(wid)):.4f}")

    p1 = DIGITS.word_map["one"]
    p2 = DIGITS.word_map["two"]
    bp = lm.bigram_prob(p1, p2)
    print(f"\n  P(two | one) = {math.exp(bp):.4f} (log = {bp:.4f})")

    test = [DIGITS.word_map[w] for w in "three four five".split()]
    print(f"\n  Perplexity of 'three four five': {lm.perplexity(test):.2f}")
