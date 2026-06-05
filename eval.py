"""
Word Error Rate evaluation (paper Section IX, questionnaire Q4).

Computes WER, CER, SER via Levenshtein edit distance, with bootstrap confidence intervals.

Matches the output format of Kaldi's compute-wer:
    %WER 4.3 [ 7 / 162, 0 ins, 2 del, 5 sub ]
    %SER 7.1 [ 3 / 42 ]
"""

import random
import math
import numpy as np
from typing import List, Tuple


def edit_distance(ref: List, hyp: List) -> Tuple[int, int, int, int]:
    """
    Levenshtein edit distance between two sequences.

    Returns (total_distance, insertions, deletions, substitutions).

    Uses the standard DP algorithm. Same recurrence as Viterbi:
        d[i][j] = min(d[i-1][j] + 1,    # deletion
                      d[i][j-1] + 1,    # insertion
                      d[i-1][j-1] + (ref[i-1] != hyp[j-1]))  # substitution
    """
    n = len(ref)
    m = len(hyp)

    # DP matrix
    d = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    # Traceback to count S, D, I
    i, j = n, m
    ins = del_ = sub = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            sub += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            del_ += 1
            i -= 1

    return d[n][m], ins, del_, sub


def wer(ref: List[str], hyp: List[str]) -> float:
    """Compute Word Error Rate."""
    dist, ins, del_, sub = edit_distance(ref, hyp)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return dist / len(ref)


def cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate."""
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    dist, ins, del_, sub = edit_distance(ref_chars, hyp_chars)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return dist / len(ref_chars)


def ser(refs: List[List[str]], hyps: List[List[str]]) -> float:
    """Compute Sentence Error Rate."""
    errors = 0
    for ref, hyp in zip(refs, hyps):
        if ref != hyp:
            errors += 1
    return errors / len(refs) if refs else 0.0


class WERStats:
    """Accumulate WER statistics across multiple utterances."""

    def __init__(self):
        self.total_words = 0
        self.total_errors = 0
        self.total_ins = 0
        self.total_del = 0
        self.total_sub = 0
        self.total_sentences = 0
        self.sent_errors = 0
        # Per-utterance stats for bootstrap
        self.utt_errors: List[int] = []
        self.utt_words: List[int] = []

    def add(self, ref: List[str], hyp: List[str]) -> None:
        """Add one utterance's results."""
        dist, ins, del_, sub = edit_distance(ref, hyp)
        n = len(ref)
        self.total_words += n
        self.total_errors += dist
        self.total_ins += ins
        self.total_del += del_
        self.total_sub += sub
        self.total_sentences += 1
        self.sent_errors += 1 if ref != hyp else 0
        self.utt_errors.append(dist)
        self.utt_words.append(n)

    @property
    def wer_pct(self) -> float:
        if self.total_words == 0:
            return 0.0
        return 100.0 * self.total_errors / self.total_words

    @property
    def ser_pct(self) -> float:
        if self.total_sentences == 0:
            return 0.0
        return 100.0 * self.sent_errors / self.total_sentences

    def bootstrap_ci(self, n_resamples: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for WER (paper Section VII, compute-wer-bootci).

        Resamples utterances with replacement and computes WER each time.
        Returns (lower_bound, upper_bound) for the (1-alpha) confidence interval.

        Args:
            n_resamples: number of bootstrap resamples.
            alpha: significance level (0.05 = 95% CI).

        Returns:
            (lower_pct, upper_pct) — WER as percentage.
        """
        if not self.utt_errors:
            return 0.0, 0.0

        n = len(self.utt_errors)
        wers = []
        for _ in range(n_resamples):
            total_err = 0
            total_n = 0
            for _ in range(n):
                idx = random.randrange(n)
                total_err += self.utt_errors[idx]
                total_n += self.utt_words[idx]
            if total_n > 0:
                wers.append(100.0 * total_err / total_n)

        wers.sort()
        lower_idx = int(n_resamples * alpha / 2)
        upper_idx = int(n_resamples * (1 - alpha / 2))
        lower_idx = max(0, min(lower_idx, len(wers) - 1))
        upper_idx = max(0, min(upper_idx, len(wers) - 1))

        return wers[lower_idx], wers[upper_idx]

    def report(self) -> str:
        """Return a formatted WER report matching compute-wer output."""
        ci_low, ci_high = self.bootstrap_ci()
        lines = [
            f"%WER {self.wer_pct:.1f} [ {self.total_errors} / {self.total_words}, "
            f"{self.total_ins} ins, {self.total_del} del, {self.total_sub} sub ]",
            f"%SER {self.ser_pct:.1f} [ {self.sent_errors} / {self.total_sentences} ]",
            f"95% CI: {ci_low:.1f}% -- {ci_high:.1f}%  (bootstrap, R=2000)",
        ]
        return "\n".join(lines)

    def sample_outputs(self, refs: List[List[str]], hyps: List[List[str]], n: int = 5) -> str:
        """Show n sample outputs (mix of correct and incorrect)."""
        lines = ["\n--- Sample outputs ---"]

        correct = [(r, h) for r, h in zip(refs, hyps) if r == h]
        incorrect = [(r, h) for r, h in zip(refs, hyps) if r != h]

        shown = 0
        for r, h in correct:
            if shown >= n // 2 + 1:
                break
            ref_str = " ".join(r)
            hyp_str = " ".join(h)
            lines.append(f'  OK ref: "{ref_str}"  | hyp: "{hyp_str}"')
            shown += 1

        for r, h in incorrect:
            if shown >= n:
                break
            ref_str = " ".join(r)
            hyp_str = " ".join(h)
            lines.append(f'  ER ref: "{ref_str}"  | hyp: "{hyp_str}"')
            shown += 1

        return "\n".join(lines)


if __name__ == "__main__":
    # Test
    stats = WERStats()
    stats.add(["three", "five", "seven"], ["three", "five", "seven"])  # correct
    stats.add(["one", "two"], ["one", "three"])                         # 1 sub
    stats.add(["zero"], ["zero", "one"])                                # 1 ins
    stats.add(["four", "five"], ["five"])                               # 1 del
    print(stats.report())
    print(stats.sample_outputs(
        [["three", "five", "seven"], ["one", "two"], ["zero"], ["four", "five"]],
        [["three", "five", "seven"], ["one", "three"], ["zero", "one"], ["five"]],
    ))
