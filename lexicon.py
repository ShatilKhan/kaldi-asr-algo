"""
Pronunciation lexicons (paper Section V-L).

A Lexicon bundles the phone set and the word -> phone-sequence dictionary
for one recognition task. The decoder and trainer take a Lexicon instance,
so the same pipeline runs on different datasets (digits, yesno, ...).
"""

from typing import Dict, List


class Lexicon:
    """Phone set + pronunciation dictionary for one task."""

    def __init__(self, phone_map: Dict[str, int], prons: Dict[str, List[str]], sil: str = "SIL"):
        """
        Args:
            phone_map: phone symbol -> phone id (must include `sil`).
            prons: word -> list of phone symbols.
            sil: name of the silence phone.
        """
        self.phone_map = dict(phone_map)
        self.phone_ids = {v: k for k, v in self.phone_map.items()}
        self.num_phones = len(self.phone_map)
        self.sil_phone = self.phone_map[sil]

        # word -> list of phone ids
        self.lexicon: Dict[str, List[int]] = {
            word: [self.phone_map[p] for p in phones] for word, phones in prons.items()
        }

        self.words = sorted(self.lexicon.keys())
        self.word_map = {w: i for i, w in enumerate(self.words)}
        self.word_ids = {i: w for w, i in self.word_map.items()}
        self.num_words = len(self.words)

        # LM sentence markers sit just past the real word ids
        self.start_word = self.num_words      # <s>
        self.end_word = self.num_words + 1    # </s>

    def phones_for(self, word: str) -> List[int]:
        return self.lexicon[word]

    def __repr__(self) -> str:
        return f"Lexicon({self.num_words} words, {self.num_phones} phones)"


# ---------------------------------------------------------------------------
# Task: English digits (FSDD)
# ---------------------------------------------------------------------------

_DIGIT_PHONES = {
    "SIL": 0,
    "Z": 1, "IY": 2, "R": 3, "OW": 4, "W": 5, "AH": 6, "N": 7,
    "T": 8, "UW": 9, "TH": 10, "F": 11, "AO": 12, "AY": 13, "V": 14,
    "S": 15, "IH": 16, "K": 17, "EH": 18, "EY": 19, "AYT": 20, "AYN": 21,
}

_DIGIT_PRONS = {
    "zero":  ["Z", "IY", "R", "OW"],
    "one":   ["W", "AH", "N"],
    "two":   ["T", "UW"],
    "three": ["TH", "R", "IY"],
    "four":  ["F", "AO", "R"],
    "five":  ["F", "AY", "V"],
    "six":   ["S", "IH", "K", "S"],
    "seven": ["S", "EH", "V", "AH", "N"],
    "eight": ["EY", "T"],
    "nine":  ["N", "AY", "N"],
}

DIGITS = Lexicon(_DIGIT_PHONES, _DIGIT_PRONS)

# FSDD transcripts use digit characters; map them to lexicon words.
DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


# ---------------------------------------------------------------------------
# Task: YesNo (Kaldi egs/yesno)
# ---------------------------------------------------------------------------
# Same design as Kaldi's yesno recipe: one phone per word plus silence.
# The audio is Hebrew ken/lo; transcripts use yes/no.

_YESNO_PHONES = {"SIL": 0, "Y": 1, "N": 2}

_YESNO_PRONS = {
    "yes": ["Y"],
    "no":  ["N"],
}

YESNO = Lexicon(_YESNO_PHONES, _YESNO_PRONS)


if __name__ == "__main__":
    for name, lex in [("DIGITS", DIGITS), ("YESNO", YESNO)]:
        print(f"{name}: {lex}")
        for word in lex.words:
            phones = [lex.phone_ids[p] for p in lex.lexicon[word]]
            print(f"  {word:8s} ({lex.word_map[word]}): {' '.join(phones)}")
        print(f"  <s>={lex.start_word} </s>={lex.end_word}")
