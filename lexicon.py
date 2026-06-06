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

    @classmethod
    def from_prons(cls, prons: Dict[str, List[str]], sil: str = "SIL") -> "Lexicon":
        """Build a Lexicon deriving the phone set from the pronunciations."""
        phones = sorted({p for seq in prons.values() for p in seq})
        phone_map = {sil: 0}
        for i, p in enumerate(phones, start=1):
            phone_map[p] = i
        return cls(phone_map, prons, sil=sil)


def load_pronunciation_lexicon(path: str, vocab=None, sil: str = "SIL") -> "Lexicon":
    """
    Build a Lexicon from a pronunciation dictionary file (e.g. the
    LibriSpeech lexicon, OpenSLR SLR11: "WORD  P1 P2 P3" per line).

    Stress digits on ARPAbet phones (AH0, EH1, ...) are stripped so AH0 and
    AH1 share a phone, keeping the phone set small enough for a monophone toy.
    Only the first pronunciation of each word is kept. If vocab is given, the
    lexicon is restricted to those words.
    """
    vocab = set(w.lower() for w in vocab) if vocab is not None else None
    prons: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            if vocab is not None and word not in vocab:
                continue
            if word in prons:
                continue  # keep first pronunciation only
            phones = ["".join(c for c in p if not c.isdigit()) for p in parts[1:]]
            prons[word] = phones
    return Lexicon.from_prons(prons, sil=sil)


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


# ---------------------------------------------------------------------------
# Task: Google Speech Commands v0.02 (35 words, isolated word recognition)
# ---------------------------------------------------------------------------
# Pronunciations from CMUdict with stress markers stripped.

_COMMANDS_PRONS = {
    "yes":      ["Y", "EH", "S"],
    "no":       ["N", "OW"],
    "up":       ["AH", "P"],
    "down":     ["D", "AW", "N"],
    "left":     ["L", "EH", "F", "T"],
    "right":    ["R", "AY", "T"],
    "on":       ["AA", "N"],
    "off":      ["AO", "F"],
    "stop":     ["S", "T", "AA", "P"],
    "go":       ["G", "OW"],
    "zero":     ["Z", "IY", "R", "OW"],
    "one":      ["W", "AH", "N"],
    "two":      ["T", "UW"],
    "three":    ["TH", "R", "IY"],
    "four":     ["F", "AO", "R"],
    "five":     ["F", "AY", "V"],
    "six":      ["S", "IH", "K", "S"],
    "seven":    ["S", "EH", "V", "AH", "N"],
    "eight":    ["EY", "T"],
    "nine":     ["N", "AY", "N"],
    "backward": ["B", "AE", "K", "W", "ER", "D"],
    "bed":      ["B", "EH", "D"],
    "bird":     ["B", "ER", "D"],
    "cat":      ["K", "AE", "T"],
    "dog":      ["D", "AO", "G"],
    "follow":   ["F", "AA", "L", "OW"],
    "forward":  ["F", "AO", "R", "W", "ER", "D"],
    "happy":    ["HH", "AE", "P", "IY"],
    "house":    ["HH", "AW", "S"],
    "learn":    ["L", "ER", "N"],
    "marvin":   ["M", "AA", "R", "V", "IH", "N"],
    "sheila":   ["SH", "IY", "L", "AH"],
    "tree":     ["T", "R", "IY"],
    "visual":   ["V", "IH", "ZH", "AH", "W", "AH", "L"],
    "wow":      ["W", "AW"],
}

SPEECH_COMMANDS = Lexicon.from_prons(_COMMANDS_PRONS)


if __name__ == "__main__":
    for name, lex in [("DIGITS", DIGITS), ("YESNO", YESNO), ("SPEECH_COMMANDS", SPEECH_COMMANDS)]:
        print(f"{name}: {lex}")
        for word in lex.words:
            phones = [lex.phone_ids[p] for p in lex.lexicon[word]]
            print(f"  {word:8s} ({lex.word_map[word]}): {' '.join(phones)}")
        print(f"  <s>={lex.start_word} </s>={lex.end_word}")
