"""
Pronunciation dictionary (lexicon) for English digits (paper Section V-L).

Maps words to phone sequences. Used to build the L FST for HCLG composition.

Phone set covers the sounds needed for digits 0–9 and silence.
"""

# Phone set for English digits
# Each phone gets a unique integer ID.
PHONE_MAP = {
    # Silence
    "SIL": 0,
    # Vowels
    "Z": 1,   # z (zero)
    "IY": 2,  # ee (zero, three)
    "R": 3,   # r (zero, three, four)
    "OW": 4,  # oh (zero, four)
    "W": 5,   # w (one)
    "AH": 6,  # uh (one)
    "N": 7,   # n (one, nine, seven)
    "T": 8,   # t (two)
    "UW": 9,  # oo (two)
    "TH": 10, # th (three)
    "F": 11,  # f (four, five)
    "AO": 12, # aw (four)
    "AY": 13, # eye (five, nine)
    "V": 14,  # v (five, seven)
    "S": 15,  # s (six, seven)
    "IH": 16, # ih (six)
    "K": 17,  # k (six)
    "EH": 18, # eh (seven)
    "EY": 19, # ay (eight)
    "AYT": 20,# eight / ended t
    "AYN": 21,# nine / ended n
}

# Inverse mapping: phone_id → phone_symbol
PHONE_IDS = {v: k for k, v in PHONE_MAP.items()}

# Number of phones
NUM_PHONES = len(PHONE_MAP)

# Silence phone ID
SIL_PHONE = PHONE_MAP["SIL"]

# Pronunciation dictionary: word → list of phone IDs
# Using a simple phonemic transcription for each digit.
LEXICON = {
    "zero":    [PHONE_MAP["Z"], PHONE_MAP["IY"], PHONE_MAP["R"], PHONE_MAP["OW"]],
    "one":     [PHONE_MAP["W"], PHONE_MAP["AH"], PHONE_MAP["N"]],
    "two":     [PHONE_MAP["T"], PHONE_MAP["UW"]],
    "three":   [PHONE_MAP["TH"], PHONE_MAP["R"], PHONE_MAP["IY"]],
    "four":    [PHONE_MAP["F"], PHONE_MAP["AO"], PHONE_MAP["R"]],
    "five":    [PHONE_MAP["F"], PHONE_MAP["AY"], PHONE_MAP["V"]],
    "six":     [PHONE_MAP["S"], PHONE_MAP["IH"], PHONE_MAP["K"], PHONE_MAP["S"]],
    "seven":   [PHONE_MAP["S"], PHONE_MAP["EH"], PHONE_MAP["V"], PHONE_MAP["AH"], PHONE_MAP["N"]],
    "eight":   [PHONE_MAP["EY"], PHONE_MAP["T"]],
    "nine":    [PHONE_MAP["N"], PHONE_MAP["AY"], PHONE_MAP["N"]],
}

# All digit words in sorted order
WORDS = sorted(LEXICON.keys())

# Word-to-ID mapping
WORD_MAP = {w: i for i, w in enumerate(WORDS)}
WORD_IDS = {i: w for w, i in WORD_MAP.items()}
NUM_WORDS = len(WORDS)

# A word ID for sentence start/end markers (used in LM)
START_WORD = NUM_WORDS       # <s>
END_WORD = NUM_WORDS + 1     # </s>
SENTENCE_BEGIN = START_WORD
SENTENCE_END = END_WORD


def phone_symbols(phone_ids):
    """Convert a list of phone IDs to phone symbols for display."""
    return [PHONE_IDS[pid] for pid in phone_ids]


def word_phones(word):
    """Get phone sequence for a word (as phone symbols)."""
    return phone_symbols(LEXICON[word])


def phone_id(symbol: str) -> int:
    """Get the integer ID for a phone symbol."""
    return PHONE_MAP[symbol]


def word_id(word: str) -> int:
    """Get the integer ID for a word."""
    return WORD_MAP[word]


if __name__ == "__main__":
    print(f"Phone set ({NUM_PHONES} phones):")
    for sym, pid in sorted(PHONE_MAP.items(), key=lambda x: x[1]):
        print(f"  {pid:3d}: {sym}")

    print(f"\nLexicon ({NUM_WORDS} words):")
    for word in WORDS:
        phones = [PHONE_IDS[p] for p in LEXICON[word]]
        print(f"  {word:8s}: {' '.join(phones)}")

    print(f"\nWord IDs:")
    for w, wid in sorted(WORD_MAP.items(), key=lambda x: x[1]):
        print(f"  {wid:3d}: {w}")
    print(f"  {START_WORD:3d}: <s>")
    print(f"  {END_WORD:3d}: </s>")
