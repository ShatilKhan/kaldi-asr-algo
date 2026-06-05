# Kaldi-ASR-Algo

Python + numpy implementation of **"The Kaldi Speech Recognition Toolkit"**
(Povey, Ghoshal et al., IEEE ASRU 2011) — built to understand the paper through code.

## What this implements

The full monophone GMM-HMM pipeline from the Kaldi example recipes (`egs/an4/s5/run.sh`):

```
raw audio (.wav)
    │
    ├── data/reader.py  — Load FSDD digit dataset
    │
    ├── feats.py        — MFCC + delta + delta-delta + CMVN (39-dim)
    │
    ├── hmm.py          — 3-state left-to-right HMM topology per phone
    ├── gmm.py          — Diagonal GMM with EM training + split
    │
    ├── train.py        — Flat-start → align → re-estimate → split × 2
    │
    ├── fst.py          — WFST class with epsilon-filter composition
    ├── lexicon.py      — Pronunciation dictionary (10 digits, 22 phones)
    ├── lm.py           — Bigram language model with add-k smoothing
    │
    ├── decoder.py      — HCLG = H ∘ L ∘ G, token-passing Viterbi decode
    │
    ├── eval.py         — WER/CER/SER with bootstrap CI (matching compute-wer format)
    │
    └── run.py          — End-to-end demo
```

## How to run

```bash
uv venv && uv pip install numpy
uv run python run.py
```

First run downloads FSDD (~10 MB) automatically. Expect ~2 minutes for training
and ~30 seconds for decoding 1000 test utterances.

## What the code covers

| Paper section | Implementation |
|---|---|
| II: Toolkit overview + FSTs | `fst.py` — FST class, 3-state epsilon-filter composition, best-path |
| III: Feature extraction (MFCC, PLP) | `feats.py` — FFT → Mel filterbank → log → DCT → deltas → CMVN |
| IV-A: GMM acoustic models | `gmm.py` — diagonal GMM with EM, log-likelihood scoring, split |
| IV-C: HMM topology | `hmm.py` — 3-state left-to-right per phone |
| V: Phonetic decision trees | Skipped (monophone only, no context tying) |
| VI: Language modeling | `lm.py` — bigram with add-k smoothing |
| VII: Creating decoding graphs | `decoder.py` — HCLG = H ∘ L ∘ G composition |
| VIII: Decoders | Token-passing Viterbi with acoustic weight scaling |
| IX: Evaluation | `eval.py` — WER/CER/SER + bootstrap 95% CI |

## Known limitations

- **Model quality**: The flat-start monophone training with perturbed initialization
  doesn't converge to a useful model. The Viterbi alignment assigns most frames to
  silence (pdf-id 0) because all initial GMMs are nearly identical. This is a known
  challenge even in Kaldi's real `train_mono.sh` — it uses many more iterations (40)
  with careful Gaussian splitting schedules (`--max-iters`, `--realign-iters`, etc.).
  Increasing training iterations and using better initial separation between phones
  would improve WER.

- **Monophone only**: No triphone context dependency (paper Section V / C FST).
  Real Kaldi recipes use triphones after monophone training for better accuracy.

- **No LDA/MLLT fMLLR/SAT**: The feature and speaker adaptation transforms
  described in the paper are not implemented. These can significantly improve WER.

- **No lattices**: The decoder outputs the single best path only, not a lattice
  for rescoring (paper Section VIII mentions lattice generation as future work).

- **No C FST**: Context-dependent triphone expansion is skipped. The HCLG graph
  uses monophone HMMs directly composed with L and G.

## File structure

```
data/download_fsdd.py     # Download FSDD dataset
data/reader.py            # .wav loader + train/test split by speaker
feats.py                  # MFCC feature extraction
hmm.py                    # 3-state left-to-right HMM topology
gmm.py                    # Diagonal GMM with EM training
fst.py                    # WFST class + composition + best-path search
lexicon.py                # Phone set + pronunciation dictionary
lm.py                     # Bigram language model + perplexity
decoder.py                # HCLG assembly + token-passing Viterbi
train.py                  # Monophone training pipeline
eval.py                   # WER/CER/SER + bootstrap CI
run.py                    # End-to-end demo entry point
models/                   # Saved trained GMMs (JSON)
```

## References

- Povey, Ghoshal et al., "The Kaldi Speech Recognition Toolkit," IEEE ASRU 2011.
- FSDD: Free Spoken Digit Dataset (Jakobovski, MIT License).
- Mohri, Pereira, Riley, "Weighted Finite-State Transducers in Speech Recognition," 2002.
