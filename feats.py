"""
MFCC feature extraction for ASR.

Implements the textbook MFCC pipeline (paper Section III):
  Audio → framing (25ms, 10ms shift) → Hamming window → FFT
  → Mel filterbank (26 bands) → log → DCT (keep 13)
  → + delta + delta-delta → 39-dim vector per frame
  → CMVN (cepstral mean/variance normalization per speaker)

Depends on numpy for FFT and matrix operations.
"""

import numpy as np
from typing import Optional

# --- Constants ---
SAMPLE_RATE = 8000       # FSDD sample rate
FRAME_LEN_MS = 25.0      # frame length in ms
FRAME_SHIFT_MS = 10.0    # frame shift in ms
N_FFT = 512              # FFT size (power of 2 > frame length)
N_MELS = 26              # number of Mel filterbank bands
N_MFCC = 13              # number of DCT coefficients to keep
LOW_FREQ = 0.0           # low cutoff for filterbank
HIGH_FREQ = SAMPLE_RATE / 2.0  # Nyquist
PREEMPH_ALPHA = 0.97     # pre-emphasis coefficient


# ---- Utility functions ----

def hertz_to_mel(freq: float) -> float:
    """Convert frequency in Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hertz(mel: float) -> float:
    """Convert Mel scale back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def preemphasis(samples: np.ndarray, alpha: float = PREEMPH_ALPHA) -> np.ndarray:
    """
    Apply pre-emphasis filter: y[n] = x[n] - alpha * x[n-1].
    Boosts high frequencies, which are naturally quieter in speech.
    """
    out = np.empty_like(samples)
    out[0] = samples[0]
    out[1:] = samples[1:] - alpha * samples[:-1]
    return out


def frame_signal(
    samples: np.ndarray,
    sample_rate: int,
    frame_len_ms: float = FRAME_LEN_MS,
    frame_shift_ms: float = FRAME_SHIFT_MS,
) -> np.ndarray:
    """
    Chop a 1D signal into overlapping frames.
    Returns (num_frames, frame_len) array.
    """
    frame_len = int(sample_rate * frame_len_ms / 1000.0)
    frame_shift = int(sample_rate * frame_shift_ms / 1000.0)
    num_frames = max(1, int(np.ceil((len(samples) - frame_len) / frame_shift)) + 1)

    # Pad signal if needed
    padded_len = (num_frames - 1) * frame_shift + frame_len
    if len(samples) < padded_len:
        samples = np.pad(samples, (0, padded_len - len(samples)))

    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_len)
    # Take every frame_shift-th window
    frames = frames[::frame_shift]
    return frames[:num_frames]


def hamming_window(frame_len: int) -> np.ndarray:
    """Generate a Hamming window of length frame_len."""
    n = np.arange(frame_len)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (frame_len - 1))


def mel_filterbank(
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    sample_rate: int = SAMPLE_RATE,
    low_freq: float = LOW_FREQ,
    high_freq: float = HIGH_FREQ,
) -> np.ndarray:
    """
    Build a Mel filterbank matrix.
    Returns (n_mels, n_fft // 2 + 1) array.
    Each row is a triangular filter on the Mel scale.
    """
    low_mel = hertz_to_mel(low_freq)
    high_mel = hertz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hertz(mel_points)

    fft_bins = np.floor((n_fft / 2 + 1) * hz_points / (sample_rate / 2)).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)

    for i in range(n_mels):
        left = fft_bins[i]
        center = fft_bins[i + 1]
        right = fft_bins[i + 2]
        # Rising edge
        for j in range(left, center):
            fbank[i, j] = (j - left) / (center - left)
        # Falling edge
        for j in range(center, right):
            fbank[i, j] = (right - j) / (right - center)

    return fbank


def dct_matrix(n_input: int, n_output: int) -> np.ndarray:
    """
    Type-II DCT matrix. Returns (n_output, n_input).
    y_k = sum_{i=0}^{n_input-1} x_i * cos(k * pi * (i + 0.5) / n_input)
    """
    i = np.arange(n_input)
    k = np.arange(n_output)[:, None]
    matrix = np.cos(k * np.pi * (i + 0.5) / n_input)
    # Standard normalization
    matrix[0] *= np.sqrt(1.0 / n_input)
    matrix[1:] *= np.sqrt(2.0 / n_input)
    return matrix


# ---- Full MFCC pipeline ----

def mfcc(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    n_mfcc: int = N_MFCC,
    include_deltas: bool = True,
) -> np.ndarray:
    """
    Compute MFCC features for a 1D audio signal.

    Args:
        samples: 1D float32 array normalized to [-1, 1].
        sample_rate: Sample rate in Hz.
        n_fft: FFT size.
        n_mels: Number of Mel filterbank bands.
        n_mfcc: Number of DCT coefficients.
        include_deltas: If True, append delta and delta-delta (39-dim).

    Returns:
        (num_frames, n_mfcc) or (num_frames, 3*n_mfcc) array.
    """
    # 1. Pre-emphasis
    samples = preemphasis(samples)

    # 2. Framing
    frames = frame_signal(samples, sample_rate)  # (num_frames, frame_len)

    # 3. Hamming window
    frame_len = frames.shape[1]
    win = hamming_window(frame_len)
    frames = frames * win  # broadcast

    # 4. FFT → power spectrum
    spectrum = np.fft.rfft(frames, n=n_fft)  # (num_frames, n_fft//2+1)
    power = np.abs(spectrum) ** 2

    # 5. Mel filterbank
    fbank = mel_filterbank(n_fft, n_mels, sample_rate)  # (n_mels, n_fft//2+1)
    mel_energy = power @ fbank.T  # (num_frames, n_mels)

    # Avoid log(0)
    mel_energy = np.maximum(mel_energy, 1e-10)

    # 6. Log
    log_mel = np.log(mel_energy)

    # 7. DCT → MFCCs
    dct = dct_matrix(n_mels, n_mfcc)
    mfcc_feats = log_mel @ dct.T  # (num_frames, n_mfcc)

    if not include_deltas:
        return mfcc_feats.astype(np.float64)

    # 8. Delta and delta-delta
    delta = _compute_delta(mfcc_feats)
    delta2 = _compute_delta(delta)

    # Stack: [mfcc, delta, delta2] along feature axis
    return np.hstack([mfcc_feats, delta, delta2]).astype(np.float64)


def _compute_delta(feats: np.ndarray, window: int = 2) -> np.ndarray:
    """
    Compute delta (velocity) features.
    Simple regression: delta_t = sum_{w=1}^{W} w * (x_{t+w} - x_{t-w}) / (2 * sum w^2)
    """
    padded = np.pad(feats, ((window, window), (0, 0)), mode="edge")
    denom = 2.0 * sum(w ** 2 for w in range(1, window + 1))
    delta = np.zeros_like(feats)
    for w in range(1, window + 1):
        delta += w * (padded[window + w: len(padded) - window + w] - padded[window - w: len(padded) - window - w])
    delta /= denom
    return delta


def apply_cmvn(
    feats: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply Cepstral Mean and Variance Normalization per utterance.

    CMVN subtracts the mean and divides by the standard deviation of each
    feature dimension across time. This cancels microphone/channel effects.

    If mean/std are provided, they are used instead of computing from feats
    (useful for normalizing test data with training stats).

    Args:
        feats: (num_frames, feat_dim) array.
        mean: (feat_dim,) optional pre-computed mean.
        std: (feat_dim,) optional pre-computed std.

    Returns:
        Normalized (num_frames, feat_dim) array.
    """
    if mean is None:
        mean = np.mean(feats, axis=0)
    if std is None:
        std = np.std(feats, axis=0)
    std = np.maximum(std, 1e-10)  # avoid division by zero
    return (feats - mean) / std


def extract_mfcc(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convenience: compute MFCC + deltas + optional CMVN.

    Args:
        samples: 1D audio signal as float32 [-1, 1].
        sample_rate: Sample rate in Hz.
        normalize: If True, apply CMVN.

    Returns:
        (num_frames, 39) array.
    """
    feats = mfcc(samples, sample_rate, include_deltas=True)
    if normalize and feats.shape[0] > 0:
        feats = apply_cmvn(feats)
    return feats


# ---- Test / demo ----
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.reader import load_fsdd, prepare_dataset
    from data.download_fsdd import ensure_fsdd

    data_dir = ensure_fsdd()
    train, _ = prepare_dataset(data_dir)
    print(f"Loaded {len(train)} training recordings")

    # Extract MFCCs for the first recording
    rec = train[0]
    feats = extract_mfcc(rec["samples"], rec["sample_rate"])
    print(f"\n--- MFCC for {rec['digit']} by {rec['speaker']} ---")
    print(f"  Audio: {len(rec['samples'])} samples @ {rec['sample_rate']} Hz")
    print(f"  Frames: {feats.shape[0]}")
    print(f"  Feature dim: {feats.shape[1]} (13 MFCC + 13 delta + 13 delta-delta)")
    print(f"  Mean (dim 0): {np.mean(feats[:, 0]):.4f}")
    print(f"  Std  (dim 0): {np.std(feats[:, 0]):.4f}")
    print(f"  Shape: {feats.shape}")
    print(f"  First frame (first 5 dims): {feats[0, :5]}")

    # Quick sanity: extract all training MFCCs and check shape
    all_feats = [extract_mfcc(r["samples"], r["sample_rate"]) for r in train[:10]]
    print(f"\n  MFCC shapes for first 10 utterances:")
    for i, f in enumerate(all_feats):
        print(f"    {train[i]['digit']} ({train[i]['speaker']}): {f.shape}")
