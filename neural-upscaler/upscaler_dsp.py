"""Small numpy/scipy helpers shared by the main pipeline and the AudioSR venv."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def detect_cutoff(audio, sr):
    """Frequency where the spectrum falls off a cliff (an MP3 / AAC shelf).

    Returns ``sr / 2`` when there is no clear shelf, i.e. the file is already
    full band. The value sits just below the knee, so a crossover there hands
    the whole empty region to the restorer.
    """
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    if rms(mono) < 1e-6:
        return sr / 2
    freqs, psd = welch(mono, fs=sr, nperseg=8192)
    level = np.convolve(10 * np.log10(psd + 1e-20), np.ones(9) / 9, mode="same")
    df = freqs[1] - freqs[0]
    step = int(round(1000 / df))
    idx = np.where((freqs > 4000) & (freqs < sr / 2 - 1500))[0]
    if len(idx) == 0:
        return sr / 2
    drops = level[idx] - level[idx + step]
    i = idx[int(np.argmax(drops))]
    plateau = np.median(level[(freqs > freqs[i] - 3000) & (freqs < freqs[i] - 500)])
    floor = np.median(level[(freqs > freqs[i] + 1000) & (freqs < freqs[i] + 2500)])
    if drops.max() < 20 or plateau - floor < 30:
        return sr / 2
    j = i + step
    while j > 0 and level[j] < plateau - 6:
        j -= 1
    return float(max(freqs[j] - 300, 2000))


def band_rms(signal, sr, lo, hi):
    sos = butter(4, [lo, min(hi, 0.99 * sr / 2)], btype="bandpass", fs=sr, output="sos")
    return rms(sosfiltfilt(sos, signal, axis=0))


def crossover(low_source, high_source, cutoff, sr, order=8):
    """``low_source`` below ``cutoff`` + ``high_source`` above it.

    Butterworth low/high-pass pairs are power complementary
    (|L|² + |H|² = 1) and filtfilt applies |·|² with zero phase, so the two
    bands sum back to exactly unity gain with no phase smear at the seam.
    """
    lp = butter(order, cutoff, btype="lowpass", fs=sr, output="sos")
    hp = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(lp, low_source, axis=0) + sosfiltfilt(hp, high_source, axis=0)


def chunk_starts(total, chunk, overlap):
    hop = chunk - overlap
    starts = [0]
    while starts[-1] + chunk < total:
        starts.append(starts[-1] + hop)
    return starts


def crossfade(length, overlap, fade_in, fade_out):
    w = np.ones(length, dtype=np.float32)
    n = min(overlap, length)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if fade_in and n:
        w[:n] = ramp
    if fade_out and n:
        w[-n:] = np.minimum(w[-n:], ramp[::-1])
    return w
