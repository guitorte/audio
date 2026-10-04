"""Optional octave check: compare where two vocals sit and suggest a whole-octave shift.

Built for clean, dry vocal stems. Each vocal is pitch-tracked with pYIN, only
confidently voiced frames are kept, and the medians are compared on a log
scale. The gap is rounded to whole octaves, so a gap under 6 semitones means
no shift; a gap near 6 is reported as borderline, because either choice is
defensible there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

SR = 16_000
HOP = 320          # 20 ms
FMIN, FMAX = 65.0, 1050.0   # C2 .. C6: low bass to high soprano
MIN_VOICED_S = 3.0
MAX_ANALYSED_S = 120.0
SEGMENT_S = 15.0


@dataclass
class Pitch:
    median_hz: float
    low_hz: float      # 10th percentile
    high_hz: float     # 90th percentile
    voiced_s: float


@dataclass
class OctaveAdvice:
    source: Pitch
    target: Pitch
    gap_semitones: float
    octaves: int
    borderline: bool

    @property
    def shift(self) -> int:
        return 12 * self.octaves


def note(hz: float) -> str:
    import librosa
    return librosa.hz_to_note(hz, unicode=False)


def _segments(y: np.ndarray, sr: int) -> np.ndarray:
    """Long vocals: evenly spread 15 s pieces, up to 2 minutes in total (pYIN is slow)."""
    if len(y) <= MAX_ANALYSED_S * sr:
        return y
    n, seg = int(MAX_ANALYSED_S // SEGMENT_S), int(SEGMENT_S * sr)
    starts = np.linspace(0, len(y) - seg, n).astype(int)
    return np.concatenate([y[s:s + seg] for s in starts])


def track(audio: np.ndarray, sr: int) -> Pitch | None:
    """Pitch statistics of a vocal (mono or stereo array). None if there's too little singing."""
    import librosa
    y = audio.mean(axis=1) if audio.ndim == 2 else audio
    if sr != SR:
        g = np.gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g)
    y = _segments(y.astype(np.float32), SR)
    f0, voiced, prob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=2048, hop_length=HOP)
    f = f0[voiced & (prob > 0.5) & np.isfinite(f0)]
    voiced_s = len(f) * HOP / SR
    if voiced_s < MIN_VOICED_S:
        return None
    lo, med, hi = np.percentile(np.log2(f), [10, 50, 90])
    return Pitch(float(2 ** med), float(2 ** lo), float(2 ** hi), voiced_s)


def advise(source: Pitch, target: Pitch) -> OctaveAdvice:
    gap = 12 * np.log2(target.median_hz / source.median_hz)
    octaves = int(np.sign(gap) * np.floor(abs(gap) / 12 + 0.5))  # half away from zero, not numpy's half-to-even
    borderline = abs(abs(gap) % 12 - 6) < 1.0
    return OctaveAdvice(source, target, float(gap), octaves, borderline)


def describe(a: OctaveAdvice) -> str:
    def span(p: Pitch) -> str:
        return f"median {note(p.median_hz)} ({p.median_hz:.0f} Hz), range {note(p.low_hz)}–{note(p.high_hz)}"
    if a.octaves == 0:
        verdict = "no octave change"
    else:
        verdict = f"{a.shift:+d} semitones ({abs(a.octaves)} octave{'s' if abs(a.octaves) > 1 else ''} " \
                  f"{'up' if a.octaves > 0 else 'down'})"
    lines = [f"   source vocal: {span(a.source)}",
             f"   target voice: {span(a.target)}",
             f"   gap {a.gap_semitones:+.1f} semitones → {verdict}"]
    if a.borderline:
        lines.append("   ⚠️ borderline (gap close to half an octave): listen to both, or set the shift by hand")
    return "\n".join(lines)
