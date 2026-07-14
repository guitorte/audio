"""Audio-derived timbral features for one stem (via librosa).

This is the layer the MIDI can't provide — brightness, texture, dynamics,
harmonicity — the stuff that distinguishes "dark, distorted, punchy" from
"airy, tonal, even". Heavy deps (numpy, librosa) are imported lazily. For speed
on mobile Colab we analyse a bounded window (``analyze_seconds`` from
``offset``) at a modest sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .descriptors import (
    brightness_word,
    density_word,
    dynamics_word,
    harmonicity_word,
    texture_word,
)


@dataclass
class AudioFeatures:
    name: str
    sr: int
    duration_s: float
    centroid_hz: float = 0.0
    rolloff_hz: float = 0.0
    bandwidth_hz: float = 0.0
    flatness: float = 0.0
    zcr: float = 0.0
    rms: float = 0.0
    crest_db: float = 0.0
    onset_rate: float = 0.0
    harmonic_ratio: float = 0.0
    # derived words
    brightness: str = ""
    texture: str = ""
    dynamics: str = ""
    harmonicity: str = ""
    density: str = ""


def extract_audio_features(
    wav_path: str,
    *,
    name: Optional[str] = None,
    sr: int = 22050,
    analyze_seconds: Optional[float] = 60.0,
    offset: float = 0.0,
    quiet: bool = False,
) -> AudioFeatures:
    """Extract timbral descriptors from ``wav_path``.

    Loads a mono window (``offset`` .. ``offset+analyze_seconds``) and reduces
    each frame-wise spectral feature to its mean, then maps the numbers to short
    words via :mod:`descriptors`.
    """
    import numpy as np
    import librosa

    import os
    base = name or os.path.splitext(os.path.basename(wav_path))[0]

    y, sr = librosa.load(wav_path, sr=sr, mono=True,
                         offset=offset, duration=analyze_seconds)
    dur = float(len(y) / sr) if sr else 0.0
    feat = AudioFeatures(name=base, sr=sr, duration_s=round(dur, 2))

    if y.size == 0 or float(np.max(np.abs(y))) == 0.0:
        return feat  # silent / empty stem — leave descriptors blank

    feat.centroid_hz = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feat.rolloff_hz = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    feat.bandwidth_hz = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feat.flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    feat.zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms)) or 1e-9
    peak = float(np.max(np.abs(y))) or 1e-9
    feat.rms = round(mean_rms, 5)
    feat.crest_db = round(20.0 * float(np.log10(peak / mean_rms)), 2)

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    feat.onset_rate = round(len(onsets) / dur, 2) if dur > 0 else 0.0

    try:
        y_h, y_p = librosa.effects.hpss(y)
        e_h = float(np.sum(y_h ** 2))
        e_p = float(np.sum(y_p ** 2))
        feat.harmonic_ratio = round(e_h / (e_h + e_p), 3) if (e_h + e_p) > 0 else 0.0
    except Exception as exc:  # noqa: BLE001 - hpss is best-effort
        if not quiet:
            print(f"[audio] HPSS indisponível para {base} ({exc})")

    # Round the raw numbers for the compact output.
    feat.centroid_hz = round(feat.centroid_hz, 1)
    feat.rolloff_hz = round(feat.rolloff_hz, 1)
    feat.bandwidth_hz = round(feat.bandwidth_hz, 1)
    feat.flatness = round(feat.flatness, 4)
    feat.zcr = round(feat.zcr, 4)

    feat.brightness = brightness_word(feat.centroid_hz)
    feat.texture = texture_word(feat.flatness)
    feat.dynamics = dynamics_word(feat.crest_db)
    feat.harmonicity = harmonicity_word(feat.harmonic_ratio)
    feat.density = density_word(feat.onset_rate)
    return feat
