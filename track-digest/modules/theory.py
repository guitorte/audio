"""Music-theory helpers: note names, GM drum map, key estimation.

``note_name`` and ``GM_DRUMS`` are copied from
``music-to-midi/modules/analyze.py`` to keep this project self-contained.
Key estimation uses the Krumhansl-Kessler pitch-class profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# General MIDI percussion map (channel 10) — readable labels for drum pitches.
GM_DRUMS = {
    35: "AcousticBassDrum", 36: "BassDrum1", 37: "SideStick", 38: "AcousticSnare",
    39: "HandClap", 40: "ElectricSnare", 41: "LowFloorTom", 42: "ClosedHiHat",
    43: "HighFloorTom", 44: "PedalHiHat", 45: "LowTom", 46: "OpenHiHat",
    47: "LowMidTom", 48: "HiMidTom", 49: "CrashCymbal1", 50: "HighTom",
    51: "RideCymbal1", 52: "ChineseCymbal", 53: "RideBell", 54: "Tambourine",
    55: "SplashCymbal", 56: "Cowbell", 57: "CrashCymbal2", 59: "RideCymbal2",
}

# Interval class (semitones mod 12) → short label, for chord/interval hints.
INTERVAL_NAMES = {
    0: "unison", 1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "P4",
    6: "tritone", 7: "P5", 8: "m6", 9: "M6", 10: "m7", 11: "M7",
}


def note_name(pitch: int) -> str:
    """MIDI pitch number → scientific note name (60 → 'C4')."""
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def drum_group(pitch: int) -> str:
    """Collapse a GM drum pitch into a coarse family for pattern summaries."""
    if pitch in (35, 36):
        return "kick"
    if pitch in (37, 38, 40):
        return "snare"
    if pitch in (42, 44, 46):
        return "hihat"
    if pitch in (41, 43, 45, 47, 48, 50):
        return "tom"
    if pitch in (49, 51, 52, 53, 55, 57, 59):
        return "cymbal"
    if pitch == 39:
        return "clap"
    return "perc"


# Krumhansl-Kessler key profiles (major / minor), tonic-relative weights.
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


@dataclass
class KeyEstimate:
    """Estimated key/scale of a pitch-class weight vector."""

    tonic: str
    mode: str          # 'major' | 'minor' | 'atonal'
    confidence: float  # best correlation, ~[-1, 1]
    second: Optional[str] = None  # runner-up label, e.g. 'A minor'

    def label(self) -> str:
        if self.mode == "atonal":
            return "atonal"
        return f"{self.tonic} {self.mode}"


def _pearson(a: List[float], b: List[float]) -> float:
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    num = sum(x * y for x, y in zip(da, db))
    den = (sum(x * x for x in da) * sum(y * y for y in db)) ** 0.5
    return num / den if den else 0.0


def estimate_key(pc_weights) -> KeyEstimate:
    """Estimate key from a 12-bin (duration-weighted) pitch-class vector.

    Correlates each of 12 rotations against the major and minor Krumhansl
    profiles; the best correlation wins. Duration-weighting (rather than raw
    note counts) resists transcription noise (spurious short notes).
    """
    weights = [float(w) for w in pc_weights]
    if len(weights) != 12 or sum(weights) <= 0:
        return KeyEstimate(tonic="?", mode="atonal", confidence=0.0)

    scored = []  # (corr, tonic_pc, mode)
    for tonic in range(12):
        rotated = weights[tonic:] + weights[:tonic]
        scored.append((_pearson(rotated, MAJOR_PROFILE), tonic, "major"))
        scored.append((_pearson(rotated, MINOR_PROFILE), tonic, "minor"))
    scored.sort(key=lambda t: t[0], reverse=True)

    best_corr, best_tonic, best_mode = scored[0]
    second_corr, second_tonic, second_mode = scored[1]
    return KeyEstimate(
        tonic=NOTE_NAMES[best_tonic],
        mode=best_mode,
        confidence=round(best_corr, 3),
        second=f"{NOTE_NAMES[second_tonic]} {second_mode}",
    )
