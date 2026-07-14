"""Threshold-band mappers: numeric audio/MIDI features → short words.

Each function maps a measured value to a compact qualitative descriptor. All
bands live here as module-level constants so they are easy to tune in one place.
The words are chosen to read naturally when pasted into an AI assistant, e.g.
"dark, distorted, punchy" for a driven drum bus.
"""

from __future__ import annotations


def _band(value: float, edges, words: list) -> str:
    """Return ``words[i]`` for the first edge ``value`` falls under."""
    for edge, word in zip(edges, words):
        if value < edge:
            return word
    return words[-1]


# --- audio timbre ---------------------------------------------------------

def brightness_word(centroid_hz: float) -> str:
    """Spectral centroid → perceived brightness (low = 'dark'/low-pass)."""
    return _band(centroid_hz, (800, 2000, 4000), ["dark", "warm", "bright", "airy"])


def texture_word(flatness: float) -> str:
    """Spectral flatness → tonal ↔ noisy/distorted texture."""
    return _band(flatness, (0.02, 0.10, 0.30),
                 ["tonal", "textured", "noisy", "distorted"])


def dynamics_word(crest_db: float) -> str:
    """Crest factor (dB) → how compressed vs. transient/punchy the level is."""
    return _band(crest_db, (9, 15, 20),
                 ["compressed", "even", "dynamic", "punchy"])


def harmonicity_word(h_ratio: float) -> str:
    """Harmonic/(harmonic+percussive) energy ratio → tonal vs. percussive."""
    return _band(h_ratio, (0.35, 0.65), ["percussive", "mixed", "harmonic"])


def density_word(rate: float) -> str:
    """Event rate (per second) → sparse ↔ dense."""
    return _band(rate, (0.7, 2.5, 5.0), ["sparse", "moderate", "busy", "dense"])


# --- MIDI musical ---------------------------------------------------------

def vel_spread_word(spread: float) -> str:
    """Velocity range (0-127) → dynamic variation in the performance."""
    return _band(spread, (12, 35), ["flat", "moderate", "wide"])


def poly_word(polyphony: float) -> str:
    """Mean concurrent voices → monophonic ↔ chordal."""
    return _band(polyphony, (1.15, 2.5), ["mono", "light-poly", "chordal"])


def syncopation_word(offbeat_frac: float) -> str:
    """Fraction of off-grid-beat onsets → on-beat ↔ syncopated feel."""
    return _band(offbeat_frac, (0.15, 0.35),
                 ["on-beat", "some syncopation", "syncopated"])
