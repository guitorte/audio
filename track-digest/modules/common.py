"""Shared helpers + track-folder resolution.

This sub-project consumes an existing ``music-to-midi`` output track folder, so
the small utilities it needs (``list_audio_files``, ``canonical_stem_type``) are
*copied* here rather than cross-imported — mirroring the self-containment
precedent in ``music-to-midi/modules/transcribe.py`` (which copies from
``stem-to-midi``). Cross-importing would only work in Colab where the whole repo
is cloned, and would break local / CLI use from inside this folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

# Accepted audio extensions (copied from music-to-midi/modules/separate.py).
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

# Preferred display/processing order for the canonical stem types.
STEM_ORDER = ("vocals", "bass", "guitar", "piano", "other", "drums")


def list_audio_files(folder: str) -> List[str]:
    """Return sorted audio files (by extension) directly inside ``folder``."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in _AUDIO_EXTS
    )


def canonical_stem_type(name: str) -> Optional[str]:
    """Infer a known stem type from a filename or label.

    Returns one of ``STEM_ORDER`` or ``None``. Substring-based and
    case-insensitive so Demucs outputs ('vocals.wav') and looser names
    ('lead_vocal') both resolve. Copied from
    ``music-to-midi/modules/transcribe.py``.
    """
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    if stem.startswith("no_") or "instrumental" in stem:
        return "other"
    aliases = [
        ("vocals", ("vocal", "vox", "voz", "voc")),
        ("drums", ("drum", "bateria", "perc")),
        ("bass", ("bass", "baixo")),
        ("guitar", ("guitar", "guitarra", "violao", "violão")),
        ("piano", ("piano", "keys", "teclado")),
        ("other", ("other", "outros")),
    ]
    for canonical, needles in aliases:
        if any(n in stem for n in needles):
            return canonical
    return None


def stem_sort_key(name: str):
    """Sort key placing canonical stems in ``STEM_ORDER``, unknowns last."""
    canon = canonical_stem_type(name) or name
    try:
        return (0, STEM_ORDER.index(canon), name)
    except ValueError:
        return (1, 0, name)


@dataclass
class TrackPaths:
    """Resolved layout of a ``music-to-midi`` output track folder.

    Any of the optional dirs/files may be ``None`` when absent — this project
    degrades gracefully rather than assuming a full pipeline run.
    """

    track_name: str
    track_dir: str
    stems_dir: Optional[str] = None
    stems_clean_dir: Optional[str] = None
    midi_dir: Optional[str] = None
    merged_midi: Optional[str] = None
    instrumental_midi: Optional[str] = None
    analysis_dir: Optional[str] = None

    def audio_stems_dir(self, prefer: str = "raw") -> Optional[str]:
        """Return the stems dir to read audio from ('raw' or 'clean')."""
        if prefer == "clean" and self.stems_clean_dir:
            return self.stems_clean_dir
        return self.stems_dir or self.stems_clean_dir


def resolve_track_paths(path: str) -> TrackPaths:
    """Resolve a track folder from either ``<track>/`` or ``<track>/stems``.

    Probes for the standard music-to-midi sub-layout (``stems/``,
    ``stems_clean/``, ``midi/``, ``analysis/``, ``<track>.mid``). Missing pieces
    are left as ``None``.
    """
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Track folder not found: {path}")

    # Allow pointing straight at a stems/ dir — step up to the track dir.
    if os.path.basename(path) in ("stems", "stems_clean"):
        track_dir = os.path.dirname(path)
    else:
        track_dir = path

    track_name = os.path.basename(track_dir.rstrip(os.sep))

    def _dir(name: str) -> Optional[str]:
        d = os.path.join(track_dir, name)
        return d if os.path.isdir(d) else None

    def _file(name: str) -> Optional[str]:
        f = os.path.join(track_dir, name)
        return f if os.path.isfile(f) else None

    return TrackPaths(
        track_name=track_name,
        track_dir=track_dir,
        stems_dir=_dir("stems"),
        stems_clean_dir=_dir("stems_clean"),
        midi_dir=_dir("midi"),
        merged_midi=_file(f"{track_name}.mid"),
        instrumental_midi=_file(f"{track_name}_instrumental.mid"),
        analysis_dir=_dir("analysis"),
    )
