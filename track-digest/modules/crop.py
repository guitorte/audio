"""Batch-crop all stems of a track to one time region → WAV.

Reuses the ``music-to-midi`` I/O idiom (``librosa.load(offset=, duration=)`` →
``soundfile.write(..., subtype="PCM_24")``). Cropping every stem to the same
region at once is the common need — grab a chorus across vocals/drums/bass/etc.
Heavy deps are imported lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .common import canonical_stem_type, list_audio_files


@dataclass
class CropResult:
    src_dir: str
    out_dir: str
    start_s: float
    end_s: float
    crops: Dict[str, str] = field(default_factory=dict)  # stem name -> path
    skipped: List[str] = field(default_factory=list)


def probe_duration(path: str) -> float:
    """Duration of an audio file in seconds (for bounding UI sliders)."""
    import librosa
    return float(librosa.get_duration(path=path))


def _fmt(x: float) -> str:
    """Format a time for filenames: 12.5 -> '012p50' (sortable, dot-free)."""
    return f"{x:06.2f}".replace(".", "p")


def crop_audio(
    path: str,
    out_path: str,
    start_s: float,
    end_s: float,
    *,
    to_mono: bool = False,
    sr: Optional[int] = None,
    subtype: str = "PCM_24",
) -> str:
    """Crop ``[start_s, end_s)`` of ``path`` to ``out_path`` (WAV)."""
    import librosa
    import soundfile as sf

    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be > start_s ({start_s})")

    y, sr_ = librosa.load(path, sr=sr, mono=to_mono,
                          offset=float(start_s), duration=float(end_s - start_s))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    data = y.T if y.ndim > 1 else y  # soundfile wants (frames, channels)
    sf.write(out_path, data, sr_, subtype=subtype)
    return out_path


def _resolve_stems_dir(src: str) -> str:
    """Accept a track dir (use its stems/) or a stems dir directly."""
    src = os.path.abspath(src)
    stems = os.path.join(src, "stems")
    if os.path.isdir(stems):
        return stems
    return src  # assume src already points at a folder of stems


def crop_stems(
    src: str,
    start_s: float,
    end_s: float,
    *,
    out_dir: Optional[str] = None,
    to_mono: bool = False,
    sr: Optional[int] = None,
    subtype: str = "PCM_24",
    include: Optional[List[str]] = None,
    quiet: bool = False,
) -> CropResult:
    """Crop every stem in a folder to ``[start_s, end_s)`` and write WAVs.

    ``src`` may be a track dir (its ``stems/`` is used) or a stems folder.
    Outputs go to ``out_dir`` (default ``<src>/crops/``) named
    ``<stem>_<start>-<end>s.wav``. ``include`` filters by canonical stem name
    (e.g. ``["vocals", "bass"]``). ``end_s`` is clamped to each file's duration.
    """
    stems_dir = _resolve_stems_dir(src)
    files = list_audio_files(stems_dir)
    if not files:
        raise FileNotFoundError(f"No audio stems found in {stems_dir}")

    if out_dir is None:
        base = os.path.dirname(stems_dir) if os.path.basename(stems_dir) == "stems" else stems_dir
        out_dir = os.path.join(base, "crops")
    os.makedirs(out_dir, exist_ok=True)

    result = CropResult(src_dir=stems_dir, out_dir=out_dir,
                        start_s=float(start_s), end_s=float(end_s))

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if include is not None:
            canon = canonical_stem_type(path) or stem
            if stem not in include and canon not in include:
                result.skipped.append(stem)
                continue
        dur = probe_duration(path)
        end = min(float(end_s), dur)
        if end <= start_s:
            if not quiet:
                print(f"[crop] {stem}: região fora do stem (dur {dur:.1f}s) — pulando")
            result.skipped.append(stem)
            continue
        out_path = os.path.join(out_dir, f"{stem}_{_fmt(start_s)}-{_fmt(end)}s.wav")
        if not quiet:
            print(f"[crop] {stem}: {start_s:.2f}-{end:.2f}s -> {os.path.basename(out_path)}")
        crop_audio(path, out_path, start_s, end, to_mono=to_mono, sr=sr, subtype=subtype)
        result.crops[stem] = out_path

    return result
