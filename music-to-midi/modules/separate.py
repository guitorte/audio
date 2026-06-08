"""Source separation: song → stems via Demucs.

Wraps the Demucs CLI (`python -m demucs`) and flattens its nested output
(`<out>/<model>/<track>/<stem>.wav`) into a flat ``stems/`` directory named by
stem type (``vocals.wav``, ``drums.wav``, ...), which is exactly the layout the
transcription stage expects.

This is the "separa uma música em stems" half of the unified workflow,
adapted from the ``Stems_4`` Colab notebook.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

# Which stem types each Demucs model emits. htdemucs_6s is the only model that
# splits guitar/piano, so it is the default for a full song → MIDI run.
DEMUCS_MODELS: Dict[str, List[str]] = {
    "htdemucs": ["vocals", "drums", "bass", "other"],
    "htdemucs_ft": ["vocals", "drums", "bass", "other"],
    "htdemucs_6s": ["vocals", "drums", "bass", "guitar", "piano", "other"],
    "mdx_extra": ["vocals", "drums", "bass", "other"],
    "mdx_extra_q": ["vocals", "drums", "bass", "other"],
}

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


@dataclass
class SeparationResult:
    """Outcome of a Demucs separation run."""

    track_name: str
    model: str
    stems_dir: str
    stems: Dict[str, str]  # stem_type -> absolute path

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        listed = ", ".join(sorted(self.stems))
        return (
            f"SeparationResult(track={self.track_name!r}, model={self.model!r}, "
            f"stems=[{listed}])"
        )


def _pick_device(device: Optional[str]) -> str:
    """Return 'cuda' when available, else 'cpu', unless explicitly forced."""
    if device:
        return device
    try:
        import torch  # noqa: WPS433 (lazy: keep import optional)

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def separate_stems(
    song_path: str,
    output_dir: str,
    *,
    model: str = "htdemucs_6s",
    two_stems: Optional[str] = None,
    mp3: bool = False,
    mp3_bitrate: int = 320,
    shifts: int = 1,
    overlap: float = 0.25,
    jobs: int = 0,
    device: Optional[str] = None,
    quiet: bool = False,
) -> SeparationResult:
    """Separate ``song_path`` into stems and flatten them into ``output_dir``.

    Parameters mirror the Demucs CLI flags. ``shifts`` and ``overlap`` trade
    runtime for quality (the ``Stems_4`` notebook used ``shifts=5``,
    ``overlap=0.75``). ``two_stems`` restricts output to ``<stem>`` and
    ``no_<stem>`` (e.g. ``"vocals"`` for a vocal/instrumental split).

    Returns a :class:`SeparationResult` mapping stem type → flattened path.
    """
    if not os.path.exists(song_path):
        raise FileNotFoundError(f"Song not found: {song_path}")
    if model not in DEMUCS_MODELS:
        known = ", ".join(sorted(DEMUCS_MODELS))
        print(f"[separate] aviso: modelo '{model}' não catalogado (conhecidos: {known})")

    track_name = os.path.splitext(os.path.basename(song_path))[0]
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Demucs builds <tmp>/<model>/<track>/<stem>.<ext>; isolate it then flatten.
    tmp_root = os.path.join(output_dir, "_demucs_raw")
    os.makedirs(tmp_root, exist_ok=True)

    ext = "mp3" if mp3 else "wav"
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "-o", tmp_root,
        "-d", _pick_device(device),
        "--shifts", str(shifts),
        "--overlap", str(overlap),
        "-j", str(jobs),
    ]
    if mp3:
        cmd += ["--mp3", "--mp3-bitrate", str(mp3_bitrate)]
    if two_stems:
        cmd += ["--two-stems", two_stems]
    cmd.append(song_path)

    if not quiet:
        print("[separate]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    produced = os.path.join(tmp_root, model, track_name)
    raw_files = sorted(glob.glob(os.path.join(produced, f"*.{ext}")))
    if not raw_files:
        raise RuntimeError(
            f"Demucs não produziu stems em {produced} — verifique o modelo/instalação."
        )

    stems: Dict[str, str] = {}
    for src in raw_files:
        stem_type = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(output_dir, f"{stem_type}.{ext}")
        shutil.move(src, dst)
        stems[stem_type] = dst
        if not quiet:
            print(f"[separate]   {stem_type:8s} -> {dst}")

    # Clean the now-empty Demucs scaffolding.
    shutil.rmtree(tmp_root, ignore_errors=True)

    return SeparationResult(
        track_name=track_name,
        model=model,
        stems_dir=output_dir,
        stems=stems,
    )


def list_audio_files(folder: str) -> List[str]:
    """Return sorted audio files (by extension) directly inside ``folder``."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in _AUDIO_EXTS
    )
