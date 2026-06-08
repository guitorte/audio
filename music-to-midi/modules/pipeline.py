"""End-to-end workflow: song → stems → multi-track MIDI.

Glues the two halves of the project — Demucs separation (``separate.py``) and
batch transcription (``batch.py``) — into a single call, writing everything to
an organised per-track directory tree::

    <output_root>/<track>/
    ├── stems/          # vocals.wav, drums.wav, bass.wav, ...
    ├── midi/           # vocals.mid, drums.mid, bass.mid, ...   (per-stem)
    └── <track>.mid     # consolidated multi-track MIDI
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

from .batch import BatchResult, batch_stems_to_midi
from .separate import SeparationResult, separate_stems


@dataclass
class PipelineResult:
    track_name: str
    track_dir: str
    stems_dir: str
    midi_dir: str
    merged_midi: str
    separation: SeparationResult
    transcription: BatchResult


def song_to_midi(
    song_path: str,
    output_root: str,
    *,
    model: str = "htdemucs_6s",
    stem_types: Optional[Dict[str, str]] = None,
    # separation knobs (forwarded to separate_stems)
    two_stems: Optional[str] = None,
    shifts: int = 1,
    overlap: float = 0.25,
    jobs: int = 0,
    device: Optional[str] = None,
    mp3_stems: bool = False,
    skip_separation_if_exists: bool = True,
    quiet: bool = False,
) -> PipelineResult:
    """Run the full song → stems → MIDI workflow for one track.

    Output is organised under ``<output_root>/<track>/`` (see module docstring).
    Set ``skip_separation_if_exists`` to reuse stems already separated in a
    previous run. ``stem_types`` remaps non-standard stem filenames.
    """
    if not os.path.exists(song_path):
        raise FileNotFoundError(f"Song not found: {song_path}")

    track_name = os.path.splitext(os.path.basename(song_path))[0]
    track_dir = os.path.join(os.path.abspath(output_root), track_name)
    stems_dir = os.path.join(track_dir, "stems")
    midi_dir = os.path.join(track_dir, "midi")
    merged_midi = os.path.join(track_dir, f"{track_name}.mid")
    os.makedirs(stems_dir, exist_ok=True)
    os.makedirs(midi_dir, exist_ok=True)

    # --- Stage 1: separation -------------------------------------------------
    from .separate import list_audio_files

    existing = list_audio_files(stems_dir)
    if skip_separation_if_exists and existing:
        if not quiet:
            print(f"[pipeline] reaproveitando {len(existing)} stems em {stems_dir}")
        stems = {
            os.path.splitext(os.path.basename(p))[0]: p for p in existing
        }
        separation = SeparationResult(
            track_name=track_name,
            model=model,
            stems_dir=stems_dir,
            stems=stems,
        )
    else:
        if not quiet:
            print(f"[pipeline] separando '{track_name}' com {model} ...")
        separation = separate_stems(
            song_path,
            stems_dir,
            model=model,
            two_stems=two_stems,
            mp3=mp3_stems,
            shifts=shifts,
            overlap=overlap,
            jobs=jobs,
            device=device,
            quiet=quiet,
        )

    # --- Stage 2: transcription ---------------------------------------------
    if not quiet:
        print(f"[pipeline] transcrevendo {len(separation.stems)} stems -> MIDI ...")
    transcription = batch_stems_to_midi(
        stems_dir,
        merged_midi,
        stem_types=stem_types,
        per_stem_dir=midi_dir,
        quiet=quiet,
    )

    if not quiet:
        print(f"[pipeline] pronto: {merged_midi}")

    return PipelineResult(
        track_name=track_name,
        track_dir=track_dir,
        stems_dir=stems_dir,
        midi_dir=midi_dir,
        merged_midi=merged_midi,
        separation=separation,
        transcription=transcription,
    )
