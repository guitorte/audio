"""Stem → MIDI conversion.

Dispatches based on stem type:
  * drums       → ADTOF-pytorch (kick/snare/hat/tom/cymbals, GM drum map)
  * everything  → Spotify Basic Pitch (pitched polyphonic transcription)

Also provides batch_stems_to_midi() for converting all stems of a Demucs
output directory and merging them into a single multi-track MIDI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TranscriptionResult:
    midi_path: str
    midi_data: "pretty_midi.PrettyMIDI"  # noqa: F821
    note_events: Optional[list]
    n_notes: int
    duration_s: float
    engine: str


@dataclass
class BatchResult:
    midi_path: str
    midi_data: "pretty_midi.PrettyMIDI"  # noqa: F821
    stem_results: Dict[str, TranscriptionResult]
    total_notes: int
    duration_s: float


STEM_PRESETS = {
    "bass": {
        "engine": "basic_pitch",
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length": 80.0,
        "minimum_frequency": 30.0,
        "maximum_frequency": 350.0,
    },
    "vocals": {
        "engine": "basic_pitch",
        "onset_threshold": 0.6,
        "frame_threshold": 0.3,
        "minimum_note_length": 100.0,
        "minimum_frequency": 80.0,
        "maximum_frequency": 1100.0,
    },
    "guitar": {
        "engine": "basic_pitch",
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length": 58.0,
        "minimum_frequency": 70.0,
        "maximum_frequency": 1500.0,
    },
    "piano": {
        "engine": "basic_pitch",
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length": 58.0,
        "minimum_frequency": 27.5,
        "maximum_frequency": 4200.0,
    },
    "drums": {
        "engine": "adtof",
    },
    "other": {
        "engine": "basic_pitch",
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length": 58.0,
    },
}


# General MIDI program assigned to each stem in the merged multi-track .mid.
# Drum tracks use is_drum=True so their program byte is ignored by GM players.
DEFAULT_STEM_PROGRAMS = {
    "vocals": 53,   # Voice Oohs
    "bass":   33,   # Electric Bass (finger)
    "guitar": 27,   # Electric Guitar (clean)
    "piano":  0,    # Acoustic Grand Piano
    "other":  0,    # Acoustic Grand Piano (neutral)
    "drums":  0,    # is_drum=True → program ignored
}

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def stem_to_midi(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    stem_type: str = "other",
) -> TranscriptionResult:
    """Transcribe a single audio stem into a MIDI file.

    Routes to the appropriate transcription engine based on stem_type.
    Drums use ADTOF-pytorch; all other types use Basic Pitch with a
    per-instrument preset (see STEM_PRESETS).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    preset = STEM_PRESETS.get(stem_type, STEM_PRESETS["other"]).copy()
    engine = preset.pop("engine")

    if output_path is None:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + ".mid"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    if engine == "adtof":
        return _transcribe_drums(input_path, output_path)
    if engine == "basic_pitch":
        return _transcribe_pitched(input_path, output_path, **preset)
    raise ValueError(f"Unknown engine '{engine}' for stem_type='{stem_type}'")


def _transcribe_pitched(
    input_path: str,
    output_path: str,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
) -> TranscriptionResult:
    """Pitched transcription via Spotify Basic Pitch (ONNX backend)."""
    from basic_pitch.inference import predict

    _model_output, midi_data, note_events = predict(
        input_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    midi_data.write(output_path)
    return TranscriptionResult(
        midi_path=output_path,
        midi_data=midi_data,
        note_events=note_events,
        n_notes=sum(len(i.notes) for i in midi_data.instruments),
        duration_s=midi_data.get_end_time(),
        engine="basic_pitch",
    )


def _transcribe_drums(input_path: str, output_path: str) -> TranscriptionResult:
    """Drum transcription via ADTOF-pytorch (kick/snare/hat/tom/cymbals, GM)."""
    from adtof_pytorch import transcribe_to_midi
    import pretty_midi

    transcribe_to_midi(input_path, output_path)

    midi_data = pretty_midi.PrettyMIDI(output_path)
    return TranscriptionResult(
        midi_path=output_path,
        midi_data=midi_data,
        note_events=None,
        n_notes=sum(len(i.notes) for i in midi_data.instruments),
        duration_s=midi_data.get_end_time(),
        engine="adtof",
    )


def batch_stems_to_midi(
    stems_dir: str,
    output_path: str,
    *,
    stem_types: Optional[Dict[str, str]] = None,
    extra_stems: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> BatchResult:
    """Transcribe every stem in a Demucs-style directory and merge into one MIDI.

    Scans `stems_dir` for audio files (.wav/.mp3/.flac/.m4a/.ogg) whose basename
    matches a known stem_type (vocals, drums, bass, guitar, piano, other).
    Runs `stem_to_midi()` on each, writes per-stem `<stem_type>.mid` files next
    to `output_path`, and merges all tracks into a single multi-track MIDI at
    `output_path`.

    Args:
        stems_dir: directory containing per-stem audio files (e.g. Demucs's
            `separated/htdemucs_6s/<track>/` folder with vocals.wav, drums.wav,
            bass.wav, ...).
        output_path: path for the merged multi-track .mid file.
        stem_types: optional mapping {filename_stem_without_ext: stem_type} to
            remap non-standard filenames (e.g. {"lead_vox": "vocals"}).
        extra_stems: optional {stem_type: absolute_file_path} to include files
            from outside `stems_dir` (overrides discoveries with the same type).
        verbose: print progress for each stem.
    """
    import pretty_midi

    if not os.path.isdir(stems_dir):
        raise FileNotFoundError(f"Stems directory not found: {stems_dir}")

    output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(output_dir, exist_ok=True)

    known = set(STEM_PRESETS.keys())
    rename = dict(stem_types or {})

    discovered: Dict[str, str] = {}
    for entry in sorted(os.listdir(stems_dir)):
        full = os.path.join(stems_dir, entry)
        if not os.path.isfile(full):
            continue
        name, ext = os.path.splitext(entry)
        if ext.lower() not in _AUDIO_EXTS:
            continue
        stem_type = rename.get(name, name.lower())
        if stem_type not in known:
            if verbose:
                print(f"  skip {entry}: '{stem_type}' is not a known stem_type")
            continue
        if stem_type in discovered:
            if verbose:
                print(f"  skip {entry}: duplicate stem_type '{stem_type}'")
            continue
        discovered[stem_type] = full

    for stem_type, path in (extra_stems or {}).items():
        if stem_type not in known:
            raise ValueError(f"Unknown stem_type in extra_stems: '{stem_type}'")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"extra_stems['{stem_type}'] not found: {path}")
        discovered[stem_type] = path

    if not discovered:
        raise ValueError(
            f"No recognizable stems found in {stems_dir}. "
            f"Expected files named like vocals.wav, drums.wav, bass.wav, ..."
        )

    if verbose:
        print(f"Found {len(discovered)} stem(s):")
        for stem_type, path in discovered.items():
            print(f"  {stem_type:8s} <- {os.path.basename(path)}")

    stem_results: Dict[str, TranscriptionResult] = {}
    for stem_type, audio_path in discovered.items():
        per_stem_midi = os.path.join(output_dir, f"{stem_type}.mid")
        if verbose:
            print(f"\n[{stem_type}] -> {per_stem_midi}")
        result = stem_to_midi(audio_path, per_stem_midi, stem_type=stem_type)
        if verbose:
            print(
                f"  engine={result.engine} notes={result.n_notes} "
                f"dur={result.duration_s:.1f}s"
            )
        stem_results[stem_type] = result

    merged = pretty_midi.PrettyMIDI()
    for stem_type, result in stem_results.items():
        program = DEFAULT_STEM_PROGRAMS.get(stem_type, 0)
        for src in result.midi_data.instruments:
            inst = pretty_midi.Instrument(
                program=src.program if src.is_drum else program,
                is_drum=src.is_drum,
                name=stem_type,
            )
            inst.notes = list(src.notes)
            inst.pitch_bends = list(src.pitch_bends)
            inst.control_changes = list(src.control_changes)
            merged.instruments.append(inst)
    merged.write(output_path)

    total_notes = sum(r.n_notes for r in stem_results.values())
    duration_s = merged.get_end_time()
    if verbose:
        print(f"\nMerged -> {output_path}")
        print(
            f"  tracks={len(merged.instruments)} "
            f"total_notes={total_notes} duration={duration_s:.1f}s"
        )

    return BatchResult(
        midi_path=output_path,
        midi_data=merged,
        stem_results=stem_results,
        total_notes=total_notes,
        duration_s=duration_s,
    )
