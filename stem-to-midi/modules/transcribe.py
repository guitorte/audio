"""Stem → MIDI conversion.

Dispatches based on stem type:
  * drums       → ADTOF-pytorch (kick/snare/hat/tom/cymbals, GM drum map)
  * everything  → Spotify Basic Pitch (pitched polyphonic transcription)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionResult:
    midi_path: str
    midi_data: "pretty_midi.PrettyMIDI"  # noqa: F821
    note_events: Optional[list]
    n_notes: int
    duration_s: float
    engine: str


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
