"""Stem → MIDI conversion using Spotify Basic Pitch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionResult:
    midi_path: str
    midi_data: "pretty_midi.PrettyMIDI"  # noqa: F821
    note_events: list
    n_notes: int
    duration_s: float


def stem_to_midi(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
) -> TranscriptionResult:
    """Transcribe a single audio stem into a MIDI file.

    Parameters
    ----------
    input_path : str
        Path to the input audio file (wav/mp3/flac/m4a).
    output_path : str, optional
        Destination .mid path. Defaults to input_path with .mid suffix.
    onset_threshold : float
        Basic Pitch onset detection threshold (0–1). Higher = fewer notes.
    frame_threshold : float
        Basic Pitch frame activation threshold (0–1). Higher = stricter.
    minimum_note_length : float
        Minimum note length in milliseconds.
    minimum_frequency, maximum_frequency : float, optional
        Restrict pitch range in Hz (useful for bass / vocals).
    """
    from basic_pitch.inference import predict

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    if output_path is None:
        stem, _ = os.path.splitext(input_path)
        output_path = stem + ".mid"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    _model_output, midi_data, note_events = predict(
        input_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    midi_data.write(output_path)

    n_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    duration_s = midi_data.get_end_time()

    return TranscriptionResult(
        midi_path=output_path,
        midi_data=midi_data,
        note_events=note_events,
        n_notes=n_notes,
        duration_s=duration_s,
    )


# Suggested Basic Pitch parameter presets per stem type.
# These are sensible defaults — tune them per song.
STEM_PRESETS = {
    "bass": {
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length": 80.0,
        "minimum_frequency": 30.0,
        "maximum_frequency": 350.0,
    },
    "vocals": {
        "onset_threshold": 0.6,
        "frame_threshold": 0.3,
        "minimum_note_length": 100.0,
        "minimum_frequency": 80.0,
        "maximum_frequency": 1100.0,
    },
    "guitar": {
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length": 58.0,
        "minimum_frequency": 70.0,
        "maximum_frequency": 1500.0,
    },
    "piano": {
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length": 58.0,
        "minimum_frequency": 27.5,
        "maximum_frequency": 4200.0,
    },
    "other": {
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length": 58.0,
    },
}
