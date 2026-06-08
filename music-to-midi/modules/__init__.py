"""music-to-midi: unified song → stems → multi-track MIDI workflow.

Combines Demucs source separation with per-stem transcription
(Basic Pitch + ADTOF-pytorch) behind a single :func:`song_to_midi` call.
"""

from .analyze import (
    midi_to_text,
    save_piano_roll,
    text_to_midi,
    write_track_analysis,
)
from .batch import BatchResult, batch_stems_to_midi
from .pipeline import PipelineResult, song_to_midi
from .separate import (
    DEMUCS_MODELS,
    SeparationResult,
    list_audio_files,
    separate_stems,
)
from .transcribe import (
    DEFAULT_STEM_PROGRAMS,
    STEM_PRESETS,
    TranscriptionResult,
    canonical_stem_type,
    stem_to_midi,
)

__all__ = [
    "song_to_midi",
    "PipelineResult",
    "separate_stems",
    "SeparationResult",
    "DEMUCS_MODELS",
    "list_audio_files",
    "batch_stems_to_midi",
    "BatchResult",
    "stem_to_midi",
    "TranscriptionResult",
    "STEM_PRESETS",
    "DEFAULT_STEM_PROGRAMS",
    "canonical_stem_type",
    "save_piano_roll",
    "midi_to_text",
    "text_to_midi",
    "write_track_analysis",
]
