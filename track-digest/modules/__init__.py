"""track-digest — compact AI digest + batch time-crop for a music-to-midi track.

Read-only post-processing over a ``music-to-midi`` output track folder
(``<output_root>/<track>/`` with ``stems/``, ``midi/``, ``analysis/`` …):

* :func:`build_track_digest` + :func:`render_digest` — a tiny ``DIGEST v1``
  text summary (melody/harmony/rhythm from MIDI + timbre from audio) that's
  cheap to paste into an AI assistant.
* :func:`crop_stems` — crop every stem to one time region and export WAVs.

Heavy deps (librosa, soundfile, pretty_midi, numpy) are imported lazily inside
functions, so ``import modules`` stays cheap.
"""

from __future__ import annotations

from .common import (
    TrackPaths,
    canonical_stem_type,
    list_audio_files,
    resolve_track_paths,
)
from .theory import KeyEstimate, estimate_key, note_name, GM_DRUMS, drum_group
from .descriptors import (
    brightness_word,
    density_word,
    dynamics_word,
    harmonicity_word,
    texture_word,
)
from .midi_features import MidiFeatures, RhythmProfile, extract_midi_features
from .audio_features import AudioFeatures, extract_audio_features
from .digest import (
    StemDigest,
    TrackDigest,
    build_track_digest,
    default_digest_path,
    estimate_tempo,
    render_digest,
    write_digest,
)
from .crop import CropResult, crop_audio, crop_stems, probe_duration

__all__ = [
    # digest
    "build_track_digest",
    "render_digest",
    "write_digest",
    "default_digest_path",
    "estimate_tempo",
    "TrackDigest",
    "StemDigest",
    # features
    "extract_midi_features",
    "extract_audio_features",
    "MidiFeatures",
    "RhythmProfile",
    "AudioFeatures",
    # theory
    "estimate_key",
    "KeyEstimate",
    "note_name",
    "GM_DRUMS",
    "drum_group",
    # descriptors
    "brightness_word",
    "texture_word",
    "dynamics_word",
    "harmonicity_word",
    "density_word",
    # crop
    "crop_stems",
    "crop_audio",
    "probe_duration",
    "CropResult",
    # paths
    "resolve_track_paths",
    "TrackPaths",
    "list_audio_files",
    "canonical_stem_type",
]
