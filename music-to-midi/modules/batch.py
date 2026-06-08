"""Batch stem → multi-track MIDI.

Transcribes every recognised stem in a folder and consolidates the results
into a single multi-track ``.mid`` file (one instrument per stem, drums on
channel 10). Mirrors the ``Stem_to_MIDI_Batch`` Colab notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from .separate import list_audio_files
from .transcribe import (
    DEFAULT_STEM_PROGRAMS,
    TranscriptionResult,
    canonical_stem_type,
    stem_to_midi,
)


@dataclass
class BatchResult:
    midi_path: str
    midi_data: "pretty_midi.PrettyMIDI"  # noqa: F821
    stem_results: Dict[str, TranscriptionResult] = field(default_factory=dict)
    total_notes: int = 0
    duration_s: float = 0.0


def _merge_into(merged, result: TranscriptionResult, stem_type: str) -> None:
    """Append a stem's transcription as one (or more) tracks in ``merged``."""
    import pretty_midi

    program = DEFAULT_STEM_PROGRAMS.get(stem_type, 0)
    is_drum = stem_type == "drums"

    target = pretty_midi.Instrument(
        program=0 if is_drum else program,
        is_drum=is_drum,
        name=stem_type,
    )
    for inst in result.midi_data.instruments:
        for note in inst.notes:
            target.notes.append(note)
    if target.notes:
        merged.instruments.append(target)


def batch_stems_to_midi(
    stems_dir: str,
    output_path: str,
    *,
    stem_types: Optional[Dict[str, str]] = None,
    per_stem_dir: Optional[str] = None,
    quiet: bool = False,
) -> BatchResult:
    """Transcribe all stems in ``stems_dir`` into one multi-track MIDI.

    ``stem_types`` optionally overrides the auto-detected type for specific
    filenames (basename → stem type), e.g. ``{'Kim_Vocal_2': 'vocals'}``.
    Per-stem ``.mid`` files land in ``per_stem_dir`` (default: alongside
    ``output_path``); the consolidated file is written to ``output_path``.
    """
    import pretty_midi

    files = list_audio_files(stems_dir)
    if not files:
        raise RuntimeError(f"Nenhum stem de áudio encontrado em {stems_dir}")

    if per_stem_dir is None:
        per_stem_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(per_stem_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    overrides = {k.lower(): v for k, v in (stem_types or {}).items()}

    merged = pretty_midi.PrettyMIDI()
    stem_results: Dict[str, TranscriptionResult] = {}

    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        stem_type = overrides.get(base.lower()) or canonical_stem_type(path)
        if stem_type is None:
            if not quiet:
                print(f"[batch] ignorado (tipo desconhecido): {os.path.basename(path)}")
            continue
        if stem_type in stem_results:
            if not quiet:
                print(f"[batch] '{stem_type}' já processado — pulando {os.path.basename(path)}")
            continue

        per_stem_path = os.path.join(per_stem_dir, f"{stem_type}.mid")
        if not quiet:
            print(f"[batch] {os.path.basename(path)} -> {stem_type}")
        result = stem_to_midi(path, per_stem_path, stem_type=stem_type)
        stem_results[stem_type] = result
        _merge_into(merged, result, stem_type)

    if not merged.instruments:
        raise RuntimeError(
            "Nenhum stem reconhecido produziu notas. Use 'stem_types' para "
            "remapear nomes de arquivos não-padrão."
        )

    merged.write(output_path)
    total_notes = sum(r.n_notes for r in stem_results.values())
    duration_s = merged.get_end_time()

    return BatchResult(
        midi_path=output_path,
        midi_data=merged,
        stem_results=stem_results,
        total_notes=total_notes,
        duration_s=duration_s,
    )
