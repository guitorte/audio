"""Orchestration: build a compact ``DIGEST v1`` for a music-to-midi track.

Combines MIDI-derived musical features (:mod:`midi_features`) with
audio-derived timbre (:mod:`audio_features`) into one small, token-cheap text
block per stem — a drop-in replacement for the huge note-by-note MIDI-TEXT dump
when you just want an AI assistant to grasp the melody, tone, timbre and groove.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .common import (
    TrackPaths,
    canonical_stem_type,
    list_audio_files,
    resolve_track_paths,
    stem_sort_key,
)
from .midi_features import MidiFeatures, extract_midi_features
from .audio_features import AudioFeatures, extract_audio_features
from .theory import KeyEstimate, estimate_key


@dataclass
class StemDigest:
    stem: str
    midi: Optional[MidiFeatures] = None
    audio: Optional[AudioFeatures] = None


@dataclass
class TrackDigest:
    track_name: str
    bpm: float
    bpm_source: str
    duration_s: float
    time_sig: str = "4/4"
    key: Optional[KeyEstimate] = None
    stems: List[StemDigest] = field(default_factory=list)
    overview: str = ""


# --- tempo ---------------------------------------------------------------

def _tempo_from_midi(midi_path: str) -> float:
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        _times, tempi = pm.get_tempo_changes()
        if len(tempi):
            return float(tempi[0])
    except Exception:
        pass
    return 120.0


def estimate_tempo(track: TrackPaths, *, source: str = "audio",
                   sr: int = 22050, quiet: bool = False) -> Tuple[float, str]:
    """Estimate BPM. Prefer audio beat-tracking on the drums stem.

    Basic Pitch MIDIs carry a default 120 BPM, so reading tempo from MIDI would
    silently break notes/bar and the rhythm grid. Audio beat-tracking (drums
    first, then any stem) is the reliable source; MIDI is the fallback.
    """
    if source == "midi" and track.merged_midi:
        return _tempo_from_midi(track.merged_midi), "midi"

    stems_dir = track.audio_stems_dir("raw")
    candidates = []
    if stems_dir:
        files = list_audio_files(stems_dir)
        # Prefer drums, then bass, then whatever's there.
        files.sort(key=lambda p: (canonical_stem_type(p) != "drums",
                                  canonical_stem_type(p) != "bass"))
        candidates = files

    for path in candidates:
        try:
            import librosa
            import numpy as np
            y, sr_ = librosa.load(path, sr=sr, mono=True, duration=90.0)
            if y.size == 0:
                continue
            tempo, _beats = librosa.beat.beat_track(y=y, sr=sr_)
            # librosa >=0.10 returns tempo as an array; take the scalar.
            tempo = float(np.atleast_1d(tempo)[0])
            if 40.0 <= tempo <= 250.0:
                return round(tempo, 1), "audio"
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                print(f"[tempo] beat-track falhou em {os.path.basename(path)} ({exc})")
            continue

    if track.merged_midi:
        return _tempo_from_midi(track.merged_midi), "midi"
    return 120.0, "default"


# --- MIDI feature loading ------------------------------------------------

def _load_stem_instruments(track: TrackPaths) -> Dict[str, "object"]:
    """Map stem name → pretty_midi.Instrument.

    Prefers per-stem ``midi/<stem>.mid``; falls back to splitting the merged
    ``<track>.mid`` by instrument name.
    """
    import pretty_midi

    out: Dict[str, object] = {}
    if track.midi_dir and os.path.isdir(track.midi_dir):
        for fn in sorted(os.listdir(track.midi_dir)):
            if not fn.lower().endswith(".mid"):
                continue
            stem = os.path.splitext(fn)[0]
            try:
                pm = pretty_midi.PrettyMIDI(os.path.join(track.midi_dir, fn))
            except Exception:
                continue
            insts = [i for i in pm.instruments if i.notes]
            if not insts:
                continue
            # A per-stem file usually holds one instrument; merge if several.
            merged = insts[0]
            for extra in insts[1:]:
                merged.notes.extend(extra.notes)
            merged.name = merged.name or stem
            out[stem] = merged
    elif track.merged_midi:
        try:
            pm = pretty_midi.PrettyMIDI(track.merged_midi)
            for i, inst in enumerate(pm.instruments):
                if inst.notes:
                    out[inst.name or f"track{i}"] = inst
        except Exception:
            pass
    return out


# --- build ---------------------------------------------------------------

def build_track_digest(
    track_dir: str,
    *,
    sr: int = 22050,
    timbre_from: str = "raw",
    tempo_source: str = "audio",
    bpm: Optional[float] = None,
    analyze_seconds: Optional[float] = 60.0,
    quiet: bool = False,
) -> TrackDigest:
    """Build a :class:`TrackDigest` from a music-to-midi output track folder.

    ``timbre_from``: 'raw' (default) reads ``stems/``; 'clean' reads
    ``stems_clean/`` (denoised+normalized — distorts timbre, use with care).
    ``bpm`` overrides tempo estimation entirely.
    """
    track = resolve_track_paths(track_dir)

    if bpm is not None:
        bpm_val, bpm_src = float(bpm), "manual"
    else:
        bpm_val, bpm_src = estimate_tempo(track, source=tempo_source, sr=sr, quiet=quiet)

    midi_insts = _load_stem_instruments(track)

    audio_dir = track.audio_stems_dir(timbre_from)
    audio_files = {}
    if audio_dir:
        for p in list_audio_files(audio_dir):
            audio_files[os.path.splitext(os.path.basename(p))[0]] = p

    # Union of stem names from MIDI + audio, in canonical order.
    names = sorted(set(midi_insts) | set(audio_files), key=stem_sort_key)

    stems: List[StemDigest] = []
    track_pc = [0.0] * 12
    duration_s = 0.0
    for name in names:
        sd = StemDigest(stem=name)
        inst = midi_insts.get(name)
        if inst is not None:
            sd.midi = extract_midi_features(inst, bpm=bpm_val, name=name)
            duration_s = max(duration_s, sd.midi.duration_s)
            if not sd.midi.is_drum:
                for nt in inst.notes:
                    track_pc[nt.pitch % 12] += (nt.end - nt.start)
        wav = audio_files.get(name)
        if wav is not None and not quiet:
            print(f"[digest] timbre: {os.path.basename(wav)}")
        if wav is not None:
            sd.audio = extract_audio_features(
                wav, name=name, sr=sr, analyze_seconds=analyze_seconds, quiet=quiet)
            duration_s = max(duration_s, sd.audio.duration_s)
        stems.append(sd)

    track_key = estimate_key(track_pc) if sum(track_pc) > 0 else None
    overview = _overview(stems)

    return TrackDigest(
        track_name=track.track_name,
        bpm=bpm_val,
        bpm_source=bpm_src,
        duration_s=round(duration_s, 1),
        key=track_key,
        stems=stems,
        overview=overview,
    )


def _overview(stems: List[StemDigest]) -> str:
    from .descriptors import density_word, dynamics_word, poly_word

    n = len(stems)
    polys = [s.midi.polyphony for s in stems if s.midi and not s.midi.is_drum]
    dens = [s.midi.notes_per_sec for s in stems if s.midi and not s.midi.is_drum]
    crests = [s.audio.crest_db for s in stems if s.audio and s.audio.crest_db]
    poly = poly_word(max(polys)) if polys else "?"
    density = density_word(sum(dens) / len(dens)) if dens else "?"
    dyn = dynamics_word(sum(crests) / len(crests)) if crests else "?"
    poly_full = {"mono": "monophonic", "light-poly": "light-poly",
                 "chordal": "polyphonic"}.get(poly, poly)
    return f"{n} stems | {poly_full} | {density} | {dyn}"


# --- render --------------------------------------------------------------

def render_digest(td: TrackDigest) -> str:
    """Render a :class:`TrackDigest` to the compact ``DIGEST v1`` text."""
    key = td.key.label() if td.key else "n/a"
    conf = f" (conf {td.key.confidence:+.2f})" if td.key and td.key.mode != "atonal" else ""
    lines = [
        f"DIGEST v1 | track: {td.track_name} | {td.time_sig} | "
        f"{td.bpm:g} bpm ({td.bpm_source}) | {td.duration_s:g}s | key: {key}{conf}",
        "UNITS: pitch=sci-name | dens=notes/s | vel=0-127 | "
        "centroid=Hz | crest=dB | h=harmonic-ratio",
        f"OVERVIEW: {td.overview}",
        "",
    ]
    for sd in td.stems:
        lines.extend(_render_stem(sd))
    return "\n".join(lines).rstrip() + "\n"


def _render_stem(sd: StemDigest) -> List[str]:
    m = sd.midi
    a = sd.audio
    out: List[str] = []

    if m and m.is_drum:
        pieces = "+".join(m.drum_pieces) if m.drum_pieces else "drums"
        hits = f"{m.notes_per_sec:g} hits/s"
        pat = m.drum_pattern or ""
        out.append(f"[{sd.stem}] {pieces} | {hits} | {pat}".rstrip())
    elif m and m.n_notes:
        key_fit = ""
        if m.key and m.key.mode != "atonal":
            key_fit = f" | key-fit {m.key.tonic}{'maj' if m.key.mode == 'major' else 'min'}"
        intervals = ""
        if m.dominant_intervals:
            intervals = " | intervals " + ",".join(m.dominant_intervals)
        out.append(
            f"[{sd.stem}] {m.poly_word} | {m.pitch_lo_name}..{m.pitch_hi_name} | "
            f"{m.n_notes} notes {m.notes_per_sec:g}/s ~{m.notes_per_bar:g}/bar"
            f"{key_fit}{intervals}"
        )
        if m.rhythm:
            out.append(f"  rhythm: {m.rhythm.label()} | "
                       f"vel {m.vel_min}-{m.vel_max} {m.vel_word}")
    else:
        # audio-only stem (no MIDI notes)
        out.append(f"[{sd.stem}]")

    if a and a.brightness:
        out.append(
            f"  timbre: {a.brightness}, {a.texture}, {a.dynamics}  "
            f"(centroid {a.centroid_hz:g} · flat {a.flatness:g} · "
            f"crest {a.crest_db:g} · h {a.harmonic_ratio:g} · onset {a.onset_rate:g}/s)"
        )
    return out


def write_digest(td: TrackDigest, out_path: str) -> str:
    """Write the rendered digest to ``out_path`` and return the path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(render_digest(td))
    return out_path


def default_digest_path(track_dir: str) -> str:
    """Where write_digest puts the file by default: analysis/<track>_digest.txt."""
    track = resolve_track_paths(track_dir)
    out_dir = track.analysis_dir or track.track_dir
    return os.path.join(out_dir, f"{track.track_name}_digest.txt")
