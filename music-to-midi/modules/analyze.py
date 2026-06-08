"""Per-stem analysis artefacts: piano-roll images + AI-readable MIDI text.

Two deliverables per MIDI:

* **Piano roll PNG** — a visual of the notes (pitched tracks coloured per stem,
  drums on a separate row with GM labels).
* **MIDI-TEXT** — a plain-text, line-oriented description of every note that an
  LLM can read to "understand" the melody/pattern *and* regenerate the exact
  MIDI. :func:`text_to_midi` is the round-trip parser that proves the format is
  complete (it rebuilds the file from the MIDI-note column).

matplotlib is used via the object-oriented ``Figure`` API (no pyplot), so saving
images never disturbs the notebook's inline backend. Heavy deps are imported
lazily.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# General MIDI percussion map (channel 10) — readable labels for drum pitches.
GM_DRUMS = {
    35: "AcousticBassDrum", 36: "BassDrum1", 37: "SideStick", 38: "AcousticSnare",
    39: "HandClap", 40: "ElectricSnare", 41: "LowFloorTom", 42: "ClosedHiHat",
    43: "HighFloorTom", 44: "PedalHiHat", 45: "LowTom", 46: "OpenHiHat",
    47: "LowMidTom", 48: "HiMidTom", 49: "CrashCymbal1", 50: "HighTom",
    51: "RideCymbal1", 52: "ChineseCymbal", 53: "RideBell", 54: "Tambourine",
    55: "SplashCymbal", 56: "Cowbell", 57: "CrashCymbal2", 59: "RideCymbal2",
}

# Short drum labels for the piano-roll y-axis.
GM_DRUMS_SHORT = {
    35: "BD", 36: "Kick", 38: "Snare", 39: "Clap", 40: "ElSnr", 41: "LoFlT",
    42: "HH", 43: "HiFlT", 44: "PdHH", 45: "LoT", 46: "OpHH", 47: "LMT",
    48: "HMT", 49: "Crash", 50: "HiT", 51: "Ride",
}

STEM_COLORS = {
    "vocals": "#d62728", "bass": "#1f77b4", "guitar": "#2ca02c",
    "piano": "#ff7f0e", "other": "#9467bd", "drums": "#7f7f7f",
}


def note_name(pitch: int) -> str:
    """MIDI pitch number → scientific note name (60 → 'C4')."""
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _tempo_bpm(pm) -> float:
    """Best-effort initial tempo in BPM (pretty_midi default is 120)."""
    try:
        _times, tempi = pm.get_tempo_changes()
        if len(tempi):
            return float(tempi[0])
    except Exception:
        pass
    return 120.0


def save_piano_roll(pm, path: str, *, title: Optional[str] = None) -> str:
    """Render ``pm`` (a PrettyMIDI) to a piano-roll PNG at ``path``.

    Pitched instruments share one panel (coloured by stem name); drums get a
    second panel with GM percussion labels. Uses a standalone ``Figure`` so the
    notebook's inline matplotlib backend is left untouched.
    """
    from matplotlib.figure import Figure

    pitched = [i for i in pm.instruments if not i.is_drum and i.notes]
    drums = [i for i in pm.instruments if i.is_drum and i.notes]
    n_rows = (1 if pitched else 0) + (1 if drums else 0)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if n_rows == 0:
        fig = Figure(figsize=(8, 2))
        ax = fig.subplots()
        ax.text(0.5, 0.5, "(sem notas)", ha="center", va="center")
        ax.axis("off")
        if title:
            fig.suptitle(title, fontweight="bold")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        return path

    fig = Figure(figsize=(14, 4 * n_rows))
    axes = fig.subplots(n_rows, 1, squeeze=False)
    ax_idx = 0

    if pitched:
        ax = axes[ax_idx][0]
        ax_idx += 1
        for inst in pitched:
            color = STEM_COLORS.get(inst.name, "#000000")
            for n in inst.notes:
                ax.barh(n.pitch, n.end - n.start, left=n.start, height=0.9,
                        color=color, alpha=0.65, edgecolor="none")
            ax.scatter([], [], color=color, label=f"{inst.name} (n={len(inst.notes)})")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Pitch MIDI")
        ax.set_title("Pitched tracks", fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, axis="y", alpha=0.3)

    if drums:
        ax = axes[ax_idx][0]
        for inst in drums:
            for n in inst.notes:
                ax.barh(n.pitch, max(0.05, n.end - n.start), left=n.start, height=0.85,
                        color=STEM_COLORS["drums"], alpha=0.8, edgecolor="none")
        unique_pitches = sorted({n.pitch for inst in drums for n in inst.notes})
        ax.set_yticks(unique_pitches)
        ax.set_yticklabels([GM_DRUMS_SHORT.get(p, f"p{p}") for p in unique_pitches])
        ax.set_xlabel("Tempo (s)")
        ax.set_title("Drum track (GM channel 10)", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def midi_to_text(pm, path: str) -> str:
    """Serialise ``pm`` to a line-oriented, AI-readable text file at ``path``.

    The format is self-describing: a header with tempo/duration, then one
    ``TRACK_BEGIN…TRACK_END`` block per instrument, each listing notes as
    ``NOTE  start_s  dur_s  pitch  midi  vel``. The ``midi`` column is the
    authoritative pitch (0-127); ``pitch`` is a human label (note name, or GM
    drum name for drum tracks). :func:`text_to_midi` reads it back exactly.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    bpm = _tempo_bpm(pm)

    lines = [
        "# MIDI-TEXT v1 — representação textual de um MIDI, recriável por IA.",
        "# Tempos em segundos. Cada nota: NOTE  start_s  dur_s  pitch  midi  vel",
        "# 'midi' (0-127) é o pitch autoritativo; 'pitch' é só o rótulo legível.",
        "# Em tracks de bateria (IS_DRUM: true), 'pitch' é o nome GM da peça.",
        "# Para regenerar o .mid: modules.analyze.text_to_midi(este_arquivo).",
        f"TEMPO_BPM: {bpm:.2f}",
        f"DURATION_S: {pm.get_end_time():.3f}",
        f"N_TRACKS: {len(pm.instruments)}",
    ]

    for inst in pm.instruments:
        notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
        lines.append("TRACK_BEGIN")
        lines.append(f"NAME: {inst.name or 'track'}")
        lines.append(f"PROGRAM: {inst.program}")
        lines.append(f"IS_DRUM: {'true' if inst.is_drum else 'false'}")
        lines.append(f"N_NOTES: {len(notes)}")
        if notes:
            pitches = [n.pitch for n in notes]
            if inst.is_drum:
                lo = GM_DRUMS.get(min(pitches), str(min(pitches)))
                hi = GM_DRUMS.get(max(pitches), str(max(pitches)))
            else:
                lo, hi = note_name(min(pitches)), note_name(max(pitches))
            lines.append(f"PITCH_RANGE: {lo}..{hi}")
        lines.append("# NOTE  start_s  dur_s  pitch  midi  vel")
        for n in notes:
            label = GM_DRUMS.get(n.pitch, f"Drum{n.pitch}") if inst.is_drum else note_name(n.pitch)
            lines.append(
                f"NOTE\t{n.start:.3f}\t{n.end - n.start:.3f}\t{label}\t{n.pitch}\t{n.velocity}"
            )
        lines.append("TRACK_END")

    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def text_to_midi(text_path: str, out_path: Optional[str] = None):
    """Parse a MIDI-TEXT file (see :func:`midi_to_text`) back into a PrettyMIDI.

    Reconstructs from the authoritative ``midi`` pitch column, so the round-trip
    is lossless. Writes to ``out_path`` when given. Returns the PrettyMIDI.
    """
    import pretty_midi

    bpm = 120.0
    tracks = []
    cur = None
    with open(text_path) as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("TEMPO_BPM:"):
                bpm = float(s.split(":", 1)[1])
            elif s == "TRACK_BEGIN":
                cur = {"program": 0, "is_drum": False, "name": "track", "notes": []}
            elif s == "TRACK_END":
                if cur is not None:
                    tracks.append(cur)
                cur = None
            elif s == "END":
                break
            elif cur is None:
                continue
            elif s.startswith("NAME:"):
                cur["name"] = s.split(":", 1)[1].strip()
            elif s.startswith("PROGRAM:"):
                cur["program"] = int(s.split(":", 1)[1])
            elif s.startswith("IS_DRUM:"):
                cur["is_drum"] = "true" in s.split(":", 1)[1].lower()
            elif s.startswith("NOTE"):
                parts = s.split("\t") if "\t" in s else s.split()
                start, dur = float(parts[1]), float(parts[2])
                midi, vel = int(parts[4]), int(parts[5])
                cur["notes"].append((start, dur, midi, vel))

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for d in tracks:
        inst = pretty_midi.Instrument(
            program=d["program"], is_drum=d["is_drum"], name=d["name"]
        )
        for start, dur, midi, vel in d["notes"]:
            inst.notes.append(
                pretty_midi.Note(velocity=vel, pitch=midi, start=start, end=start + dur)
            )
        pm.instruments.append(inst)

    if out_path:
        pm.write(out_path)
    return pm


def write_track_analysis(pm, out_dir: str, track_name: str) -> Dict[str, Dict[str, str]]:
    """Write piano-roll PNG + MIDI-TEXT for the whole track and each stem.

    Produces, under ``out_dir``:
      * ``<track>_pianoroll.png`` / ``<track>.txt``    — the full multi-track MIDI
      * ``<stem>_pianoroll.png`` / ``<stem>.txt``      — one per instrument/stem

    Returns ``{'piano_rolls': {...}, 'texts': {...}}`` keyed by stem name
    (``'_merged'`` for the consolidated files).
    """
    import pretty_midi

    os.makedirs(out_dir, exist_ok=True)
    bpm = _tempo_bpm(pm)
    out: Dict[str, Dict[str, str]] = {"piano_rolls": {}, "texts": {}}

    merged_png = os.path.join(out_dir, f"{track_name}_pianoroll.png")
    merged_txt = os.path.join(out_dir, f"{track_name}.txt")
    save_piano_roll(pm, merged_png, title=f"{track_name} — multi-track")
    midi_to_text(pm, merged_txt)
    out["piano_rolls"]["_merged"] = merged_png
    out["texts"]["_merged"] = merged_txt

    for inst in pm.instruments:
        name = inst.name or "track"
        single = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        single.instruments.append(inst)
        png = os.path.join(out_dir, f"{name}_pianoroll.png")
        txt = os.path.join(out_dir, f"{name}.txt")
        save_piano_roll(single, png, title=name)
        midi_to_text(single, txt)
        out["piano_rolls"][name] = png
        out["texts"][name] = txt

    return out
