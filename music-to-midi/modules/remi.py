"""Assemble per-stem MIDIs into one song and write it as MIDI-TEXT v2 (REMI-style).

Used by ``notebooks/Assemble_MIDI_to_REMI.ipynb``.

Assembly
--------
Takes the files the bake-off writes (``<separator>__<engine>__<stem>.mid``) plus
an external vocals file (e.g. the vocal lab's ``<song>_vocals.mid``), picks one
file per stem, and merges them into a playable multi-track MIDI and an
instrumental version. Picking order: an external vocals file beats every
bake-off vocal engine; BS-RoFormer SW stems beat htdemucs_6s; drums come from
the engine the user chooses (ADT_STR or ADTOF). MuScriptor's tracks are a
whole-song system of their own and are left out.

The playable MIDI gets a mix that keeps the melody audible: the vocal on a
square lead, a fixed volume (CC7) and pan per track, and an optional velocity
lift. Tracks whose velocities are all the same (the vocal lab writes a flat 64)
are set to a fixed velocity first.

MIDI-TEXT v2
------------
The same format, header and analysis as the "Song -> Stems -> MIDI v2" Colab
notebook: one ``NOTE bar pos16 start_s dur_s pitch midi vel`` line per note,
with key, tempo and per-bar chords in the header. ``start_s``/``midi`` are
authoritative, so :func:`remi_text_to_midi` gets the notes back exactly. The
text keeps the transcription's own velocities, not the playback mix.

Beat grid (librosa), key (Krumhansl-Schmuckler) and chords (per-bar triad
templates on chroma) come from the song's audio when it is given, otherwise from
a FluidSynth render of the assembled MIDI. Key and chords are estimates.
"""

from __future__ import annotations

import copy
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "neural-upscaler"))
import upscaler as U  # noqa: E402  (shared Drive/file helpers: find_folder, list_audio, select_files, safe_stem)

STEMS = ("vocals", "guitar", "piano", "bass", "other", "drums")
DRUM_ENGINES = ("adt_str", "adtof")
SEPARATOR_RANK = ("bs_roformer_sw", "htdemucs_6s")
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

# Playback mix: GM program, channel volume (CC7), pan (CC10).
MIX = {
    "vocals": (80, 106, 64),   # square lead: the melody cuts through
    "guitar": (27, 106, 40),
    "piano": (0, 102, 88),
    "bass": (33, 127, 64),
    "other": (48, 83, 64),
    "drums": (0, 115, 64),
}
FLAT_VELOCITY = 100

# --- MIDI-TEXT v2 constants (as in the v2 notebook) ---
TARGET_SR = 22050
SUBDIV = 4               # 16th notes
ASSUME_TS = (4, 4)
_KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_GM_DRUM = {35: "AcousticBassDrum", 36: "Kick", 37: "SideStick", 38: "Snare", 39: "Clap",
            40: "ElSnare", 41: "LowFloorTom", 42: "ClosedHiHat", 43: "HiFloorTom", 44: "PedalHiHat",
            45: "LowTom", 46: "OpenHiHat", 47: "LowMidTom", 48: "HiMidTom", 49: "Crash1", 50: "HighTom",
            51: "Ride1", 52: "ChinaCymbal", 53: "RideBell", 54: "Tambourine", 55: "Splash", 56: "Cowbell",
            57: "Crash2", 59: "Ride2"}
VOCAL_NAMES = {"vocals", "vocal", "voz", "voice"}
_TRIADS: dict[str, np.ndarray] = {}
for _i, _root in enumerate(_PC):
    for _name, _steps in ((_root, (0, 4, 7)), (_root + "m", (0, 3, 7))):
        _t = np.zeros(12)
        _t[[(_i + s) % 12 for s in _steps]] = 1
        _TRIADS[_name] = _t


# --------------------------------------------------------------------------- picking files

# An external vocals file is named like "<song>_vocals.mid" ("voz"/"voice" also work), optionally
# with a take suffix the vocal lab or the user adds: "<song>_vocals_dry.mid", "<song>_voz_limpa.mid".
_VOCAL_FILE = re.compile(r"(^|[_\s.-])(vocals?|voz|voice)([_\s.-](dry|wet|clean|seco|seca|limpo|limpa|final|lab))?$", re.I)
# Plain names: "bass.mid", "<song>_bateria.mid" ... (the last word of the name decides).
STEM_WORDS = {"vocals": "vocals", "vocal": "vocals", "voz": "vocals", "voice": "vocals",
              "drums": "drums", "drum": "drums", "bateria": "drums",
              "bass": "bass", "baixo": "bass",
              "guitar": "guitar", "guitarra": "guitar", "violao": "guitar", "violão": "guitar",
              "piano": "piano", "keys": "piano", "teclado": "piano",
              "other": "other", "outros": "other"}
# The bake-off's own whole-song files ("A_atual_htdemucs_6s_Basic_Pitch_ADTOF.mid", ...).
_BAKEOFF_SYSTEM = re.compile(r"^[A-D]_(atual|RoFormer_SW|MuScriptor)", re.I)
_BAKEOFF = re.compile(r"^(?P<sep>[a-z0-9_]+?)__(?P<engine>[a-z0-9_.-]+?)__(?P<stem>vocals|drums|bass|guitar|piano|other)$")


def _clean_stem(path: Path) -> str:
    """File stem without the ' (1)' a browser or Drive adds to repeated downloads."""
    return re.sub(r"\s*\(\d+\)$", "", path.stem).strip()


@dataclass
class Part:
    stem: str
    path: Path
    source: str   # how it was recognised, for the listing


def classify(path: Path) -> Part | None:
    """What a MIDI file is: a bake-off file, an external vocals file, or neither (None)."""
    name = _clean_stem(path)
    if _BAKEOFF_SYSTEM.match(name):
        return None
    m = _BAKEOFF.match(name.lower())
    if m:
        return Part(m["stem"], path, f'{m["sep"]} · {m["engine"]}')
    if _VOCAL_FILE.search(name):
        return Part("vocals", path, "voz externa")
    last = re.split(r"[_\s.-]+", name.lower())[-1]
    if last in STEM_WORDS:
        return Part(STEM_WORDS[last], path, "nome simples")
    return None


def gather(source: Path, work: Path) -> list[Path]:
    """The .mid files in a folder (not recursive into stems/) or inside a .zip (extracted to ``work``)."""
    source = Path(source)
    if source.suffix.lower() == ".zip":
        out = work / U.safe_stem(source)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        with zipfile.ZipFile(source) as z:
            for info in z.infolist():
                name = Path(info.filename).name
                if name.lower().endswith((".mid", ".midi")) and not info.is_dir():
                    (out / name).write_bytes(z.read(info))
        source = out
    if source.is_file():
        return [source]
    files = [p for p in source.rglob("*") if p.suffix.lower() in (".mid", ".midi") and "stems" not in p.parts]
    return sorted(files, key=lambda p: U._nfc(p.name))


def pick(files: list[Path], drums: str = "adt_str") -> tuple[dict[str, Part], list[str]]:
    """One file per stem. Returns the picks and notes on what was skipped."""
    if drums not in DRUM_ENGINES:
        raise ValueError(f"DRUMS deve ser um de {DRUM_ENGINES}")
    parts = [p for p in (classify(f) for f in files) if p]
    notes = []
    for f in files:
        if classify(f) is None:
            why = ("montagem do bake-off" if _BAKEOFF_SYSTEM.match(_clean_stem(f))
                   else "MuScriptor (sistema próprio)" if _clean_stem(f).lower().startswith("mix__")
                   else "nome não reconhecido")
            notes.append(f"ignorado ({why}): {f.name}")

    def rank(p: Part) -> tuple:
        sep, _, engine = p.source.partition(" · ")
        if p.source == "voz externa":
            return (0, 0)
        if p.source == "nome simples":         # a file the user named: as good as the default engine
            return (1, len(SEPARATOR_RANK))
        if sep.startswith("mix"):          # MuScriptor: its own whole-song system
            return (9, 0)
        r_sep = SEPARATOR_RANK.index(sep) if sep in SEPARATOR_RANK else len(SEPARATOR_RANK)
        if p.stem == "drums":
            r_eng = 0 if engine == drums else 1
        elif p.stem == "vocals":
            r_eng = 0 if engine.startswith("game") else 1
        else:
            r_eng = 0 if engine == "basic_pitch" else 1
        return (1 + r_eng, r_sep)

    chosen: dict[str, Part] = {}
    for stem in STEMS:
        cands = sorted((p for p in parts if p.stem == stem and rank(p)[0] < 9), key=rank)
        notes += [f"ignorado (MuScriptor, sistema próprio): {p.path.name}"
                  for p in parts if p.stem == stem and rank(p)[0] >= 9]
        if cands:
            chosen[stem] = cands[0]
            notes += [f"{stem}: preferi {cands[0].path.name} a {c.path.name}" for c in cands[1:]]
    return chosen, notes


# --------------------------------------------------------------------------- assembling

def _tempo_of(path: Path) -> float | None:
    import mido
    for track in mido.MidiFile(str(path)).tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return round(mido.tempo2bpm(msg.tempo), 2)
    return None


def song_tempo(chosen: dict[str, Part]) -> float | None:
    """The vocals file's tempo (the vocal lab writes the detected one), else any non-default tempo, else None."""
    order = [chosen["vocals"]] if "vocals" in chosen else []
    order += [p for s, p in chosen.items() if s != "vocals"]
    for p in order:
        t = _tempo_of(p.path)
        if t and abs(t - 120.0) > 0.01:
            return t
    return None


HAT_NOTES = {42, 44, 46}   # GM closed / pedal / open hi-hat


def drum_tempo_hint(chosen: dict[str, Part]) -> float | None:
    """A tempo hint from the drum MIDI's hi-hat grid, for when no file carries a tempo.

    The beat tracker's default guess (120 BPM) can lock onto a dotted kick pattern (raio-x: 107.7
    instead of ~160). The hats usually tick on a steady grid: find its period by phase coherence
    around the median spacing (the plain median is biased by onset jitter) and read it as 8ths,
    or as 16ths/quarters when that keeps the tempo within 90-180 BPM.
    """
    if "drums" not in chosen:
        return None
    import pretty_midi
    t = np.array(sorted(n.start for i in pretty_midi.PrettyMIDI(str(chosen["drums"].path)).instruments
                        for n in i.notes if n.pitch in HAT_NOTES))
    if len(t) < 32:
        return None
    ioi = np.diff(t)
    ioi = ioi[(ioi > 0.06) & (ioi < 0.6)]
    if len(ioi) < 16:
        return None
    med = float(np.median(ioi))
    cands = np.linspace(0.97 * med, 1.03 * med, 601)
    step = float(cands[np.argmax([abs(np.exp(2j * np.pi * t / c).mean()) for c in cands])])
    bpm = 60 / (2 * step)
    while bpm > 180:
        bpm /= 2
    while bpm < 90:
        bpm *= 2
    return round(bpm, 2)


def assemble(chosen: dict[str, Part], tempo: float):
    """Merge the picked files into one PrettyMIDI with the transcriptions' own notes and velocities.

    Pitch bends are dropped. Basic Pitch writes in-tune notes with a +1/3-semitone bend, and a bend
    stays on the channel until the next one, so keeping them made most instrument notes play about
    33 cents sharp against the vocal. The MIDI-TEXT never carried bends, so it doesn't change.
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(*ASSUME_TS, 0.0))
    for stem in STEMS:
        if stem not in chosen:
            continue
        src = pretty_midi.PrettyMIDI(str(chosen[stem].path))
        inst = pretty_midi.Instrument(program=MIX[stem][0], is_drum=stem == "drums", name=stem)
        for i in src.instruments:
            inst.notes.extend(copy.deepcopy(i.notes))
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        pm.instruments.append(inst)
    return pm


def playback_mix(pm, velocity_boost: float = 1.35):
    """A copy for listening: fixed velocity for flat tracks, velocity lift, CC7 volume and pan per track."""
    import pretty_midi
    out = copy.deepcopy(pm)
    for inst in out.instruments:
        _, vol, pan = MIX.get(inst.name, (inst.program, 100, 64))
        if inst.notes and len({n.velocity for n in inst.notes}) == 1:
            for n in inst.notes:
                n.velocity = FLAT_VELOCITY
        for n in inst.notes:
            n.velocity = int(min(127, max(1, round(n.velocity * velocity_boost))))
        inst.control_changes = [pretty_midi.ControlChange(7, vol, 0.0),
                                pretty_midi.ControlChange(10, pan, 0.0)] + inst.control_changes
    return out


def without_vocals(pm):
    out = copy.deepcopy(pm)
    out.instruments = [i for i in out.instruments if (i.name or "").strip().lower() not in VOCAL_NAMES]
    return out


# --------------------------------------------------------------------------- structure (v2 notebook)

def estimate_key(y, sr) -> str:
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    if chroma.sum() == 0:
        return "C major"
    best = (-1.0, 0, "major")
    for i in range(12):
        rot = np.roll(chroma, -i)
        for prof, mode in ((_KS_MAJ, "major"), (_KS_MIN, "minor")):
            c = float(np.corrcoef(rot, prof)[0, 1])
            if c > best[0]:
                best = (c, i, mode)
    return f"{_PC[best[1]]} {best[2]}"


def _beat_grid(beats, subdiv: int) -> np.ndarray:
    grid = []
    for i in range(len(beats) - 1):
        a, b = beats[i], beats[i + 1]
        grid += [a + (b - a) * k / subdiv for k in range(subdiv)]
    if len(beats) >= 2:
        step = np.median(np.diff(beats)) / subdiv
        grid += [beats[-1] + step * k for k in range(subdiv)]
    return np.array(sorted(grid))


def estimate_chords(y, sr, beats, ts=ASSUME_TS) -> list[tuple[float, str]]:
    import librosa
    if len(beats) < 2:
        return []
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    ctimes = librosa.times_like(chroma, sr=sr, hop_length=512)
    chords = []
    for b0 in range(0, len(beats) - 1, ts[0]):
        t0, t1 = beats[b0], beats[min(b0 + ts[0], len(beats) - 1)]
        mask = (ctimes >= t0) & (ctimes < t1)
        if not mask.any():
            continue
        v = chroma[:, mask].mean(axis=1)
        if v.sum() == 0:
            chords.append((float(t0), "N.C."))
            continue
        v = v / (np.linalg.norm(v) + 1e-9)
        chords.append((float(t0), max(_TRIADS, key=lambda k: float(np.dot(v, _TRIADS[k])))))
    return chords


def analyze_structure(audio: str | Path, start_bpm: float | None = None) -> dict:
    """Beat grid, tempo, key and chords of an audio file. ``start_bpm`` guides the tracker (half/double tempo)."""
    import librosa
    y, sr = librosa.load(str(audio), sr=TARGET_SR, mono=True)
    kw = {"start_bpm": start_bpm} if start_bpm else {}
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", **kw)
    return {"tempo": float(np.atleast_1d(tempo)[0]), "beats": beats,
            "grid": _beat_grid(beats, SUBDIV) if len(beats) >= 2 else np.array([]),
            "key": estimate_key(y, sr), "chords": estimate_chords(y, sr, beats), "ts": ASSUME_TS}


def render(pm, wav: Path, sr: int = 44100) -> Path:
    """FluidSynth render (General MIDI soundfont); a sine render if FluidSynth isn't installed."""
    wav = Path(wav)
    if shutil.which("fluidsynth") and Path(SF2).exists():
        mid = wav.with_suffix(".mid")
        pm.write(str(mid))
        subprocess.run(["fluidsynth", "-ni", "-q", "-F", str(wav), "-r", str(sr), SF2, str(mid)],
                       check=True, capture_output=True)
        mid.unlink()
    else:
        import soundfile as sf
        print("⚠️ FluidSynth não está instalado: usando senoides (a bateria fica de fora).")
        sf.write(str(wav), pm.synthesize(fs=sr), sr)
    return wav


def structure_for(pm, audio: str | Path | None = None, work: Path | None = None,
                  hint: float | None = None) -> tuple[dict, str]:
    """Structure from the song's audio if given, else from a render of ``pm``. Returns it and its source.

    ``hint`` guides the beat tracker; by default it's the MIDI's own tempo.
    """
    if hint is None:
        hint = float(pm.get_tempo_changes()[1][0]) if len(pm.get_tempo_changes()[1]) else None
    if audio:
        return analyze_structure(audio, start_bpm=hint), f"áudio ({Path(audio).name})"
    work = Path(work or tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)
    wav = render(pm, work / "_structure.wav", sr=TARGET_SR)
    try:
        return analyze_structure(wav, start_bpm=hint), "render do MIDI"
    finally:
        wav.unlink(missing_ok=True)


# --------------------------------------------------------------------------- MIDI-TEXT v2

def _pitch_label(m: int, is_drum: bool) -> str:
    return _GM_DRUM.get(m, f"p{m}") if is_drum else f"{_PC[m % 12]}{m // 12 - 1}"


def _bar_pos(t: float, s: dict):
    grid = s.get("grid")
    if grid is None or len(grid) == 0:
        return "-", "-"
    per_bar = SUBDIV * s.get("ts", ASSUME_TS)[0]
    j = int(np.argmin(np.abs(grid - t)))
    return j // per_bar, j % per_bar


def midi_to_remi_text(pm, structure: dict, out_txt: str | Path, drop_vocals: bool = False) -> Path:
    insts = [i for i in pm.instruments
             if not (drop_vocals and (i.name or "").strip().lower() in VOCAL_NAMES)]
    ts = structure.get("ts", ASSUME_TS)
    lines = ["# MIDI-TEXT v2 (REMI-style) — bar-relative + exact seconds; AI-recreatable.",
             "# NOTE  bar  pos16  start_s  dur_s  pitch  midi  vel   (midi+start_s authoritative)",
             f'KEY: {structure.get("key", "?")}',
             f'TEMPO_BPM: {structure.get("tempo", 120):.2f}',
             f"TIME_SIG: {ts[0]}/{ts[1]}",
             f"DURATION_S: {pm.get_end_time():.3f}",
             f"N_TRACKS: {len(insts)}"]
    ch = structure.get("chords") or []
    if ch:
        lines.append("CHORDS: " + " | ".join(f"{t:.2f}:{n}" for t, n in ch))
    for inst in insts:
        notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
        lines += ["TRACK_BEGIN", f'NAME: {inst.name or "track"}', f"PROGRAM: {inst.program}",
                  f'IS_DRUM: {"true" if inst.is_drum else "false"}', f"N_NOTES: {len(notes)}"]
        for n in notes:
            bar, pos = _bar_pos(n.start, structure)
            lines.append(f"NOTE\t{bar}\t{pos}\t{n.start:.3f}\t{n.end - n.start:.3f}"
                         f"\t{_pitch_label(n.pitch, inst.is_drum)}\t{n.pitch}\t{n.velocity}")
        lines.append("TRACK_END")
    out_txt = Path(out_txt)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_txt


def remi_text_to_midi(txt_path: str | Path, out_mid: str | Path | None = None):
    """Lossless reconstruction from START_S/DUR_S/MIDI/VEL/IS_DRUM (the v2 notebook's ``text_to_midi``)."""
    import pretty_midi
    raw = Path(txt_path).read_text(encoding="utf-8").splitlines()
    bpm = 120.0
    for ln in raw:
        if ln.startswith("TEMPO_BPM:"):
            try:
                bpm = float(ln.split(":", 1)[1])
            except ValueError:
                pass
    pm, inst = pretty_midi.PrettyMIDI(initial_tempo=bpm), None
    for ln in raw:
        if ln.startswith("TRACK_BEGIN"):
            inst = pretty_midi.Instrument(program=0)
        elif ln.startswith("NAME:") and inst is not None:
            inst.name = ln.split(":", 1)[1].strip()
        elif ln.startswith("PROGRAM:") and inst is not None:
            try:
                inst.program = int(ln.split(":", 1)[1])
            except ValueError:
                pass
        elif ln.startswith("IS_DRUM:") and inst is not None:
            inst.is_drum = ln.split(":", 1)[1].strip().lower() == "true"
        elif ln.startswith("NOTE\t") and inst is not None:
            p = ln.split("\t")
            start, dur = float(p[3]), float(p[4])
            inst.notes.append(pretty_midi.Note(velocity=int(p[7]), pitch=int(p[6]), start=start, end=start + dur))
        elif ln.startswith("TRACK_END") and inst is not None:
            pm.instruments.append(inst)
            inst = None
    if out_mid:
        pm.write(str(out_mid))
    return pm


def check_round_trip(pm, txt_path: str | Path, drop_vocals: bool = False) -> None:
    """Raise if the text doesn't give back the same notes (to the millisecond, as written)."""
    back = remi_text_to_midi(txt_path)
    insts = [i for i in pm.instruments
             if not (drop_vocals and (i.name or "").strip().lower() in VOCAL_NAMES)]
    if len(insts) != len(back.instruments):
        raise AssertionError(f"{txt_path}: {len(insts)} tracks, text gives {len(back.instruments)}")
    for a, b in zip(insts, back.instruments):
        sa = sorted((round(n.start, 3), n.pitch, n.velocity) for n in a.notes)
        sb = sorted((round(n.start, 3), n.pitch, n.velocity) for n in b.notes)
        if sa != sb or a.name != b.name or a.is_drum != b.is_drum:
            raise AssertionError(f"{txt_path}: track {a.name} doesn't round-trip")


# --------------------------------------------------------------------------- one call

def build(chosen: dict[str, Part], out_dir: Path, song: str, audio: str | Path | None = None,
          velocity_boost: float = 1.35, overwrite: bool = False, work: Path | None = None) -> dict[str, Path]:
    """Write ``<song>.mid``, ``<song>_instrumental.mid``, ``<song>.remi.txt``, ``<song>_instrumental.remi.txt``.

    The MIDI's tempo is the vocals file's; if no file carries one, the detected
    tempo is used. Existing files are kept unless ``overwrite``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"mid": out_dir / f"{song}.mid", "instrumental_mid": out_dir / f"{song}_instrumental.mid",
             "remi": out_dir / f"{song}.remi.txt", "instrumental_remi": out_dir / f"{song}_instrumental.remi.txt"}
    todo = {k for k, p in paths.items() if overwrite or not p.exists()}
    tempo = song_tempo(chosen)
    hint = None if tempo else drum_tempo_hint(chosen)
    if hint:
        print(f"Nenhum arquivo traz o tempo: a grade do chimbal sugere {hint:.2f} BPM.")
    pm = assemble(chosen, tempo or hint or 120.0)
    structure, source = None, ""
    if todo & {"remi", "instrumental_remi"} or tempo is None:
        structure, source = structure_for(pm, audio, work, hint=hint)
        if tempo is None:   # notes are in seconds: re-assembling only changes the tick grid
            tempo = round(structure["tempo"], 2)
            pm = assemble(chosen, tempo)
    play = playback_mix(pm, velocity_boost)
    if "mid" in todo:
        play.write(str(paths["mid"]))
    if "instrumental_mid" in todo:
        without_vocals(play).write(str(paths["instrumental_mid"]))
    for key, drop in (("remi", False), ("instrumental_remi", True)):
        if key in todo:
            midi_to_remi_text(pm, structure, paths[key], drop_vocals=drop)
            check_round_trip(pm, paths[key], drop_vocals=drop)
    for k in sorted(set(paths) - todo):
        print(f"⏭️ {paths[k].name} já existe: mantido (marque OVERWRITE para substituir).")
    if structure is not None:
        print(f"Estrutura tirada do {source}: {structure['key']}, {structure['tempo']:.2f} BPM, "
              f"{len(structure['chords'])} acordes. Tempo do MIDI: {tempo:.2f} BPM.")
    return paths


def convert(midi: str | Path, out_dir: Path, song: str, audio: str | Path | None = None,
            overwrite: bool = False, work: Path | None = None) -> dict[str, Path]:
    """Already-assembled MIDI -> ``<song>.remi.txt`` (+ ``_instrumental`` when it has a vocal track). The MIDI isn't changed."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    has_vocals = any((i.name or "").strip().lower() in VOCAL_NAMES for i in pm.instruments)
    paths = {"remi": out_dir / f"{song}.remi.txt"}
    if has_vocals:
        paths["instrumental_remi"] = out_dir / f"{song}_instrumental.remi.txt"
    todo = {k for k, p in paths.items() if overwrite or not p.exists()}
    if todo:
        structure, source = structure_for(pm, audio, work)
        for key in todo:
            drop = key == "instrumental_remi"
            midi_to_remi_text(pm, structure, paths[key], drop_vocals=drop)
            check_round_trip(pm, paths[key], drop_vocals=drop)
        print(f"Estrutura tirada do {source}: {structure['key']}, {structure['tempo']:.2f} BPM, "
              f"{len(structure['chords'])} acordes.")
    for k in sorted(set(paths) - todo):
        print(f"⏭️ {paths[k].name} já existe: mantido (marque OVERWRITE para substituir).")
    if not has_vocals:
        print("(nenhuma faixa chamada vocals/voz/voice: sem versão instrumental)")
    return paths


def is_single_song(files: list[Path]) -> bool:
    """One .mid that isn't a recognisable part: an already-assembled song to convert as it is."""
    return len(files) == 1 and classify(files[0]) is None


def list_sources(folder: Path, max_depth: int = 3) -> list[Path]:
    """Where MIDIs can come from in ``folder``: .zip files, folders that hold .mid files, and loose .mid files at the top."""
    folder = Path(folder)
    zips, dirs, loose = [], [], []
    for p in folder.rglob("*"):
        rel = p.relative_to(folder).parts
        if len(rel) > max_depth or "stems" in rel or any(x.startswith(".") for x in rel):
            continue
        if p.is_file() and p.suffix.lower() == ".zip":
            zips.append(p)
        elif p.is_file() and p.suffix.lower() in (".mid", ".midi") and p.parent == folder:
            loose.append(p)
        elif p.is_dir() and any(c.is_file() and c.suffix.lower() in (".mid", ".midi") for c in p.iterdir()):
            dirs.append(p)
    key = lambda q: U._nfc(str(q.relative_to(folder)))
    return sorted(zips, key=key) + sorted(dirs, key=key) + sorted(loose, key=key)


def resolve(spec: str, folder: Path, candidates: list[Path]) -> Path | None:
    """A number or name fragment from ``candidates`` (see ``U.select_files``), or a path (absolute or inside ``folder``)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    for p in (Path(spec), Path(folder) / spec):
        if p.exists():
            return p
    hits = U.select_files(candidates, spec)
    if len(hits) > 1:
        print(f"⚠️ '{spec}' casou com {len(hits)} itens; usando o primeiro: {hits[0].name}")
    return hits[0]


def process(source: Path, out_root: Path, song: str | None = None, audio: str | Path | None = None,
            drums: str = "adt_str", velocity_boost: float = 1.35, overwrite: bool = False,
            work: Path | None = None) -> dict[str, Path]:
    """Folder/zip of parts -> assembled MIDIs + texts; a single assembled MIDI -> texts only."""
    work = Path(work or tempfile.mkdtemp())
    files = gather(Path(source), work)
    if not files:
        raise FileNotFoundError(f"Nenhum .mid em {source}")
    if is_single_song(files):
        song = song or U.safe_stem(files[0])
        print(f"🎼 {files[0].name}: já montado, só gerando o texto.")
        return convert(files[0], Path(out_root) / song, song, audio, overwrite, work)
    chosen, notes = pick(files, drums)
    if not chosen:
        raise ValueError("Nenhum arquivo reconhecido como parte (vocals, guitar, piano, bass, other, drums).")
    for n in notes:
        print("  ", n)
    print()
    for stem in STEMS:
        p = chosen.get(stem)
        print(f"{stem:7s} {p.path.name}  ({p.source})" if p else f"{stem:7s} —")
    song = song or default_song_name(chosen, Path(source))
    print(f"\n🎵 {song}")
    return build(chosen, Path(out_root) / song, song, audio, velocity_boost, overwrite, work)


def main(argv: list[str] | None = None) -> None:
    """Local use: ``python -m modules.remi <folder|zip|file.mid> <out_dir> [--song NAME] [--audio FILE] [--drums adtof]``."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("out_dir")
    ap.add_argument("--song")
    ap.add_argument("--audio")
    ap.add_argument("--drums", default="adt_str", choices=DRUM_ENGINES)
    ap.add_argument("--velocity-boost", type=float, default=1.35)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    paths = process(Path(a.source), Path(a.out_dir), a.song, a.audio, a.drums, a.velocity_boost, a.overwrite)
    for k, p in paths.items():
        print(f"{k:18s} {p}")


def default_song_name(chosen: dict[str, Part], source: Path) -> str:
    """The external vocals file's name without '_vocals', else the source folder/zip name."""
    v = chosen.get("vocals")
    name = _VOCAL_FILE.sub("", _clean_stem(v.path)).strip(" _-.") if v and v.source == "voz externa" else ""
    name = name or (Path(source).stem if Path(source).suffix else Path(source).name)
    return U.safe_stem(Path(f"{name}.mid"))


if __name__ == "__main__":
    main()
