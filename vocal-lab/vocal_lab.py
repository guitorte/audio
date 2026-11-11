"""Vocal Stem Lab: isolate a vocal, clean it, rebuild its highs, turn it into MIDI, analyse and chop it.

One "active" file flows through the modules. Each module reads it, writes its
results to ``<project>/<stage>/`` and, unless told otherwise, makes its main
result the new active file. ``route()`` goes back to any earlier stage.

Where each engine runs, and why:

* audio-separator (vocal split, lead/backing, de-reverb, denoise) runs as a
  separate process. Its memory is returned when it exits, which matters
  because AudioSR needs most of Colab's system RAM just to load.
* AudioSR runs in neural-upscaler's Python 3.10 venv, through its worker, so
  the upscale is the same Mid/Side splice: only the band above the cutoff is new.
* GAME (singing voice to MIDI) pins numpy<2, so it gets a Python 3.12 venv of
  its own, with CPU-only torch: it transcribes a song in a few minutes without
  a GPU and the install is ten times smaller.
* Beat This (tempo) imports next to Colab's own packages and runs in the kernel.
"""
from __future__ import annotations

import collections
import contextlib
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "neural-upscaler"))
import upscaler as U  # noqa: E402
from upscaler_dsp import bandwidth, detect_cutoff  # noqa: E402

WORK = Path("/content/vocal_lab") if Path("/content").exists() else HERE / "_work"
DRIVE_BASE = Path("/content/drive/MyDrive/audio/outputs")
SR = 44_100
SYNCED = (".wav", ".mid", ".csv", ".zip", ".json")

GAME_REPO = "https://github.com/openvpi/GAME.git"
GAME_COMMIT = "69ba01ded0ddda5495e51ba373685f1be1a67ea2"
GAME_WEIGHTS = "https://github.com/openvpi/GAME/releases/download/v1.0.0/GAME-1.0-{size}.zip"
GAME_SIZES = ("small", "medium", "large")       # about 47 / 185 / 366 MB
GAME_LANGUAGES = ("en", "ja", "yue", "zh")      # anything else: leave the language empty

# What each kind of separation keeps, and how the model's stem names are filed.
TASKS = {
    "vocals": {"folder": "vocals", "keep": "vocals",
               "names": {"vocals": "vocals", "instrumental": "instrumental", "other": "instrumental"}},
    "lead": {"folder": "lead", "keep": "lead",
             "names": {"vocals": "lead", "instrumental": "backing", "other": "backing"}},
    "dereverb": {"folder": "dry", "keep": "dry",
                 "names": {"noreverb": "dry", "no reverb": "dry", "dry": "dry", "reverb": "reverb", "no dry": "reverb"}},
    "denoise": {"folder": "clean", "keep": "clean",
                "names": {"dry": "clean", "no noise": "clean", "other": "noise", "noise": "noise"}},
}

# Menu label -> (task, audio-separator model file). Every file here is in
# audio-separator 0.48's registry with known stem names.
MODELS = {
    "Vocals · BS-RoFormer by viperx": ("vocals", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
    "Vocals · Mel-RoFormer Big Beta 5e by unwa (fuller vocal)": ("vocals", "melband_roformer_big_beta5e.ckpt"),
    "Vocals · Demucs v4 fine-tuned (also drums, bass, other)": ("vocals", "htdemucs_ft.yaml"),
    "Lead vs backing · Mel-RoFormer Karaoke by aufr33 & viperx": ("lead", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"),
    "Lead vs backing · UVR MDX-Net Karaoke 2 (older)": ("lead", "UVR_MDXNET_KARA_2.onnx"),
    "De-reverb · Mel-RoFormer by anvuew": ("dereverb", "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"),
    "De-reverb · Mel-RoFormer by anvuew, less aggressive": ("dereverb", "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt"),
    "De-reverb · UVR DeEcho-DeReverb (older)": ("dereverb", "UVR-DeEcho-DeReverb.pth"),
    "Denoise · Mel-RoFormer by aufr33": ("denoise", "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"),
    "Denoise · Mel-RoFormer by aufr33, aggressive": ("denoise", "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt"),
}

# Router menu label -> stage folder.
STAGES = {"Vocals": "vocals", "Lead vocal": "lead", "De-reverbed": "dry", "Upscaled": "upscaled", "Denoised": "clean"}


# --------------------------------------------------------------------------- project

@dataclass
class Lab:
    project: str
    out: Path                     # local outputs, one folder per stage
    work: Path                    # scratch: the active file, temporary files
    drive: Path | None            # mirror on Google Drive, or None when Drive is not mounted
    state: dict = field(default_factory=dict)

    @property
    def active(self) -> Path:
        return self.work / "active.wav"

    @property
    def original(self) -> Path:
        return self.work / "original.wav"

    @property
    def name(self) -> str:
        """Descriptive name of the active file, e.g. ``Beterraba_vocals_dry``. Outputs are named after it."""
        return self.state.get("active_name", "audio")

    def save(self) -> None:
        (self.out / "lab.json").write_text(json.dumps(self.state, ensure_ascii=False, indent=2))


def start(project: str, drive_base: Path = DRIVE_BASE) -> Lab:
    """Open (or create) a project. Outputs go to ``<drive_base>/<project>/`` when Drive is mounted."""
    project = project.strip()
    bad = sorted(set(project) & set("\\/:*?\"<>|'"))
    if not project or project in (".", "..") or bad or not project.isprintable():
        raise ValueError(f"Invalid project name {project!r}. Use a simple folder name such as 'my-song', "
                         "without any of these characters: \\ / : * ? \" < > | '")
    mounted = os.path.ismount("/content/drive") or not Path("/content").exists()
    lab = Lab(project, WORK / "out" / project, WORK / "tmp" / project, Path(drive_base) / project if mounted else None)
    for folder in (lab.out, lab.work, *(lab.out / s for s in (*STAGES.values(), "midi", "slices"))):
        folder.mkdir(parents=True, exist_ok=True)
    if lab.drive:
        lab.drive.mkdir(parents=True, exist_ok=True)
    # The project's memory: which file each stage produced. A new Colab session
    # has an empty disk, so fall back to the copy kept on Drive.
    saved = [p for p in (lab.out / "lab.json", lab.drive / "lab.json" if lab.drive else None) if p and p.exists()]
    lab.state = json.loads(saved[0].read_text()) if saved else {"stages": {}}
    print(f"📁 Project: {project}\n   ↳ local outputs: {lab.out}")
    print(f"   ↳ Drive folder:  {_drive_label(lab)}/" if lab.drive else
          "   ↳ ⚠️ Google Drive is not mounted: outputs stay on the Colab disk and are lost when the session ends")
    if lab.active.exists():
        print(f"   ↳ active file:   {lab.name}")
    elif lab.state.get("stages"):
        print("   ↳ this project has earlier results on Drive: use the Router to pick one up, or load a song again")
    return lab


def _drive_label(lab: Lab) -> str:
    try:
        return "MyDrive/" + lab.drive.relative_to("/content/drive/MyDrive").as_posix()
    except ValueError:
        return str(lab.drive)


def specs(path: Path) -> str:
    info = sf.info(str(path))
    return f"{info.samplerate} Hz | {info.channels} ch | {info.subtype} | {info.duration:.2f} s"


def _channels(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    return int(out.split(",")[0]) if out.split(",")[0].isdigit() else 2


def load(lab: Lab, source: str | Path) -> Path:
    """Bring a song into the project as a 44.1 kHz 24-bit WAV. The source file is never changed."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"File not found: '{source}'. Check the path, or untick use_drive_file to upload one.")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-vn",
                    "-ac", "1" if _channels(source) == 1 else "2", "-ar", str(SR), "-c:a", "pcm_s24le",
                    str(lab.original)], check=True)
    lab.state.update(source=str(source), source_name=U.safe_stem(source), stages={})
    lab.state.pop("bpm", None)
    print(f"📂 Loaded: {source}")
    return _set_active(lab, lab.original, lab.state["source_name"])


def _set_active(lab: Lab, path: Path, name: str) -> Path:
    if Path(path).resolve() != lab.active.resolve():
        shutil.copy2(path, lab.active)
    lab.state["active_name"] = name
    lab.save()
    print(f"🎯 Active file: {name}\n   ↳ {specs(lab.active)}")
    return lab.active


def _record(lab: Lab, stage: str, path: Path, name: str, advance: bool) -> None:
    lab.state.setdefault("stages", {})[stage] = {"path": str(path.relative_to(lab.out)), "name": name}
    lab.state["latest"] = stage
    lab.save()
    if advance:
        _set_active(lab, path, name)
    else:
        print("   (active file unchanged; use the router to pick this result)")


def route(lab: Lab, target: str = "Latest output", custom_path: str = "") -> Path:
    """Choose which file the next module works on: an earlier result, the original, or any file."""
    if target == "Original input":
        if not lab.original.exists():
            raise FileNotFoundError("No song loaded yet: run the Load cell first.")
        return _set_active(lab, lab.original, lab.state.get("source_name", "audio"))
    if target == "Custom path":
        path = Path(custom_path)
        if not path.exists():
            raise FileNotFoundError(f"Custom path does not exist: {custom_path}")
        if path.suffix.lower() != ".wav":
            raise ValueError("The custom path must be a WAV file. Use the Load cell for other formats.")
        return _set_active(lab, path, U.safe_stem(path))
    stage = lab.state.get("latest") if target == "Latest output" else STAGES[target]
    entry = lab.state.get("stages", {}).get(stage)
    if not entry:
        print(f"⚠️ Nothing has been made for '{target}' yet. The active file stays as it is.")
        return lab.active
    for base in (lab.out, lab.drive):
        if base and (base / entry["path"]).exists():
            return _set_active(lab, base / entry["path"], entry["name"])
    print(f"⚠️ The file for '{target}' is gone ({entry['path']}). The active file stays as it is.")
    return lab.active


def sync(lab: Lab) -> None:
    """Copy every new or changed output to Google Drive."""
    if not lab.drive:
        print(f"\n⚠️ Google Drive is not mounted, upload skipped. Outputs are only in {lab.out}.")
        return
    uploaded, failed = [], []
    for src in sorted(p for p in lab.out.rglob("*") if p.is_file() and p.suffix.lower() in SYNCED):
        rel = src.relative_to(lab.out)
        dst = lab.drive / rel
        if rel.as_posix() == "lab.json":   # bookkeeping: keep Drive's copy current without reporting it
            with contextlib.suppress(OSError):
                shutil.copy2(src, dst)
            continue
        try:
            if dst.exists():
                a, b = src.stat(), dst.stat()
                # Same size AND not newer: unchanged. A rerun with other settings
                # often writes the same name and size, so size alone is not enough.
                if a.st_size == b.st_size and a.st_mtime <= b.st_mtime + 2:
                    continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            uploaded.append(rel)
        except OSError as err:   # a Drive hiccup must not make a finished module look failed
            failed.append((rel, err))
    if uploaded:
        print(f"\n☁️ Uploaded {len(uploaded)} file(s) to Drive → {_drive_label(lab)}/")
        for rel in uploaded[:12]:
            print(f"   ↳ {rel}")
        if len(uploaded) > 12:
            print(f"   ↳ … and {len(uploaded) - 12} more")
    elif not failed:
        print("\n☁️ Drive already up to date.")
    for rel, err in failed:
        print(f"   ⚠️ Could not upload {rel}: {err}")


def status(lab: Lab) -> None:
    files = sorted((p for p in lab.out.rglob("*") if p.is_file() and p.suffix.lower() in (".wav", ".mid", ".zip")
                    and "slices" not in p.relative_to(lab.out).parts[:-1]),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    print("\n📊 Outputs so far:" + ("" if files else "\n   (none yet)"))
    for p in files:
        print(f"   • [{p.parent.name}] {p.name} ({p.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------- helpers

def _need(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        U.run([sys.executable, "-m", "pip", "install", "-q", package])
        importlib.invalidate_caches()


def _uv() -> None:
    if shutil.which("uv") is None:
        U.run([sys.executable, "-m", "pip", "install", "-q", "uv"])


def _write24(path: Path, audio: np.ndarray, sr: int) -> Path:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    return U.write(path, audio * (0.999 / peak) if peak > 0.999 else audio, sr, "PCM_24")


# --------------------------------------------------------------------------- separation

def _stem(path: Path) -> str:
    """audio-separator names its files ``<input>_(<Stem>)_<model>.wav``."""
    found = re.findall(r"_\(([^()]+)\)_", path.name)
    return found[-1].lower() if found else path.stem.lower()


def run_separator(src: Path, model: str, out_dir: Path) -> list[Path]:
    """One audio-separator run in its own process. Replaced by a fake in tests."""
    _need("audio_separator", "audio-separator[gpu]>=0.47")
    U.run([sys.executable, "-c", "from audio_separator.utils.cli import main; main()", src, "-m", model,
           "--output_dir", out_dir, "--output_format", "WAV", "--model_file_dir", WORK / "models",
           "--normalization", "1.0", "--log_level", "warning"])
    return sorted(Path(out_dir).glob("*.wav"))


def separate(lab: Lab, label: str, advance: bool = True) -> dict:
    """Vocal split, lead/backing split, de-reverb or denoise, depending on the model picked from ``MODELS``."""
    task, model = MODELS[label]
    spec = TASKS[task]
    tmp = lab.work / "sep"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    src = tmp / "in.wav"
    shutil.copy2(lab.active, src)
    print(f"🚀 {label}\n   on: {lab.name}")
    if sf.info(str(lab.active)).samplerate > SR:
        print(f"   ⚠️ these models work at {SR / 1000:g} kHz, so anything above 22 kHz in this file is dropped. "
              "Run the upscaler after this module, not before.")
    produced = run_separator(src, model, tmp / "out")
    if not produced:
        raise RuntimeError("The separator wrote no files. The lines printed above explain why.")
    saved, base = {}, lab.name
    for path in produced:
        kind = spec["names"].get(_stem(path), _stem(path).replace(" ", "-"))
        dest = lab.out / spec["folder"] / f"{base}_{kind}.wav"
        shutil.move(str(path), dest)
        saved[kind] = dest
        print(f"   ↳ {kind}: {dest.name}")
    main = saved.get(spec["keep"])
    if main is None:
        print(f"⚠️ This model gave {sorted(saved)} and none of them is the '{spec['keep']}' part I expected. "
              "The files are saved; pick one with the router's Custom path.")
    else:
        _record(lab, spec["folder"], main, main.stem, advance)
    sync(lab)
    return saved


# --------------------------------------------------------------------------- upscale

def upscale(lab: Lab, engine: str = "AudioSR", steps: int = 50, cutoff_hz: float = 0.0, advance: bool = True):
    """Rebuild the highs above the point where they were cut. Below it the audio stays the original.

    ``engine``: "AudioSR" (slow, proven) or "FlashSR" (one step, experimental).
    Returns the new file, or None when there is nothing to rebuild.
    """
    tmp = lab.work / "sr"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mono = sf.info(str(lab.active)).channels == 1
    print(f"⚡ {engine} on: {lab.name}")
    if engine.startswith("FlashSR"):
        import flashsr_trial as T
        audio = U.read(U.decode(lab.active, tmp / "in48.wav", T.SR))[0]
        cut = float(cutoff_hz) or detect_cutoff(audio, T.SR)
        if cut >= 0.95 * T.SR / 2:
            cut = None
        else:
            if cut > T.TRAINED_UP_TO_HZ:
                print(f"   ⚠️ the highs already reach {cut / 1000:.1f} kHz; FlashSR was trained on audio cut at "
                      f"{T.TRAINED_UP_TO_HZ / 1000:.0f} kHz or lower, so expect little.")
            model, device = T.load(T.setup())
            audio = T.enhance(T.make_infer(model, device, cut), audio, cut)
            del model
            T._release_memory()
    else:
        job = {"in": str(lab.active), "out": str(tmp / "audiosr.wav"), "cutoff_hz": float(cutoff_hz)}
        U.audiosr(U.setup_audiosr(), [job], int(steps), 3.5, 42)
        meta = Path(job["out"]).with_suffix(".json")
        cut = json.loads(meta.read_text()).get("cutoff_hz") if meta.exists() else None
        audio = U.read(Path(job["out"]))[0] if cut else None
    if not cut:
        original, sr = U.read(lab.active)
        top = bandwidth(original, sr)
        print("⏭  Nothing to rebuild: this file has no sharp cut in its highs."
              + (f" It is nearly empty above {top / 1000:.1f} kHz, though; set CUTOFF_HZ = {int(top)} to rebuild from there."
                 if top < U.HINT_BELOW_HZ else " Set CUTOFF_HZ (e.g. 16000) to force it."))
        return None
    if mono:
        audio = audio.mean(axis=1, keepdims=True)
    name = f"{lab.name}_48k"
    dest = _write24(lab.out / "upscaled" / f"{name}.wav", audio, 48_000)
    print(f"🎉 New highs above {cut / 1000:.1f} kHz → {dest.name}\n   ↳ {specs(dest)}")
    _record(lab, "upscaled", dest, name, advance)
    sync(lab)
    return dest


# --------------------------------------------------------------------------- MIDI

def setup_game(size: str = "medium") -> tuple[Path, Path, Path]:
    """GAME's venv (Python 3.12, CPU torch), pinned clone and weights. Returns (python, repo, model)."""
    if size not in GAME_SIZES:
        raise ValueError(f"GAME model size must be one of {GAME_SIZES}")
    repo, env = WORK / "GAME", WORK / "envs" / "game"
    python = env / "bin" / "python"
    if not (repo / "infer.py").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        U.run(["git", "clone", "-q", GAME_REPO, repo])
        U.run(["git", "-C", repo, "checkout", "-q", GAME_COMMIT])
    if not python.exists():
        _uv()
        print("🛠️ Setting up GAME's own Python environment (once per session, about a minute)…")
        U.run(["uv", "venv", "-q", "--python", "3.12", env])
        U.run(["uv", "pip", "install", "-q", "--python", python, "torch", "--index-url",
               "https://download.pytorch.org/whl/cpu"])
        U.run(["uv", "pip", "install", "-q", "--python", python, "-r", repo / "requirements.txt"])
    model = WORK / "models" / "game" / f"GAME-1.0-{size}" / "model.pt"
    if not model.exists():
        model.parent.parent.mkdir(parents=True, exist_ok=True)
        archive = model.parent.parent / f"{size}.zip"
        print(f"   downloading the GAME {size} model…", flush=True)
        urllib.request.urlretrieve(GAME_WEIGHTS.format(size=size), archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(model.parent.parent)
        archive.unlink()
    return python, repo, model


def run_game(wav: Path, out_dir: Path, size: str, language: str, tempo: float) -> None:
    """One GAME transcription. Replaced by a fake in tests."""
    python, repo, model = setup_game(size)
    U.run([python, "infer.py", "extract", wav, "-m", model, "--output-dir", out_dir,
           "--output-formats", "mid,csv", "--tempo", f"{tempo:g}", *(["-l", language] if language else [])],
          cwd=repo, env=U.venv_env())


NOTE = re.compile(r"^([A-G]#?)(-?\d+)")
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def transcribe(lab: Lab, size: str = "medium", language: str = "") -> Path:
    """Sung melody -> MIDI with GAME, a transcriber built for the singing voice (one note at a time).

    Works best on an isolated lead vocal. ``language`` helps note boundaries when
    it is one of ``GAME_LANGUAGES``; leave it empty otherwise. The MIDI is saved at
    the tempo found by ``analyze()`` when that has been run, else at 120 BPM.
    """
    language = language.strip().lower()
    if language and language not in GAME_LANGUAGES:
        print(f"   (GAME has no '{language}' setting; using its language-neutral mode. Options: {GAME_LANGUAGES})")
        language = ""
    tmp = lab.work / "midi"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    wav = tmp / f"{lab.name}.wav"
    shutil.copy2(lab.active, wav)
    tempo = float(lab.state.get("bpm") or 120)
    print(f"🎹 Transcribing {lab.name} with GAME ({size}), MIDI tempo {tempo:g} BPM…")
    run_game(wav, lab.out / "midi", size, language, tempo)
    midi, table = lab.out / "midi" / f"{lab.name}.mid", lab.out / "midi" / f"{lab.name}.csv"
    if not midi.exists():
        raise RuntimeError("GAME finished without writing a MIDI file. The lines printed above explain why.")
    notes = []
    if table.exists():
        with open(table, newline="") as fh:
            for row in csv.DictReader(fh):
                m = NOTE.match(row.get("pitch", ""))
                if m:
                    notes.append((float(row["onset"]), float(row["offset"]), m.group(1), int(m.group(2))))
    print(f"\n📄 {midi.name}: {len(notes)} notes")
    if notes:
        number = [PITCH_CLASSES.index(n) + 12 * (o + 1) for _, _, n, o in notes]
        low, high = notes[int(np.argmin(number))], notes[int(np.argmax(number))]
        print(f"   ↳ range {low[2]}{low[3]} → {high[2]}{high[3]}, last note ends at {max(n[1] for n in notes):.1f} s")
        hist = collections.Counter(n[2] for n in notes)
        for pc in PITCH_CLASSES:
            print(f"      {pc:>2}: {'█' * int(30 * hist[pc] / max(hist.values()))}")
    else:
        print("   ⚠️ No notes found. Check that the active file has an audible sung melody.")
    sync(lab)
    return midi


# --------------------------------------------------------------------------- analysis

def find_beats(path: Path) -> np.ndarray:
    """Beat times in seconds, from Beat This. Replaced by a fake in tests."""
    _need("beat_this", "beat-this")
    import torch
    from beat_this.inference import File2Beats
    tracker = File2Beats(checkpoint_path="final0", device="cuda" if torch.cuda.is_available() else "cpu", dbn=False)
    return np.asarray(tracker(str(path))[0], dtype=float)


MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def analyze(lab: Lab, source: str = "Original input") -> dict:
    """Tempo, key and loudness. Run it on the full song: a bare vocal has no drums to count beats from."""
    import librosa
    path = lab.original if source == "Original input" else lab.active
    label = lab.state.get("source_name", "audio") if source == "Original input" else lab.name
    print(f"🔍 Analysing: {label}")
    res = {"file": label}
    try:
        beats = find_beats(path)
        if len(beats) < 4:
            raise ValueError(f"only {len(beats)} beats found")
        res["bpm"], res["tempo_engine"] = round(60 / float(np.median(np.diff(beats))), 1), "Beat This"
    except Exception as err:   # tempo is a convenience here: fall back, do not stop the cell
        print(f"   (Beat This did not work: {str(err).splitlines()[0][:120]}. Using librosa's estimate.)")
        y, sr = librosa.load(str(path), sr=22_050, mono=True)
        res["bpm"] = round(float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0]), 1)
        res["tempo_engine"] = "librosa"
    print(f"⏱️ Tempo: {res['bpm']:g} BPM  (half {res['bpm'] / 2:g} | double {res['bpm'] * 2:g})  [{res['tempo_engine']}]")

    y, sr = librosa.load(str(path), sr=22_050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    scores = sorted(((float(np.corrcoef(chroma, np.roll(profile, i))[0, 1]), f"{PITCH_CLASSES[i]} {mode}")
                     for i in range(12) for profile, mode in ((MAJOR, "major"), (MINOR, "minor"))), reverse=True)
    res["key_candidates"] = [name for _, name in scores[:3]]
    print("🔑 Key candidates: " + " | ".join(f"{i}. {n}" for i, n in enumerate(res["key_candidates"], 1)))

    log = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    summary = log[log.rfind("Summary:"):]
    for key, pattern in (("lufs", r"I:\s*(-?[\d.]+) LUFS"), ("loudness_range_lu", r"LRA:\s*(-?[\d.]+) LU"),
                         ("true_peak_dbfs", r"Peak:\s*(-?[\d.]+) dBFS")):
        m = re.search(pattern, summary)
        if m:
            res[key] = float(m.group(1))
    print(f"🎚️ Loudness: {res.get('lufs', '?')} LUFS integrated | range {res.get('loudness_range_lu', '?')} LU | "
          f"true peak {res.get('true_peak_dbfs', '?')} dBFS")
    if source == "Original input":
        lab.state["bpm"] = res["bpm"]   # transcribe() saves its MIDI at this tempo
    lab.save()
    (lab.out / f"{label}_analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    sync(lab)
    return res


# --------------------------------------------------------------------------- chopper

def chop(lab: Lab, silence_threshold_db: float = 32, min_phrase_sec: float = 0.35, padding_sec: float = 0.05) -> Path:
    """Cut the active file at its silences into phrases, with short fades, and zip them as a sample pack."""
    import librosa
    audio, sr = sf.read(str(lab.active), always_2d=True)
    folder = lab.out / "slices" / lab.name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    pad, fade, kept = int(padding_sec * sr), int(0.008 * sr), 0
    for start, end in librosa.effects.split(audio.mean(axis=1), top_db=silence_threshold_db):
        start, end = max(0, start - pad), min(len(audio), end + pad)
        if (end - start) / sr < min_phrase_sec:
            continue
        piece = audio[start:end].copy()
        n = min(fade, len(piece) // 2)   # 8 ms fades stop clicks at the cuts
        ramp = np.linspace(0, 1, n)[:, None]
        piece[:n] *= ramp
        piece[len(piece) - n:] *= ramp[::-1]
        kept += 1
        sf.write(str(folder / f"slice_{kept:03d}_{start / sr:07.2f}s.wav"), piece, sr, subtype="PCM_24")
    archive = shutil.make_archive(str(lab.out / f"{lab.name}_sample_pack"), "zip", folder)
    print(f"✂️ {kept} phrases from {lab.name}\n🗜️ {Path(archive).name}")
    sync(lab)
    return Path(archive)
