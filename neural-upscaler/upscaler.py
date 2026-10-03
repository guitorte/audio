"""Neural audio upscaler: RoFormer vocal split -> Apollo -> AudioSR -> Matchering.

Runs in the main Colab environment. AudioSR pins old libraries (numpy 1.23,
transformers 4.30) that cannot live next to modern Colab packages, so it runs
in its own Python 3.10 venv through ``audiosr_worker.py``.

Every stage is optional; the pipeline keeps the recording itself and only
repairs it, so a stage that is switched off simply passes audio through.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upscaler_dsp import detect_cutoff, rms  # noqa: E402

HERE = Path(__file__).resolve().parent
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aif", ".aiff", ".wma", ".alac"}
APOLLO_SR = 44_100
APOLLO_COMMIT = "e84bcacc59d5455f05d86a5c97dd4aeb3c14dbb6"
VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
WORK = Path("/content/upscaler_work") if Path("/content").exists() else HERE / "_work"
APOLLO_DIR = WORK / "Apollo"
AUDIOSR_ENV = WORK / "envs" / "audiosr"


# --------------------------------------------------------------------------- files

def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def find_folder(name: str, root: str = "/content/drive/MyDrive") -> Path:
    """Resolve ``name`` under ``root``, tolerant of accent encoding and case.

    Drive may store "áudio" precomposed (NFC) or decomposed (NFD); a plain
    string comparison would miss one of the two.
    """
    current = Path(root)
    for part in [p for p in re.split(r"[\\/]+", name.strip()) if p]:
        exact = current / part
        if exact.is_dir():
            current = exact
            continue
        matches = [c for c in current.iterdir() if c.is_dir() and _nfc(c.name) == _nfc(part)]
        if not matches:
            siblings = sorted(c.name for c in current.iterdir() if c.is_dir())
            raise FileNotFoundError(f"Folder '{part}' not found in {current}. Folders there: {siblings}")
        current = matches[0]
    return current


def list_audio(folder: Path) -> list[Path]:
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS),
                  key=lambda p: _nfc(p.name))


def select_files(files: list[Path], spec: str) -> list[Path]:
    """``all`` | ``1, 3, 5-7`` (numbers from the listing) | name fragments ``know better; demo``."""
    spec = (spec or "").strip()
    if not spec or spec.lower() == "all":
        return list(files)
    chosen: list[Path] = []
    for token in [t.strip() for t in re.split(r"[;,]", spec) if t.strip()]:
        if re.fullmatch(r"\d+(\s*-\s*\d+)?", token):
            lo, _, hi = token.partition("-")
            for i in range(int(lo), int(hi or lo) + 1):
                if not 1 <= i <= len(files):
                    raise ValueError(f"No file number {i}; the listing has {len(files)} files")
                chosen.append(files[i - 1])
        else:
            hits = [f for f in files if _nfc(token) in _nfc(f.name)]
            if not hits:
                raise ValueError(f"No file name contains '{token}'")
            chosen.extend(hits)
    return list(dict.fromkeys(chosen))


def safe_stem(path: Path) -> str:
    return re.sub(r"[^\w\-. ]+", "_", unicodedata.normalize("NFC", path.stem)).strip() or "track"


# --------------------------------------------------------------------------- audio io

def decode(src: Path, dst: Path, sr: int = APOLLO_SR) -> Path:
    """Any format -> stereo float32 WAV at ``sr`` (Apollo needs exactly 44.1 kHz)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn",
                    "-ac", "2", "-ar", str(sr), "-c:a", "pcm_f32le", str(dst)], check=True)
    return dst


def read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio, sr


def write(path: Path, audio: np.ndarray, sr: int, subtype: str = "FLOAT") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype=subtype)
    return path


def fit_length(audio: np.ndarray, frames: int) -> np.ndarray:
    if len(audio) >= frames:
        return audio[:frames]
    return np.pad(audio, ((0, frames - len(audio)), (0, 0)))


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def excerpt(path: Path, start: float, seconds: float, sr: int) -> np.ndarray:
    """Stereo float excerpt [samples, 2] for listening; the window slides back to fit short songs."""
    start = max(0.0, min(start, duration(path) - seconds))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(seconds), "-i", str(path),
                          "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)


def db(x: float) -> float:
    return 20 * np.log10(max(float(x), 1e-12))


# --------------------------------------------------------------------------- setup

def run(cmd: list[str], **kw) -> None:
    """Run a command and stream its output into the notebook.

    Colab only shows what the kernel itself prints; a child process writing to
    its own stdout/stderr is invisible, so failures would surface as a bare
    "exit status 1". Pipe everything through ``print`` and keep the tail for
    the error message.
    """
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    tail: list[str] = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          errors="replace", bufsize=1, **kw) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail = (tail + [line.rstrip()])[-25:]
    if proc.returncode:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed (exit {proc.returncode}). Last output:\n"
                           + "\n".join(tail))


def venv_env() -> dict:
    """Environment for programs running in the AudioSR venv.

    The Colab kernel exports variables that only make sense inside it:
    MPLBACKEND points at the notebook's inline backend, and PYTHONPATH /
    PYTHONHOME can leak the kernel's own site-packages into another Python.
    """
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "MPLBACKEND")}
    env["MPLBACKEND"] = "Agg"
    return env


def setup_main(want_separation: bool, want_apollo: bool, want_matchering: bool) -> None:
    pkgs = []
    if want_separation:
        pkgs.append("audio-separator[gpu]>=0.47")
    if want_matchering:
        pkgs.append("matchering>=2.0.6")
    if pkgs:
        run([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    if want_apollo and not (APOLLO_DIR / "inference.py").exists():
        APOLLO_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "-q", "https://github.com/JusperLee/Apollo.git", APOLLO_DIR])
        run(["git", "-C", APOLLO_DIR, "checkout", "-q", APOLLO_COMMIT])


AUDIOSR_PINS = ["audiosr==0.0.7", "torch==2.5.1", "torchaudio==2.5.1", "torchvision==0.20.1",
                "setuptools<70",   # librosa 0.9.2 still imports pkg_resources
                "matplotlib"]      # imported by audiosr.utilities but not declared


def setup_audiosr() -> Path:
    """Isolated Python 3.10 env for AudioSR. Takes a few minutes the first time."""
    python = AUDIOSR_ENV / "bin" / "python"
    if not python.exists():
        if shutil.which("uv") is None:
            run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        run(["uv", "venv", "-q", "--python", "3.10", AUDIOSR_ENV])
    # Always re-applied: a no-op when satisfied, and it repairs an env built by
    # an older version of this list (e.g. one missing matplotlib).
    run(["uv", "pip", "install", "-q", "--python", python, *AUDIOSR_PINS])
    return python
    if shutil.which("uv") is None:
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run(["uv", "venv", "-q", "--python", "3.10", AUDIOSR_ENV])
    run(["uv", "pip", "install", "-q", "--python", python, *AUDIOSR_PINS])
    return python


# --------------------------------------------------------------------------- stages

def separate_vocals(mix_wav: Path, out_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """BS-RoFormer vocals; instrumental is the exact residual ``mix - vocals``.

    Using the residual (rather than the model's own instrumental) guarantees
    vocals + instrumental == original mix, so nothing is lost by splitting.
    """
    from audio_separator.separator import Separator
    mix, sr = read(mix_wav)
    peak = float(np.abs(mix).max()) or 1.0
    # Headroom so audio-separator's peak normalisation never rescales a stem.
    scale = 0.5 / peak
    scaled = write(out_dir / "mix_scaled.wav", mix * scale, sr)
    sep = Separator(output_dir=str(out_dir), output_format="WAV", normalization_threshold=1.0,
                    sample_rate=sr, model_file_dir=str(WORK / "models"), output_single_stem="Vocals")
    sep.load_model(VOCAL_MODEL)
    sep.separate(str(scaled), custom_output_names={"Vocals": "vocals"})
    vocals, vsr = read(out_dir / "vocals.wav")
    assert vsr == sr, f"separator returned {vsr} Hz, expected {sr} Hz"
    vocals = fit_length(vocals, len(mix)) / scale
    return vocals, mix - vocals, sr


def apollo(src: Path, dst: Path) -> Path:
    run([sys.executable, APOLLO_DIR / "inference.py", "--in_wav", src, "--out_wav", dst,
         "--chunk-seconds", "6", "--overlap-seconds", "1", "--chunk-pad-seconds", "1",
         "--chunk-batch-size", "2"], cwd=APOLLO_DIR)
    return dst


def audiosr(python: Path, jobs: list[dict], steps: int, guidance: float, seed: int) -> None:
    jobs_file = WORK / "audiosr_jobs.json"
    jobs_file.write_text(json.dumps(jobs))
    run([python, HERE / "audiosr_worker.py", "--jobs", jobs_file, "--steps", steps,
         "--guidance", guidance, "--seed", seed], env=venv_env())


def master(target: Path, reference: Path, dst: Path) -> Path:
    import matchering as mg
    mg.process(target=str(target), reference=str(reference), results=[mg.pcm24(str(dst))])
    return dst


# --------------------------------------------------------------------------- pipeline

@dataclass
class Options:
    separate: bool = True
    apollo: bool = True
    audiosr: bool = False
    audiosr_steps: int = 50
    audiosr_guidance: float = 3.5
    seed: int = 42
    cutoff_hz: float = 0.0          # 0 = detect automatically
    vocal_gain_db: float = 0.0
    reference: str = ""             # file in the folder to master against (Matchering)
    keep_parts: bool = False
    overwrite: bool = False
    out_subdir: str = "upscaled"


@dataclass
class Result:
    source: str
    output: str = ""
    skipped: str = ""
    notes: list[str] = field(default_factory=list)


def process(files: list[Path], folder: Path, opts: Options) -> list[Result]:
    out_dir = folder / opts.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = None
    if opts.reference.strip():
        reference = select_files(list_audio(folder), opts.reference)[0]
        print(f"Mastering reference: {reference.name}")

    results, pending = [], []
    for src in files:
        res = Result(source=str(src))
        results.append(res)
        name = safe_stem(src)
        final = out_dir / f"{name}_upscaled.wav"
        if final.exists() and not opts.overwrite:
            res.output, res.skipped = str(final), "already exists (tick OVERWRITE to redo)"
            print(f"\n⏭  {src.name}: {res.skipped}")
            continue
        print(f"\n🎧 {src.name}")
        work = WORK / name
        shutil.rmtree(work, ignore_errors=True)
        mix_wav = decode(src, work / "00_mix.wav")
        mix, sr = read(mix_wav)
        cutoff = detect_cutoff(mix, sr)
        res.notes.append(f"input bandwidth ≈ {cutoff / 1000:.1f} kHz"
                         + (" (full band)" if cutoff >= sr / 2 else " (lossy shelf detected)"))
        print("   " + res.notes[-1])

        stems = {"mix": mix}
        if opts.separate:
            print("   ① separating vocals (BS-RoFormer)…")
            vocals, inst, _ = separate_vocals(mix_wav, work / "sep")
            stems = {"vocals": vocals, "instrumental": inst}

        if opts.apollo:
            for stem, audio in list(stems.items()):
                print(f"   ② Apollo restoring {stem}…")
                src_wav = write(work / f"10_{stem}.wav", audio, sr)
                stems[stem] = fit_length(read(apollo(src_wav, work / f"20_{stem}_apollo.wav"))[0], len(mix))

        gain = 10 ** (opts.vocal_gain_db / 20)
        combined = sum(a * (gain if s == "vocals" else 1.0) for s, a in stems.items())
        stage_wav = write(work / "30_restored.wav", combined, sr)
        if opts.keep_parts:
            for stem, audio in stems.items():
                write(out_dir / f"{name}_parts" / f"{stem}.wav", audio, sr, "PCM_24")
        pending.append((res, name, stage_wav, final, work))

    if opts.audiosr and pending:
        print("\n③ AudioSR super-resolution (all files in one pass)…")
        python = setup_audiosr()
        jobs = [{"in": str(w), "out": str(wk / "40_audiosr.wav"), "cutoff_hz": opts.cutoff_hz}
                for _, _, w, _, wk in pending]
        audiosr(python, jobs, opts.audiosr_steps, opts.audiosr_guidance, opts.seed)
        pending = [(r, n, wk / "40_audiosr.wav", f, wk) for r, n, _, f, wk in pending]
        for (r, *_), job in zip(pending, jobs):
            meta = Path(job["out"]).with_suffix(".json")
            cut = json.loads(meta.read_text()).get("cutoff_hz") if meta.exists() else None
            r.notes.append(f"AudioSR: new highs above {cut / 1000:.1f} kHz" if cut
                           else "AudioSR skipped: already full band")

    for res, name, stage_wav, final, work in pending:
        if reference is not None:
            print(f"\n④ mastering {name} against {reference.name} (Matchering)…")
            master(stage_wav, reference, work / "50_master.wav")
            audio, sr = read(work / "50_master.wav")
            res.notes.append(f"mastered to match {reference.name}")
        else:
            audio, sr = read(stage_wav)
            peak = float(np.abs(audio).max())
            if peak > 0.999:
                audio = audio * (0.999 / peak)
                res.notes.append(f"lowered {db(peak / 0.999):.1f} dB to avoid clipping")
        write(final, audio, sr, "PCM_24")
        res.output = str(final)
        (final.with_suffix(".json")).write_text(json.dumps(asdict(res) | {"options": asdict(opts)},
                                                           ensure_ascii=False, indent=2))
        print(f"✅ {final}")
    return results
