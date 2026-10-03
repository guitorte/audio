"""Voice conversion for songs in Google Drive or a Kaggle dataset: Seed-VC (zero-shot) and RVC via Applio.

Both engines convert only the vocal. Songs are split with the neural-upscaler's
BS-RoFormer separator, the vocal is converted, level-matched and laid back over
the untouched instrumental.

Seed-VC and Applio pin torch/numpy versions that clash with Colab's own Python,
so each runs from its own pinned clone in its own ``uv`` venv, launched through
``upscaler.run`` (streams output) with ``upscaler.venv_env`` (no leaked
MPLBACKEND/PYTHONPATH).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "neural-upscaler"))
import upscaler as U  # noqa: E402  (shared Drive/audio/subprocess helpers)

SR = 44_100
WORK = Path("/content/vc_work") if Path("/content").exists() else HERE / "_work"
VOICES_SUBDIR = "voices"
OUT_SUBDIR = "converted"

SEEDVC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
SEEDVC_DIR = WORK / "seed-vc"
SEEDVC_ENV = WORK / "envs" / "seedvc"
# Inference subset of seed-vc's requirements.txt (the rest is GUI, training and eval).
SEEDVC_PINS = ["torch==2.4.0", "torchaudio==2.4.0", "numpy==1.26.4", "scipy==1.13.1", "librosa==0.10.2",
               "huggingface-hub>=0.28.1,<1.0", "munch==4.0.0", "einops==0.8.0", "descript-audio-codec==1.0.0",
               "transformers==4.46.3", "soundfile", "pyyaml", "hydra-core==1.3.2", "accelerate"]

APPLIO_REPO = "https://github.com/IAHispano/Applio.git"
APPLIO_COMMIT = "c7665ac9a305b3683570ed914f1577d53d4b75c4"
APPLIO_DIR = WORK / "Applio"
APPLIO_ENV = WORK / "envs" / "applio"
# PyPI's torch 2.11 is a CUDA 13 build, which needs a newer driver than Colab
# GPUs may have; Applio's own installer uses the cu128 index the same way.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


# --------------------------------------------------------------------------- setup

def _clone(repo: str, commit: str, dest: Path) -> None:
    if (dest / ".git").exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    U.run(["git", "clone", "-q", repo, dest])
    U.run(["git", "-C", dest, "checkout", "-q", commit])


def _venv(env: Path, python_version: str) -> Path:
    python = env / "bin" / "python"
    if not python.exists():
        if shutil.which("uv") is None:
            U.run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        U.run(["uv", "venv", "-q", "--python", python_version, env])
    return python


def setup_seedvc() -> Path:
    _clone(SEEDVC_REPO, SEEDVC_COMMIT, SEEDVC_DIR)
    python = _venv(SEEDVC_ENV, "3.10")
    U.run(["uv", "pip", "install", "-q", "--python", python, *SEEDVC_PINS])
    return python


def setup_applio(for_training: bool = False) -> Path:
    _clone(APPLIO_REPO, APPLIO_COMMIT, APPLIO_DIR)
    python = _venv(APPLIO_ENV, "3.12")
    U.run(["uv", "pip", "install", "-q", "--python", python, "-r", APPLIO_DIR / "requirements.txt",
           "--extra-index-url", TORCH_INDEX, "--index-strategy", "unsafe-best-match"])
    config = APPLIO_DIR / "assets" / "config.json"
    if not config.exists():
        # The CLI reads assets/config.json (created by the UI on first launch);
        # without it training silently skips writing the voice .pth.
        shutil.copy(APPLIO_DIR / "assets" / "config_template.json", config)
    U.run([python, "core.py", "prerequisites", "--models", "--no-exe",
           "--pretraineds-hifigan" if for_training else "--no-pretraineds-hifigan"],
          cwd=APPLIO_DIR, env=U.venv_env())
    return python


# --------------------------------------------------------------------------- voices folder

@dataclass
class Voices:
    folder: Path                                 # <input>/voices (may be read-only, may not exist)
    out: Path                                    # <output>/voices (writable: trained models, mirror)
    references: list[Path]                       # audio clips for Seed-VC
    models: dict[str, tuple[Path, Path | None]]  # RVC name -> (.pth, .index)
    datasets: dict[str, Path]                    # subfolders of recordings for RVC training


def scan_voices(folder: Path, out: Path | None = None) -> Voices:
    """Voices from ``<folder>/voices`` plus models trained into ``<out>/voices``.

    ``out`` defaults to ``folder`` (Colab: one Drive folder). On Kaggle the
    input is a read-only dataset and ``out`` is /kaggle/working.
    """
    vdir, odir = folder / VOICES_SUBDIR, (out or folder) / VOICES_SUBDIR
    odir.mkdir(parents=True, exist_ok=True)
    refs = U.list_audio(vdir) if vdir.is_dir() else []
    models = {}
    for d in dict.fromkeys([vdir, odir]):  # output last, so a freshly trained model wins
        for pth in sorted(d.glob("*.pth")) if d.is_dir() else []:
            idx = pth.with_suffix(".index")
            models[pth.stem] = (pth, idx if idx.exists() else None)
    datasets = {d.name: d for d in sorted(vdir.iterdir())
                if d.is_dir() and not d.name.startswith(".") and U.list_audio(d)} if vdir.is_dir() else {}
    return Voices(vdir, odir, refs, models, datasets)


def describe(v: Voices) -> str:
    lines = [f"🎤 {v.folder}" + (f"  (trained models → {v.out})" if v.out != v.folder else "")]
    lines.append("  Seed-VC reference clips:" if v.references
                 else "  Seed-VC reference clips: none in voices/ (a song number or name from the list above also works)")
    lines += [f"    {i:2d}. {p.name}" for i, p in enumerate(v.references, 1)]
    lines.append("  RVC models:" if v.models else "  RVC models: none yet")
    lines += [f"     • {n}" + ("" if idx else "  (no .index)") for n, (_, idx) in v.models.items()]
    lines.append("  RVC training sets:" if v.datasets else "  RVC training sets: none yet")
    lines += [f"     • {n}/  ({len(U.list_audio(d))} files)" for n, d in v.datasets.items()]
    return "\n".join(lines)


def pick(names: list[str], spec: str, what: str) -> str:
    spec = spec.strip()
    if spec.isdigit() and 1 <= int(spec) <= len(names):
        return names[int(spec) - 1]
    hits = [n for n in names if U._nfc(spec) in U._nfc(n)]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"{what} '{spec}' " + ("is ambiguous: " + ", ".join(hits) if hits
                                             else "not found. Available: " + (", ".join(names) or "none")))


def resolve_reference(folder: Path, voices: Voices, spec: str) -> Path:
    """A Seed-VC reference: a clip in voices/ first, else a file from the songs list.

    Numbers follow the listing they come from: voices/ clips when there are
    any, otherwise the song numbers printed by cell 1.
    """
    if voices.references:
        try:
            name = pick([p.name for p in voices.references], spec, "Reference clip")
            return next(p for p in voices.references if p.name == name)
        except ValueError:
            pass
    songs = U.list_audio(folder)
    try:
        ref = folder / pick([p.name for p in songs], spec, "Reference")
    except ValueError:
        raise ValueError(
            f"Reference '{spec}' matches no clip in voices/ "
            f"({', '.join(p.name for p in voices.references) or 'empty'}) "
            f"and no song ({', '.join(p.name for p in songs) or 'none'}).") from None
    print(f"Using '{ref.name}' from the songs folder as the reference voice.")
    return ref


# --------------------------------------------------------------------------- audio

def mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def to_sr(audio: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target:
        return audio
    g = np.gcd(sr, target)
    return resample_poly(audio, target // g, sr // g, axis=0).astype(np.float32)


def loudest_window(audio: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """The ``seconds``-long stretch with the most energy (Seed-VC reads only ~25 s of reference)."""
    x = mono(audio)
    n = int(seconds * sr)
    if len(x) <= n:
        return audio
    hop = sr // 2
    energy = np.convolve(np.square(x[::hop]), np.ones(max(1, n // hop)), mode="valid")
    start = int(np.argmax(energy)) * hop
    return audio[start:start + n]


def split(src: Path, work: Path, separate: bool) -> tuple[Path, np.ndarray | None, np.ndarray]:
    """Decode a song and (optionally) split it. Cached per song in ``work``.

    Returns (vocals wav, instrumental or None, original vocals array).
    """
    vocals_wav, inst_wav = work / "vocals.wav", work / "instrumental.wav"
    if separate and vocals_wav.exists() and inst_wav.exists():
        return vocals_wav, U.read(inst_wav)[0], U.read(vocals_wav)[0]
    mix_wav = U.decode(src, work / "mix.wav", SR)
    if not separate:
        return mix_wav, None, U.read(mix_wav)[0]
    print("   separating vocals (BS-RoFormer)…", flush=True)
    vocals, inst, _ = U.separate_vocals(mix_wav, work / "sep")
    U.write(inst_wav, inst, SR)
    U.write(vocals_wav, vocals, SR)
    return vocals_wav, inst, vocals


def rebuild(converted: Path, original_vocals: np.ndarray, inst: np.ndarray | None,
            vocal_gain_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Converted vocal at 44.1 kHz stereo, level-matched to the original vocal, plus the new mix."""
    audio, sr = U.read(converted)
    vc = U.fit_length(to_sr(mono(audio)[:, None], sr), len(original_vocals))
    vc = np.repeat(vc, 2, axis=1)
    # Both engines normalise their output; restore the original vocal's level.
    vc *= U.rms(original_vocals) / max(U.rms(vc), 1e-9)
    vc *= 10 ** (vocal_gain_db / 20)
    mix = vc if inst is None else inst + vc
    peak = float(np.abs(mix).max())
    if peak > 0.999:
        vc, mix = vc * (0.999 / peak), mix * (0.999 / peak)
    return vc, mix


# --------------------------------------------------------------------------- engines

def _expect(path: Path, what: str) -> Path:
    # Applio's CLI can print an error and still exit 0, so check the file itself.
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{what} did not produce {path}. See the output above for the reason.")
    return path


def _expect_files(folder: Path, pattern: str, what: str) -> None:
    if not any(folder.glob(pattern)):
        raise RuntimeError(f"{what} produced no {pattern} files in {folder}. See the output above for the reason.")


def seedvc(python: Path, source: Path, reference: Path, out_dir: Path, *, singing: bool,
           semitones: int, steps: int, cfg_rate: float, fp16: bool = True) -> Path:
    # The engines run with cwd= their clone, so every path must be absolute.
    source, reference, out_dir = source.resolve(), reference.resolve(), out_dir.resolve()
    shutil.rmtree(out_dir, ignore_errors=True)
    U.run([python, "inference.py", "--source", source, "--target", reference, "--output", out_dir,
           "--diffusion-steps", steps, "--inference-cfg-rate", cfg_rate,
           "--f0-condition", singing, "--auto-f0-adjust", False, "--semi-tone-shift", semitones,
           "--fp16", fp16], cwd=SEEDVC_DIR, env=U.venv_env())
    outs = sorted(out_dir.glob("*.wav"))
    return _expect(outs[0] if outs else out_dir / "missing.wav", "Seed-VC")


def rvc(python: Path, source: Path, pth: Path, index: Path | None, out: Path, *,
        pitch: int, index_rate: float, protect: float) -> Path:
    source, pth, out = source.resolve(), pth.resolve(), out.resolve()
    index = index.resolve() if index else None
    out.unlink(missing_ok=True)
    U.run([python, "core.py", "infer", "--input-path", source, "--output-path", out,
           "--pth-path", pth, "--index-path", index or "", "--pitch", pitch,
           "--index-rate", index_rate if index else 0, "--protect", protect,
           "--f0-method", "rmvpe", "--export-format", "WAV"], cwd=APPLIO_DIR, env=U.venv_env())
    return _expect(out, "Applio")


# --------------------------------------------------------------------------- song conversion

@dataclass
class Options:
    separate: bool = True
    vocal_gain_db: float = 0.0
    overwrite: bool = False
    # Seed-VC
    singing: bool = True
    semitones: int = 0
    steps: int = 30
    cfg_rate: float = 0.7
    isolate_reference: bool = True
    # RVC
    index_rate: float = 0.5
    protect: float = 0.33


@dataclass
class Result:
    source: str
    output: str = ""
    vocals: str = ""
    skipped: str = ""
    notes: list[str] = field(default_factory=list)


def _reference_clip(ref: Path, isolate: bool) -> Path:
    """Clean 25 s of the target voice: vocals isolated, loudest stretch."""
    work = WORK / "refs" / f"{U.safe_stem(ref)}{'_isolated' if isolate else ''}"
    clip = work / "reference.wav"
    if clip.exists():
        return clip
    vocals_wav, _, vocals = split(ref, work, isolate)
    U.write(clip, loudest_window(vocals, SR, 25.0), SR)
    return clip


def convert(files: list[Path], folder: Path, engine: str, voice: str, opts: Options,
            out: Path | None = None) -> list[Result]:
    """``engine`` is "seedvc" (``voice`` = reference clip name) or "rvc" (``voice`` = model name).

    Reads songs/voices from ``folder``; writes to ``out`` (default: ``folder``).
    """
    voices = scan_voices(folder, out)
    if engine == "seedvc":
        ref = resolve_reference(folder, voices, voice)
        label = U.safe_stem(ref)
        U.setup_main(opts.separate or opts.isolate_reference, False, False)
        python = setup_seedvc()
        print(f"\nReference voice: {ref.name}")
        reference = _reference_clip(ref, opts.isolate_reference)
    elif engine == "rvc":
        label = pick(list(voices.models), voice, "RVC model")
        pth, index = voices.models[label]
        if opts.separate:
            U.setup_main(True, False, False)
        python = setup_applio()
        print(f"\nRVC model: {pth.name}" + (f" + {index.name}" if index else " (no index)"))
    else:
        raise ValueError("engine must be 'seedvc' or 'rvc'")

    out_dir = (out or folder) / OUT_SUBDIR
    results = []
    for src in files:
        res = Result(source=str(src))
        results.append(res)
        name = U.safe_stem(src)
        final = out_dir / f"{name}__{label}__{engine}.wav"
        if final.exists() and not opts.overwrite:
            res.output, res.skipped = str(final), "already exists (tick OVERWRITE to redo)"
            print(f"\n⏭  {src.name}: {res.skipped}")
            continue
        print(f"\n🎧 {src.name}", flush=True)
        work = WORK / "songs" / name
        vocals_wav, inst, original = split(src, work, opts.separate)
        print(f"   converting vocal with {engine}…", flush=True)
        if engine == "seedvc":
            converted = seedvc(python, vocals_wav, reference, work / f"seedvc_{label}", singing=opts.singing,
                               semitones=opts.semitones, steps=opts.steps, cfg_rate=opts.cfg_rate)
            res.notes.append(f"Seed-VC {'singing' if opts.singing else 'speech'} model, {opts.steps} steps, "
                             f"{opts.semitones:+d} semitones")
        else:
            converted = rvc(python, vocals_wav, pth, index, work / f"rvc_{label}.wav", pitch=opts.semitones,
                            index_rate=opts.index_rate, protect=opts.protect)
            res.notes.append(f"RVC pitch {opts.semitones:+d}, index rate {opts.index_rate if index else 0}, "
                             f"protect {opts.protect}")
        vc, mix = rebuild(converted, original, inst, opts.vocal_gain_db)
        U.write(final, mix, SR, "PCM_24")
        vocals_out = final.with_name(final.stem + "_vocals.wav")
        U.write(vocals_out, vc, SR, "PCM_24")
        res.output, res.vocals = str(final), str(vocals_out)
        if inst is None:
            res.notes.append("no vocal split: the whole file was treated as a vocal")
        final.with_suffix(".json").write_text(json.dumps(
            asdict(res) | {"engine": engine, "voice": label, "options": asdict(opts)}, ensure_ascii=False, indent=2))
        print(f"✅ {final}")
    return results


# --------------------------------------------------------------------------- RVC training

@dataclass
class TrainOptions:
    isolate_vocals: bool = True
    epochs: int = 200
    save_every: int = 10
    batch_size: int = 8
    sample_rate: int = 40000


class _Mirror:
    """Copy finished checkpoint files to the output folder while Applio trains.

    Colab can disconnect and Kaggle sessions end; mirrored checkpoints let the
    next run resume. A file is copied once it has been unchanged for 30 s, so a
    half-written checkpoint is never mirrored.
    """
    def __init__(self, src: Path, dst: Path, patterns: tuple[str, ...], newest_only: str = ""):
        self.src, self.dst, self.patterns, self.newest_only = src, dst, patterns, newest_only
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sync(self, settle: float = 30.0) -> None:
        self.dst.mkdir(parents=True, exist_ok=True)
        for pattern in self.patterns + ((self.newest_only,) if self.newest_only else ()):
            found = sorted(self.src.glob(pattern), key=lambda p: p.stat().st_mtime)
            if pattern == self.newest_only:
                found = found[-1:]
            for f in found:
                mtime, dst = f.stat().st_mtime, self.dst / f.name
                # copy2 keeps mtime, so an unchanged (or just-restored) file is skipped.
                if time.time() - mtime > settle and not (dst.exists() and dst.stat().st_mtime == mtime):
                    shutil.copy2(f, dst)
            if pattern == self.newest_only and found:
                for old in self.dst.glob(pattern):
                    if old.name != found[-1].name:
                        old.unlink()

    def _loop(self) -> None:
        while not self.stop.wait(60):
            try:
                self.sync()
            except OSError as e:
                print(f"   (checkpoint mirror skipped: {e})", flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sync(settle=0)


def _resume_sources(voices: Voices, model_name: str) -> list[Path]:
    """Where an earlier session may have left checkpoints, best first.

    On Kaggle, a previous run's output can be attached as an input; it then
    appears read-only under /kaggle/input/<name>/voices/.training/<model>.
    """
    tail = Path(VOICES_SUBDIR) / ".training" / model_name
    found = [voices.out / ".training" / model_name, voices.folder / ".training" / model_name]
    kaggle = KAGGLE_INPUT
    if kaggle.is_dir():
        found += [p.parent.parent / tail for p in kaggle.glob(f"**/{VOICES_SUBDIR}/.training")]
    return [d for d in dict.fromkeys(found) if any(d.glob("G_*.pth"))]


def train(folder: Path, dataset: str, model_name: str, opts: TrainOptions,
          out: Path | None = None) -> tuple[Path, Path | None]:
    """Train from ``<folder>/voices/<dataset>/``; publish to ``<out>/voices/`` (default: ``folder``)."""
    voices = scan_voices(folder, out)
    ds_name = pick(list(voices.datasets), dataset, "Training folder")
    model_name = U.safe_stem(Path(model_name or ds_name)).replace(" ", "_")
    if opts.isolate_vocals:
        U.setup_main(True, False, False)
    python = setup_applio(for_training=True)

    # 1. Clean, decoded copies of the recordings.
    data = WORK / "datasets" / model_name
    files = U.list_audio(voices.datasets[ds_name])
    print(f"\nPreparing {len(files)} recordings from {ds_name}/…", flush=True)
    for src in files:
        dst = data / f"{U.safe_stem(src)}.wav"
        if dst.exists():
            continue
        vocals_wav, _, vocals = split(src, WORK / "songs" / U.safe_stem(src), opts.isolate_vocals)
        U.write(dst, mono(vocals), SR)

    # 2. Resume if a previous session got this far.
    exp = APPLIO_DIR / "logs" / model_name
    mirror = voices.out / ".training" / model_name
    sources = _resume_sources(voices, model_name)
    if not any(exp.glob("G_*.pth")) and sources:
        exp.mkdir(parents=True, exist_ok=True)
        for f in sources[0].iterdir():
            shutil.copy2(f, exp / f.name)
        print(f"   resuming from checkpoints saved in {sources[0]}")

    env, sr = U.venv_env(), str(opts.sample_rate)
    # "-" = CPU. Applio's extract silently writes nothing if told to use a GPU that isn't there.
    gpu = "0" if shutil.which("nvidia-smi") else "-"
    # Applio reports success even when a step produced nothing, so check each step's output.
    if not any((exp / "sliced_audios").glob("*.wav")):
        U.run([python, "core.py", "preprocess", "--model-name", model_name, "--dataset-path", data.resolve(),
               "--sample-rate", sr, "--cpu-cores", 2], cwd=APPLIO_DIR, env=env)
        _expect_files(exp / "sliced_audios", "*.wav", "Applio preprocess")
    if not any((exp / "extracted").glob("*.npy")):
        U.run([python, "core.py", "extract", "--model-name", model_name, "--sample-rate", sr,
               "--f0-method", "rmvpe", "--gpu", gpu, "--cpu-cores", 2], cwd=APPLIO_DIR, env=env)
        _expect_files(exp / "extracted", "*.npy", "Applio feature extraction")

    # 3. Train, mirroring checkpoints to the output folder as they appear.
    with _Mirror(exp, mirror, ("G_*.pth", "D_*.pth", "config.json", "model_info.json", "*.index"),
                    newest_only=f"{model_name}_*e_*s.pth"):
        U.run([python, "core.py", "train", "--model-name", model_name, "--sample-rate", sr,
               "--total-epoch", opts.epochs, "--save-every-epoch", opts.save_every,
               "--batch-size", opts.batch_size, "--save-only-latest", "--gpu", gpu], cwd=APPLIO_DIR, env=env)

    # 4. Publish the newest weights + index where the convert cell looks.
    weights = sorted(exp.glob(f"{model_name}_*e_*s.pth"), key=lambda p: p.stat().st_mtime)
    pth = _expect(weights[-1] if weights else exp / f"{model_name}.pth", "RVC training")
    shutil.copy2(pth, voices.out / f"{model_name}.pth")
    index = exp / f"{model_name}.index"
    if index.exists():
        shutil.copy2(index, voices.out / f"{model_name}.index")
    print(f"\n✅ {voices.out / (model_name + '.pth')}  (from {pth.name})")
    return voices.out / f"{model_name}.pth", (voices.out / f"{model_name}.index") if index.exists() else None


# --------------------------------------------------------------------------- Kaggle

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_OUTPUT = Path("/kaggle/working")


def kaggle_input(name: str = "", root: Path | None = None) -> Path:
    """The attached dataset folder that holds the songs (and/or voices/).

    ``name`` may be empty (auto-detect), a path relative to /kaggle/input, or
    part of a folder name. Attached outputs of earlier runs (they contain
    ``converted/``) are skipped so they aren't mistaken for the songs.
    """
    root = root or KAGGLE_INPUT
    if name.strip():
        direct = Path(name) if Path(name).is_absolute() else root / name
        if direct.is_dir():
            return direct
    candidates = []
    for d in [root, *sorted(p for p in root.glob("**/*") if p.is_dir() and len(p.relative_to(root).parts) <= 4)]:
        rel = d.relative_to(root).parts
        if VOICES_SUBDIR in rel or OUT_SUBDIR in rel or (d / OUT_SUBDIR).is_dir():
            continue  # inside voices/ (training sets, not songs) or an attached earlier output
        songs = U.list_audio(d) if d.is_dir() else []
        if songs or (d / VOICES_SUBDIR).is_dir():
            candidates.append((0 if songs else 1, len(d.parts), d))
    if name.strip():
        candidates = [c for c in candidates if U._nfc(name) in U._nfc(str(c[2].relative_to(root)))]
    if not candidates:
        raise FileNotFoundError(
            f"No folder with songs found under {root}. Upload your songs as a Kaggle Dataset and attach it "
            "with 'Add Input', or set INPUT to its folder.")
    candidates.sort(key=lambda c: (c[0], c[1]))
    best = [c for c in candidates if c[:2] == candidates[0][:2]]
    if len(best) > 1:
        raise ValueError("Several folders could hold your songs; set INPUT to one of: "
                         + ", ".join(str(c[2]) for c in best))
    return best[0][2]


def kaggle_setup() -> str:
    """ffmpeg and a GPU sanity check for a Kaggle session. Returns the GPU name."""
    if shutil.which("ffmpeg") is None:
        U.run(["apt-get", "-qq", "update"])
        U.run(["apt-get", "-qq", "install", "-y", "ffmpeg"])
    gpu = ""
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip().splitlines()[0:1]
        gpu = gpu[0] if gpu else ""
    if not gpu:
        print("⚠️ No GPU. Settings ▸ Accelerator ▸ GPU T4 x2.")
    elif "P100" in gpu:
        print("⚠️ P100: Seed-VC works, but RVC (Applio's PyTorch build) no longer supports this GPU. "
              "Switch to GPU T4 x2 for RVC.")
    return gpu
