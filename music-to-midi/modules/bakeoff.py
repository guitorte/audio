"""MIDI engine bake-off: run several song -> MIDI engines on the same excerpt and compare them.

Used by ``notebooks/MIDI_Engine_Bakeoff.ipynb``.

Engines
-------
Separation (feeds the per-stem transcribers), both through ``audio-separator``:
  * ``htdemucs_6s``     - what ``song_to_midi`` uses today
  * ``bs_roformer_sw``  - BS-RoFormer SW (6 stems, much cleaner piano and bass)

Transcription:
  * ``basic_pitch``  - Spotify Basic Pitch (ONNX), with ``transcribe.STEM_PRESETS``
  * ``adtof``        - ADTOF-pytorch (drums)
  * ``adt_str``      - ADT_STR (drums, 2026). Needs Python < 3.13 and torch 2.8 -> own uv venv
  * ``game``         - OpenVPI GAME (singing voice -> notes). Only ONNX weights are
                       released and GAME's own CLI loads PyTorch checkpoints, so this
                       module runs the ONNX graphs itself, following GAME's ONNX.md
  * ``muscriptor``   - Kyutai x Mirelo MuScriptor: full mix -> multi-track MIDI (gated weights)

There is no ground truth unless the user supplies a reference MIDI, so the
comparison also uses proxies:
  * speed (seconds, x real time) and peak GPU memory for torch engines;
  * chroma agreement: each MIDI is rendered with sine tones and its chroma
    (CENS) is compared frame by frame with the audio. Chroma distance tracked
    human preference well in "A Dual Evaluation for Music Transcription"
    (arXiv 2608.04511). It ignores octaves and rewards the right harmony, not
    exact notes;
  * drum onset agreement with onsets librosa detects in the drum stem;
  * real note F1 (mir_eval, onset + pitch, 50 ms) against a reference MIDI.

State (runs, candidates, stems) is module-level and is reset by :func:`start`.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import traceback
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "neural-upscaler"))
import upscaler as U  # noqa: E402  (shared Drive/file/subprocess helpers: run, venv_env, ...)

from .transcribe import STEM_PRESETS  # noqa: E402

STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
CATEGORIES = STEMS  # every MIDI track is filed under one of these

WORK = Path(os.environ.get("BAKEOFF_WORK") or
            ("/content/bakeoff_work" if Path("/content").exists() else HERE.parent / "_work"))
MODELS = WORK / "models"

SEPARATORS = {
    "htdemucs_6s": "htdemucs_6s.yaml",
    "bs_roformer_sw": "BS-Roformer-SW.ckpt",
}

GAME_COMMIT = "69ba01ded0ddda5495e51ba373685f1be1a67ea2"
GAME_ZIP = "https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-{size}-onnx.zip"
GAME_SLICER = f"https://raw.githubusercontent.com/openvpi/GAME/{GAME_COMMIT}/inference/slicer2.py"

ADT_STR_REPO = "Pierfrancesco/adt-str"
ADT_STR_ENV = WORK / "envs" / "adt_str"
# ADT_STR's pyproject pins Python < 3.13 and torch 2.8; Colab's kernel is newer.
ADT_STR_PINS = [
    "torch==2.8.0", "torchaudio==2.8.0", "torchcodec>=0.7,<0.8",
    "transformers", "safetensors", "omegaconf", "pretty_midi", "soundfile",
    "huggingface_hub", "numpy", "h5py", "pedalboard", "tqdm", "setuptools<82",
]

# Basic Pitch settings per stem: the same presets song_to_midi uses.
BASIC_PITCH_PRESETS = {k: {kk: vv for kk, vv in v.items() if kk != "engine"}
                       for k, v in STEM_PRESETS.items() if v.get("engine") == "basic_pitch"}
# GM program per category when per-stem files are assembled into one song.
CATEGORY_PROGRAM = {"vocals": 53, "bass": 33, "guitar": 27, "piano": 0, "other": 48}


# --------------------------------------------------------------------------- helpers

def download(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"baixando {url}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def _torch():
    try:
        import torch
        return torch
    except Exception:  # noqa: BLE001
        return None


def device() -> str:
    t = _torch()
    return "cuda" if t is not None and t.cuda.is_available() else "cpu"


def free_memory() -> None:
    gc.collect()
    t = _torch()
    if t is not None and t.cuda.is_available():
        t.cuda.empty_cache()


def audio_seconds(path) -> float:
    import soundfile as sf
    return sf.info(str(path)).duration


def rms_db(path) -> float:
    """Loudness of a file in dBFS (RMS). A stem far below the others is mostly bleed."""
    import soundfile as sf
    y, _ = sf.read(str(path), always_2d=True)
    return round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)), 1)


def ensure_hf_token() -> bool:
    """Copy Colab's ``HF_TOKEN`` secret into the environment (MuScriptor's weights are gated)."""
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception as e:  # noqa: BLE001  (secret missing, or notebook access not granted)
        print(f"⚠️ Sem o segredo HF_TOKEN ({type(e).__name__}): o MuScriptor não vai conseguir "
              "baixar os pesos. Veja o topo do notebook.")
        return False
    if token:
        os.environ["HF_TOKEN"] = token
    return bool(token)


# --------------------------------------------------------------------------- bookkeeping

@dataclass
class Run:
    engine: str        # e.g. "basic_pitch"
    source: str        # what it read: "mix", "htdemucs_6s:vocals", ...
    output: str        # category for transcriptions, separator name for separations
    seconds: float     # wall clock, model load included
    load_seconds: float | None
    audio_seconds: float
    peak_vram_gb: float | None
    path: str | None   # MIDI file or stems folder
    ok: bool
    error: str | None = None

    @property
    def x_realtime(self) -> float | None:
        return self.audio_seconds / self.seconds if self.ok and self.seconds > 0 else None


RUNS: list[Run] = []
# category -> {candidate label -> MIDI path}
CANDIDATES: dict[str, dict[str, Path]] = {c: {} for c in CATEGORIES}
# separator name -> {stem -> wav}
STEM_FILES: dict[str, dict[str, Path]] = {}
RUN_DIR: Path = WORK / "runs" / "_unset"


@contextmanager
def measure(engine: str, source: str, output: str, audio_s: float):
    """Time a block, track torch peak VRAM, and record success or the error.

    The block receives a dict and may set ``path`` and ``load_seconds``.
    Errors are recorded and printed, not raised, so one engine can't stop the bake-off.
    """
    t = _torch()
    cuda = t is not None and t.cuda.is_available()
    if cuda:
        t.cuda.reset_peak_memory_stats()
    info: dict = {"path": None, "load_seconds": None}
    begin = time.time()
    try:
        yield info
        ok, err = True, None
    except Exception as e:  # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {e}"
        print(f"✗ {engine} em {source} falhou:\n{traceback.format_exc()[-2500:]}", flush=True)
    seconds = time.time() - begin
    vram = t.cuda.max_memory_allocated() / 2**30 if cuda else None
    r = Run(engine, source, output, seconds, info["load_seconds"], audio_s,
            vram if (vram and vram > 0.01) else None, info["path"], ok, err)
    RUNS.append(r)
    if ok:
        xrt = f" ({r.x_realtime:.1f}x tempo real)" if r.x_realtime else ""
        print(f"✓ {engine} em {source} -> {output}: {seconds:.1f}s{xrt}", flush=True)
    free_memory()


# --------------------------------------------------------------------------- input

def start(song: str | Path, start_s: float = 0.0, length_s: float = 0.0) -> Path:
    """Reset the bake-off and decode the excerpt every engine will read (44.1 kHz stereo WAV).

    Each song/excerpt gets its own folder under ``WORK/runs``, so re-running a
    cell for the same excerpt reuses its stems, and a different song never does.
    """
    global RUN_DIR
    RUNS.clear()
    STEM_FILES.clear()
    for c in CATEGORIES:
        CANDIDATES[c] = {}
    song = Path(song)
    tag = f"{U.safe_stem(song)}__{start_s:g}s+{length_s:g}s" if length_s else \
        f"{U.safe_stem(song)}__{start_s:g}s+all"
    RUN_DIR = WORK / "runs" / tag
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    mix = RUN_DIR / "mix.wav"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s}"]
    if length_s > 0:
        cmd += ["-t", f"{length_s}"]
    U.run(cmd + ["-i", str(song), "-ac", "2", "-ar", "44100", str(mix)])
    return mix


# --------------------------------------------------------------------------- separation

_STEM_RE = re.compile(r"_\(([^)]+)\)_")


def separate(mix: Path, name: str) -> dict[str, Path]:
    """Separate with audio-separator into ``RUN_DIR/stems/<name>/<stem>.wav``."""
    out_dir = RUN_DIR / "stems" / name
    existing = {p.stem: p for p in out_dir.glob("*.wav") if p.stem in STEMS}
    if len(existing) == len(STEMS):
        print(f"[{name}] reaproveitando os stems em {out_dir}")
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    stems: dict[str, Path] = {}
    with measure(name, "mix", name, audio_seconds(mix)) as info:
        from audio_separator.separator import Separator
        sep = Separator(output_dir=str(out_dir), model_file_dir=str(MODELS), output_format="WAV",
                        log_level=30, use_autocast=device() == "cuda")
        t0 = time.time()
        sep.load_model(model_filename=SEPARATORS[name])
        info["load_seconds"] = time.time() - t0
        for f in sep.separate(str(mix)):
            f = out_dir / Path(f).name
            m = _STEM_RE.search(f.name)
            stem = m.group(1).lower() if m else None
            if stem in STEMS:
                f.rename(out_dir / f"{stem}.wav")
                stems[stem] = out_dir / f"{stem}.wav"
        info["path"] = str(out_dir)
        del sep
    return stems


# --------------------------------------------------------------------------- transcription engines

_BP_MODEL = None


def basic_pitch(stem_wav: Path, category: str, out_mid: Path) -> Path:
    global _BP_MODEL
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    if _BP_MODEL is None:
        _BP_MODEL = Model(ICASSP_2022_MODEL_PATH)
    _, midi, _ = predict(str(stem_wav), _BP_MODEL,
                         **BASIC_PITCH_PRESETS.get(category, BASIC_PITCH_PRESETS["other"]))
    midi.write(str(out_mid))
    return out_mid


def adtof(drum_wav: Path, out_mid: Path) -> Path:
    from adtof_pytorch import transcribe_to_midi
    transcribe_to_midi(str(drum_wav), str(out_mid), device=device())
    return out_mid


# ---- GAME (singing voice) via the official ONNX release ------------------------------------

_GAME: dict = {}


def _game_load(size: str) -> dict:
    if _GAME.get("size") == size:
        return _GAME
    import onnxruntime as ort

    zip_path = download(GAME_ZIP.format(size=size), MODELS / f"GAME-1.0.3-{size}-onnx.zip")
    root = MODELS / f"GAME-1.0.3-{size}-onnx"
    if not (root / "config.json").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(MODELS)
    # GAME's silence slicer is numpy-only; load it from the pinned commit instead of copying it.
    slicer_py = download(GAME_SLICER, MODELS / "game_slicer2.py")
    spec = importlib.util.spec_from_file_location("game_slicer2", slicer_py)
    slicer_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slicer_mod)

    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    sessions = {n: ort.InferenceSession(str(root / f"{n}.onnx"), providers=providers)
                for n in ("encoder", "segmenter", "estimator", "dur2bd", "bd2dur")}
    _GAME.clear()
    _GAME.update(size=size, cfg=json.loads((root / "config.json").read_text()),
                 s=sessions, Slicer=slicer_mod.Slicer)
    return _GAME


def _game_chunk(g: dict, wav: np.ndarray, language: int, nsteps: int,
                seg_threshold: float, seg_radius: int, est_threshold: float):
    """One chunk through encoder -> D3PM segmenter loop -> estimator (GAME's ONNX.md)."""
    s = g["s"]
    sr = g["cfg"]["samplerate"]
    dur = np.array([len(wav) / sr], dtype=np.float32)
    x_seg, x_est, mask_t = s["encoder"].run(None, {"waveform": wav[None].astype(np.float32),
                                                   "duration": dur})
    known = s["dur2bd"].run(None, {"durations": dur[None], "maskT": mask_t})[0]
    feeds = {"x_seg": x_seg, "known_boundaries": known, "maskT": mask_t,
             "threshold": np.array(seg_threshold, dtype=np.float32),
             "radius": np.array(seg_radius, dtype=np.int64)}
    if "language" in {i.name for i in s["segmenter"].get_inputs()}:
        feeds["language"] = np.array([language], dtype=np.int64)
    boundaries = known
    if g["cfg"].get("loop", False):
        for i in range(nsteps):  # t0 = 0, evenly spaced: GAME's d3pm_sample_ts_resolved
            feeds["prev_boundaries"] = boundaries
            feeds["t"] = np.array([i / nsteps], dtype=np.float32)
            boundaries = s["segmenter"].run(None, feeds)[0]
    else:
        boundaries = s["segmenter"].run(None, feeds)[0]
    durations, mask_n = s["bd2dur"].run(None, {"boundaries": boundaries, "maskT": mask_t})
    presence, scores = s["estimator"].run(None, {
        "x_est": x_est, "boundaries": boundaries, "maskT": mask_t, "maskN": mask_n,
        "threshold": np.array(est_threshold, dtype=np.float32)})
    n = int(mask_n[0].sum())
    return durations[0][:n], presence[0][:n], scores[0][:n]


def game(vocal_wav: Path, out_mid: Path, size: str = "medium", language: str = "",
         nsteps: int = 8, seg_threshold: float = 0.2, seg_radius: int = 2,
         est_threshold: float = 0.2, max_chunk_s: float = 20.0) -> Path:
    """Transcribe a (separated) singing voice with GAME. ``language``: '', 'en', 'ja', 'yue', 'zh'.

    Defaults are GAME's CLI defaults (``infer.py extract``): boundary threshold
    0.2, radius 0.02 s (2 frames), 8 D3PM steps, presence threshold 0.2.
    """
    import librosa
    import pretty_midi

    g = _game_load(size)
    sr = g["cfg"]["samplerate"]
    lang_id = (g["cfg"].get("languages") or {}).get(language, 0)
    wav, _ = librosa.load(str(vocal_wav), sr=sr, mono=True)
    # GAME's own CLI slicing settings (infer.py extract).
    slicer = g["Slicer"](sr=sr, threshold=-40.0, min_length=1000, min_interval=200, max_sil_kept=100)
    chunks = []
    for c in slicer.slice(wav):  # long phrases are cut further: segmenter cost is super-linear
        w, off = c["waveform"], c["offset"]
        step = int(max_chunk_s * sr)
        for k in range(0, len(w), step):
            piece = w[k:k + step]
            if len(piece) > sr * 0.2:
                chunks.append((off + k / sr, piece))
    inst = pretty_midi.Instrument(program=CATEGORY_PROGRAM["vocals"], name="vocals")
    for off, piece in chunks:
        durs, pres, scores = _game_chunk(g, piece, lang_id, nsteps, seg_threshold, seg_radius,
                                         est_threshold)
        length = len(piece) / sr
        onsets = np.minimum(np.concatenate([[0.0], np.cumsum(durs)[:-1]]), length) + off
        offsets = np.minimum(np.cumsum(durs), length) + off
        for on, of, p, v in zip(onsets, offsets, scores, pres):
            if v and of > on:
                inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(round(float(p))),
                                                   start=float(on), end=float(of)))
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(inst)
    pm.write(str(out_mid))
    return out_mid


# ---- ADT_STR (drums) in its own Python 3.12 venv ------------------------------------------

_ADT_STR_WORKER = "\n".join([
    "import sys, json",
    "from huggingface_hub import snapshot_download",
    "variants = ('setting-tau-0.4', 'setting-tau-0.6', 'setting-tau-0.8')",
    "# Code + the one checkpoint we need (each variant is ~280 MB).",
    "repo = snapshot_download(sys.argv[1], ignore_patterns=[f'{v}/*' for v in variants if v != sys.argv[2]])",
    "sys.path.insert(0, repo)",
    "from adt_transcriber import ADTTranscriber",
    "t = ADTTranscriber.from_pretrained(repo, variant=sys.argv[2])",
    "out = t.transcribe(sys.argv[3], output_dir=sys.argv[4])",
    "print('ADT_STR_OUT', json.dumps(str(out)))",
    "",
])


def setup_adt_str() -> Path:
    """Isolated Python 3.12 env for ADT_STR. The pinned install re-runs every time (no-op when met)."""
    python = ADT_STR_ENV / "bin" / "python"
    if not python.exists():
        if shutil.which("uv") is None:
            U.run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        U.run(["uv", "venv", "-q", "--python", "3.12", ADT_STR_ENV])
    torch_index = ("https://download.pytorch.org/whl/cu126" if device() == "cuda"
                   else "https://download.pytorch.org/whl/cpu")
    U.run(["uv", "pip", "install", "-q", "--python", python, "--index-strategy", "unsafe-best-match",
           "--extra-index-url", torch_index, *ADT_STR_PINS])
    return python


def adt_str(drum_wav: Path, out_mid: Path, variant: str = "setting-tau-0.8",
            python: Path | None = None) -> Path:
    python = python or setup_adt_str()
    worker = WORK / "adt_str_worker.py"
    worker.write_text(_ADT_STR_WORKER)
    tmp = out_mid.parent / "_adt_str"
    shutil.rmtree(tmp, ignore_errors=True)
    U.run([python, worker, ADT_STR_REPO, variant, Path(drum_wav).resolve(), tmp.resolve()],
          env=U.venv_env(), cwd=str(WORK))
    produced = next(tmp.glob("*.mid"))
    shutil.move(str(produced), out_mid)
    shutil.rmtree(tmp, ignore_errors=True)
    return out_mid


# ---- MuScriptor (full mix -> multi-track) ---------------------------------------------------

def muscriptor(mix_wav: Path, out_mid: Path, size: str = "medium", info: dict | None = None,
               detect_tempo: str | bool = "best-effort") -> Path:
    from muscriptor import TranscriptionModel

    t0 = time.time()
    model = TranscriptionModel.load_model(size, device=device())
    if info is not None:
        info["load_seconds"] = time.time() - t0
    midi_bytes, _grid = model.transcribe_and_postprocess(str(mix_wav), detect_tempo=detect_tempo)
    out_mid.write_bytes(midi_bytes)
    del model
    return out_mid


# --------------------------------------------------------------------------- MIDI analysis

def category_of(inst) -> str:
    """File a MIDI track under a stem category (by its name, else its GM program)."""
    if inst.is_drum:
        return "drums"
    name = (inst.name or "").lower()
    for c in STEMS:
        if c in name:
            return c
    p = inst.program
    if 52 <= p <= 54 or p == 85:  # choir / voice oohs / synth voice, "voice" lead
        return "vocals"
    if p <= 7:
        return "piano"
    if 24 <= p <= 31:
        return "guitar"
    if 32 <= p <= 39:
        return "bass"
    return "other"


def load_midi(path):
    import pretty_midi
    return pretty_midi.PrettyMIDI(str(path))


def split_by_category(pm) -> dict:
    """{category: PrettyMIDI with only that category's tracks}."""
    import pretty_midi
    out: dict = {}
    for inst in pm.instruments:
        out.setdefault(category_of(inst), pretty_midi.PrettyMIDI()).instruments.append(inst)
    return out


def midi_stats(pm) -> dict:
    notes = [n for i in pm.instruments for n in i.notes]
    if not notes:
        return {"notes": 0, "tracks": len(pm.instruments)}
    pitches = [n.pitch for n in notes]
    end = max(n.end for n in notes)
    fs = 50
    roll = np.zeros(int(end * fs) + 1, dtype=np.int32)
    for n in notes:
        roll[int(n.start * fs):max(int(n.end * fs), int(n.start * fs) + 1)] += 1
    active = roll[roll > 0]
    return {"notes": len(notes), "tracks": len(pm.instruments),
            "pitch_lo": int(min(pitches)), "pitch_hi": int(max(pitches)),
            "mean_polyphony": round(float(active.mean()), 2) if active.size else 0.0,
            "notes_per_s": round(float(len(notes) / max(end, 1e-6)), 2)}


def merge(parts: dict):
    """Assemble {category: MIDI path or PrettyMIDI} into one multi-track PrettyMIDI."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    for cat, src in parts.items():
        if src is None:
            continue
        src_pm = src if hasattr(src, "instruments") else load_midi(src)
        drum = cat == "drums"
        inst = pretty_midi.Instrument(program=0 if drum else CATEGORY_PROGRAM.get(cat, 0),
                                      is_drum=drum, name=cat)
        for i in src_pm.instruments:
            inst.notes.extend(i.notes)
        if inst.notes:
            pm.instruments.append(inst)
    return pm


# --------------------------------------------------------------------------- proxies & scores

SR_METRIC = 22050


def render_sine(pm, length_s: float) -> np.ndarray:
    """Plain sine render of the pitched tracks (drums dropped): enough for chroma."""
    import pretty_midi
    pitched = pretty_midi.PrettyMIDI()
    pitched.instruments = [i for i in pm.instruments if not i.is_drum and i.notes]
    n = int(length_s * SR_METRIC)
    if not pitched.instruments:
        return np.zeros(n, dtype=np.float32)
    y = pitched.synthesize(fs=SR_METRIC).astype(np.float32)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    peak = float(np.abs(y).max())
    return y / peak if peak > 0 else y


def chroma_agreement(ref_audio, pm) -> float | None:
    """Mean frame-wise cosine between chroma-CENS of the reference audio and the sine render.

    Frames where either side sounds count; a frame where only one side sounds
    scores 0 (a missed or invented passage). Range 0..1, higher is better.
    """
    import librosa
    ref, _ = librosa.load(str(ref_audio), sr=SR_METRIC, mono=True)
    est = render_sine(pm, len(ref) / SR_METRIC)
    hop = 2048
    c_ref = librosa.feature.chroma_cens(y=ref, sr=SR_METRIC, hop_length=hop)
    c_est = librosa.feature.chroma_cens(y=est, sr=SR_METRIC, hop_length=hop)
    e_ref = librosa.feature.rms(y=ref, hop_length=hop)[0]
    e_est = librosa.feature.rms(y=est, hop_length=hop)[0]
    t = min(c_ref.shape[1], c_est.shape[1], len(e_ref), len(e_est))
    c_ref, c_est, e_ref, e_est = c_ref[:, :t], c_est[:, :t], e_ref[:t], e_est[:t]
    on_ref = e_ref > max(1e-4, 0.05 * float(np.percentile(e_ref, 95)))
    on_est = e_est > 1e-4
    active = on_ref | on_est
    if not active.any():
        return None
    nr = np.linalg.norm(c_ref, axis=0)
    ne = np.linalg.norm(c_est, axis=0)
    cos = np.where((nr > 0) & (ne > 0), (c_ref * c_est).sum(0) / np.maximum(nr * ne, 1e-9), 0.0)
    cos = np.where(on_ref & on_est, cos, 0.0)
    return round(float(cos[active].mean()), 3)


def drum_onset_agreement(drum_audio, pm) -> dict | None:
    """Onset F1 (50 ms) between the MIDI's drum hits and onsets detected in the drum stem."""
    import librosa
    import mir_eval
    y, sr = librosa.load(str(drum_audio), sr=SR_METRIC, mono=True)
    ref = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    hits = np.unique(np.round(sorted(n.start for i in pm.instruments for n in i.notes), 3))
    if len(ref) == 0 or len(hits) == 0:
        return None
    f, p, r = mir_eval.onset.f_measure(np.asarray(ref), hits, window=0.05)
    return {"onset_f1": round(f, 3), "precision": round(p, 3), "recall": round(r, 3)}


def note_f1(ref_pm, est_pm, offset_s: float = 0.0, length_s: float = 0.0) -> dict | None:
    """Real note-level F1 (onset + pitch, 50 ms) against a reference MIDI, mir_eval style.

    ``ref_pm`` is on the song's timeline; ``offset_s``/``length_s`` cut it to the excerpt.
    """
    import mir_eval

    def arrays(pm, shift):
        iv, hz = [], []
        for i in pm.instruments:
            if i.is_drum:
                continue
            for n in i.notes:
                s, e = n.start - shift, n.end - shift
                if e <= 0 or (length_s and s >= length_s):
                    continue
                iv.append([max(s, 0.0), max(e, s + 1e-3)])
                hz.append(440.0 * 2 ** ((n.pitch - 69) / 12))
        return np.array(iv).reshape(-1, 2), np.array(hz)

    ri, rp = arrays(ref_pm, offset_s)
    ei, ep = arrays(est_pm, 0.0)
    if len(ri) == 0 or len(ei) == 0:
        return None
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
    return {"note_f1": round(f, 3), "precision": round(p, 3), "recall": round(r, 3)}


# --------------------------------------------------------------------------- orchestration

SEP_LABEL = {"htdemucs_6s": "htdemucs_6s", "bs_roformer_sw": "RoFormer SW"}
CURRENT = "Basic Pitch ← htdemucs_6s (atual)"


def _label(engine: str, sep: str) -> str:
    label = f"{engine} ← {SEP_LABEL.get(sep, sep)}"
    return CURRENT if label == "Basic Pitch ← htdemucs_6s" else label


def run_separations(mix: Path, names: list[str]) -> None:
    for name in names:
        stems = separate(mix, name)
        if stems:
            STEM_FILES[name] = stems


def run_stem_engines(mix: Path, *, basic_pitch_on: bool = True, adtof_on: bool = True,
                     game_size: str | None = "medium", game_language: str = "",
                     adt_str_on: bool = True) -> None:
    """Every per-stem engine on every separator's stems; GAME and ADT_STR on the best separator only."""
    out = RUN_DIR / "midi"
    out.mkdir(parents=True, exist_ok=True)
    dur = audio_seconds(mix)
    best = "bs_roformer_sw" if "bs_roformer_sw" in STEM_FILES else next(iter(STEM_FILES), None)
    for sep, stems in STEM_FILES.items():
        if basic_pitch_on:
            for cat in ("vocals", "bass", "guitar", "piano", "other"):
                if cat not in stems:
                    continue
                with measure("basic_pitch", f"{sep}:{cat}", cat, dur) as info:
                    p = basic_pitch(stems[cat], cat, out / f"{sep}__basic_pitch__{cat}.mid")
                    info["path"] = str(p)
                    CANDIDATES[cat][_label("Basic Pitch", sep)] = p
        if adtof_on and "drums" in stems:
            with measure("adtof", f"{sep}:drums", "drums", dur) as info:
                p = adtof(stems["drums"], out / f"{sep}__adtof__drums.mid")
                info["path"] = str(p)
                CANDIDATES["drums"][_label("ADTOF", sep)] = p
    if best is None:
        return
    stems = STEM_FILES[best]
    if game_size and "vocals" in stems:
        with measure(f"game-{game_size}", f"{best}:vocals", "vocals", dur) as info:
            p = game(stems["vocals"], out / f"{best}__game-{game_size}__vocals.mid",
                     size=game_size, language=game_language)
            info["path"] = str(p)
            CANDIDATES["vocals"][_label(f"GAME-{game_size}", best)] = p
    if adt_str_on and "drums" in stems:
        try:  # building the venv is setup, not inference: keep it out of the timing
            python = setup_adt_str()
        except Exception as e:  # noqa: BLE001
            print(f"✗ não deu para preparar o ambiente do ADT_STR, pulando: {e}")
            return
        with measure("adt_str", f"{best}:drums", "drums", dur) as info:
            p = adt_str(stems["drums"], out / f"{best}__adt_str__drums.mid", python=python)
            info["path"] = str(p)
            CANDIDATES["drums"][_label("ADT_STR", best)] = p


def run_muscriptor(mix: Path, size: str = "medium") -> Path | None:
    out = RUN_DIR / "midi"
    out.mkdir(parents=True, exist_ok=True)
    path = None
    with measure(f"muscriptor-{size}", "mix", "all", audio_seconds(mix)) as info:
        path = muscriptor(mix, out / f"mix__muscriptor-{size}.mid", size=size, info=info)
        info["path"] = str(path)
    if path is None or not Path(path).exists():
        return None
    for cat, sub in split_by_category(load_midi(path)).items():
        p = out / f"mix__muscriptor-{size}__{cat}.mid"
        sub.write(str(p))
        CANDIDATES[cat][f"MuScriptor-{size} (mix inteiro)"] = p
    return path


def reference_stem(cat: str) -> tuple[Path | None, str]:
    """The stem audio candidates are scored against: RoFormer SW's when available."""
    for sep in ("bs_roformer_sw", "htdemucs_6s"):
        if cat in STEM_FILES.get(sep, {}):
            return STEM_FILES[sep][cat], SEP_LABEL[sep]
    return None, ""


def quality_table(reference_midi: str | None = None, offset_s: float = 0.0, length_s: float = 0.0):
    """One row per (category, candidate) with the proxies, plus real F1 if a reference is given."""
    import pandas as pd
    ref_cat = split_by_category(load_midi(reference_midi)) if reference_midi else None
    rows = []
    for cat in CATEGORIES:
        stem, stem_label = reference_stem(cat)
        stem_db = rms_db(stem) if stem is not None else None
        for label, path in CANDIDATES[cat].items():
            pm = load_midi(path)
            row = {"category": cat, "candidate": label, **midi_stats(pm)}
            if stem is not None and cat != "drums":
                row["chroma_vs_stem"] = chroma_agreement(stem, pm)
            if stem is not None and cat == "drums":
                d = drum_onset_agreement(stem, pm) or {}
                row.update({f"drum_{k}": v for k, v in d.items()})
            row["reference_stem"] = stem_label
            row["stem_rms_db"] = stem_db
            if ref_cat and cat in ref_cat and cat != "drums":
                row.update({f"ref_{k}": v for k, v in
                            (note_f1(ref_cat[cat], pm, offset_s, length_s) or {}).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def systems(muscriptor_size: str | None) -> dict:
    """Whole-song systems assembled from the per-stem candidates."""
    def pick(cat, *labels):
        for lab in labels:
            if lab in CANDIDATES[cat]:
                return CANDIDATES[cat][lab]
        return None

    sw = "RoFormer SW"
    pitched = ("vocals", "bass", "guitar", "piano", "other")
    out = {}
    if "htdemucs_6s" in STEM_FILES:
        out["A · atual: htdemucs_6s + Basic Pitch + ADTOF"] = merge({
            **{c: pick(c, CURRENT) for c in pitched},
            "drums": pick("drums", "ADTOF ← htdemucs_6s")})
    if "bs_roformer_sw" in STEM_FILES:
        out["B · RoFormer SW + Basic Pitch + ADTOF"] = merge({
            **{c: pick(c, f"Basic Pitch ← {sw}") for c in pitched},
            "drums": pick("drums", f"ADTOF ← {sw}")})
        game_label = next((k for k in CANDIDATES["vocals"] if k.startswith("GAME")), None)
        if game_label or f"ADT_STR ← {sw}" in CANDIDATES["drums"]:
            out["C · RoFormer SW + Basic Pitch + GAME (voz) + ADT_STR (bateria)"] = merge({
                "vocals": pick("vocals", *(lab for lab in [game_label, f"Basic Pitch ← {sw}"] if lab)),
                **{c: pick(c, f"Basic Pitch ← {sw}") for c in pitched[1:]},
                "drums": pick("drums", f"ADT_STR ← {sw}", f"ADTOF ← {sw}")})
    ms = f"MuScriptor-{muscriptor_size} (mix inteiro)"
    ms_parts = [CANDIDATES[c][ms] for c in CATEGORIES if ms in CANDIDATES[c]]
    if ms_parts:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI()
        for p in ms_parts:
            pm.instruments.extend(load_midi(p).instruments)
        out[f"D · MuScriptor-{muscriptor_size} (mix inteiro)"] = pm
    return out


def systems_table(systems_pm: dict, mix: Path, reference_midi: str | None = None,
                  offset_s: float = 0.0, length_s: float = 0.0):
    import pandas as pd
    ref = load_midi(reference_midi) if reference_midi else None
    rows = []
    for name, pm in systems_pm.items():
        row = {"system": name, **midi_stats(pm), "chroma_vs_mix": chroma_agreement(mix, pm)}
        if ref is not None:
            row.update({f"ref_{k}": v for k, v in (note_f1(ref, pm, offset_s, length_s) or {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def save_outputs(dst: Path, systems_pm: dict, tables: dict, overwrite: bool = False) -> Path | None:
    """Copy every MIDI and the tables to ``dst``. An existing folder is left alone unless ``overwrite``."""
    if dst.exists() and any(dst.iterdir()) and not overwrite:
        print(f"⏭️ {dst} já existe: nada foi gravado. Marque OVERWRITE para substituir.")
        return None
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "per_stem").mkdir(exist_ok=True)
    for f in (RUN_DIR / "midi").glob("*.mid"):
        shutil.copy2(f, dst / "per_stem" / f.name)
    for name, pm in systems_pm.items():
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
        pm.write(str(dst / f"{safe}.mid"))
    for name, df in tables.items():
        df.to_csv(dst / f"{name}.csv", index=False)
    print(f"salvo em {dst}")
    return dst


# --------------------------------------------------------------------------- reporting

def runs_table():
    import pandas as pd
    rows = []
    for r in RUNS:
        d = asdict(r)
        d["x_realtime"] = round(r.x_realtime, 1) if r.x_realtime else None
        d["seconds"] = round(r.seconds, 1)
        d["load_seconds"] = round(r.load_seconds, 1) if r.load_seconds else None
        d["peak_vram_gb"] = round(r.peak_vram_gb, 2) if r.peak_vram_gb else None
        rows.append(d)
    cols = ["engine", "source", "output", "ok", "seconds", "load_seconds", "x_realtime",
            "peak_vram_gb", "error"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def piano_roll_grid(candidates: dict, title: str, length_s: float):
    """Stack several candidates' notes for one category on a shared time axis."""
    import matplotlib.pyplot as plt
    names = list(candidates)
    _, axes = plt.subplots(len(names), 1, figsize=(14, 1.9 * len(names)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for a, name in zip(axes, names):
        for i in candidates[name].instruments:
            for n in i.notes:
                a.plot([n.start, n.end], [n.pitch, n.pitch], lw=2.2, solid_capstyle="butt",
                       color="#7f7f7f" if i.is_drum else "#1f77b4")
        a.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        a.set_xlim(0, length_s)
        a.grid(alpha=0.25)
    axes[0].set_title(title, fontweight="bold")
    axes[-1].set_xlabel("segundos")
    plt.tight_layout()
    return axes


def render_preview(pm, length_s: float, sf2: str | None = None) -> tuple[np.ndarray, int]:
    """Audio preview: FluidSynth with a General MIDI soundfont when available, else sines."""
    if sf2 and Path(sf2).exists() and shutil.which("fluidsynth"):
        import subprocess

        import soundfile as sf
        tmp_mid = WORK / "_preview.mid"
        tmp_wav = WORK / "_preview.wav"
        pm.write(str(tmp_mid))
        subprocess.run(["fluidsynth", "-ni", "-q", "-F", str(tmp_wav), "-r", "22050", sf2,
                        str(tmp_mid)], check=True, capture_output=True)
        y, sr = sf.read(str(tmp_wav), always_2d=True)
        y = y.mean(axis=1)[: int(length_s * sr)]
        peak = float(np.abs(y).max()) if y.size else 0.0
        return (y / peak * 0.9 if peak > 0 else y).astype(np.float32), sr
    return render_sine(pm, length_s) * 0.9, SR_METRIC
