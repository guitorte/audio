"""FlashSR trial: could the one-step FlashSR stand in for AudioSR?

Experimental and separate from ``upscaler.process()``. It restores one short
excerpt of one song, so a few minutes in Colab answer three questions: does
FlashSR load here, how fast is it, and how do its highs sound.

FlashSR runs in the Colab kernel. Unlike AudioSR it has no old pins: it
imports next to current packages (checked on Python 3.13 / torch 2.15), so it
needs no venv of its own.

The splice is the same as in ``audiosr_worker.py``, so the comparison is fair:
Mid/Side, 5.12 s chunks with 0.5 s crossfades, each chunk level-matched to the
band just below the cutoff, and only the band ABOVE the cutoff taken from the
model. Everything below it stays the original audio.
"""
from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import upscaler as U  # noqa: E402
from upscaler_dsp import band_rms, chunk_starts, crossfade, crossover, detect_cutoff, rms  # noqa: E402

SR = 48_000
CHUNK = 245_760            # 5.12 s: the only length FlashSR accepts
OVERLAP = int(0.5 * SR)
TRAINED_UP_TO_HZ = 16_000  # the paper trains on inputs sampled at 4-32 kHz, i.e. cut at 16 kHz or lower
FLASHSR_REPO = "https://github.com/jakeoneijk/FlashSR_Inference.git"
FLASHSR_COMMIT = "2292814a7ef74f61a5479c8d96e653d2f90f369d"
WEIGHTS_REPO = "jakeoneijk/FlashSR_weights"   # a Hugging Face *dataset* repo, about 3.3 GB in all
WEIGHT_FILES = ("student_ldm.pth", "sr_vocoder.pth", "vae.pth")


# --------------------------------------------------------------------------- setup

def setup() -> dict:
    """Clone FlashSR at a pinned commit and download its weights. Returns {file name: path}."""
    repo = U.WORK / "FlashSR"
    if not (repo / "FlashSR" / "FlashSR.py").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        U.run(["git", "clone", "-q", FLASHSR_REPO, repo])
        U.run(["git", "-C", repo, "checkout", "-q", FLASHSR_COMMIT])
    try:
        import einops  # noqa: F401
    except ImportError:
        U.run([sys.executable, "-m", "pip", "install", "-q", "einops"])
    from huggingface_hub import hf_hub_download
    print("   downloading FlashSR weights (about 3.3 GB the first time)…", flush=True)
    target = U.WORK / "models" / "flashsr"
    return {name: hf_hub_download(WEIGHTS_REPO, name, repo_type="dataset", local_dir=str(target))
            for name in WEIGHT_FILES}


@contextlib.contextmanager
def _load_on_cpu():
    """FlashSR calls ``torch.load`` with no ``map_location``, which fails without
    a GPU if the weights were saved from one. Default it to CPU while loading."""
    import torch
    original = torch.load
    torch.load = lambda *a, **kw: original(*a, **{"map_location": "cpu", **kw})
    try:
        yield
    finally:
        torch.load = original


def load(weights: dict, device: str | None = None):
    import torch
    repo = str(U.WORK / "FlashSR")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from FlashSR.FlashSR import FlashSR
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with _load_on_cpu():
        model = FlashSR(weights["student_ldm.pth"], weights["sr_vocoder.pth"], weights["vae.pth"])
    return model.to(device).eval(), device


def make_infer(model, device: str, cutoff: float, seed: int = 42):
    """Wrap the model as ``[rows, CHUNK] float32 -> [rows, CHUNK] float32`` (at most 2 rows)."""
    import torch

    def run(rows: np.ndarray) -> np.ndarray:
        torch.manual_seed(seed)   # FlashSR starts from random noise; fix it so a rerun matches
        with torch.no_grad():
            # FlashSR's own low-pass at our cutoff clears codec residue above the
            # shelf, so the input looks like the band-limited audio it was trained on.
            out = model(torch.from_numpy(rows).to(device), lowpass_input=True, lowpass_cutoff_freq=int(cutoff))
        return out.float().cpu().numpy()

    def infer(batch: np.ndarray) -> np.ndarray:
        try:
            return run(batch)
        except torch.OutOfMemoryError:
            # Mid and Side together did not fit: do them one after the other.
            torch.cuda.empty_cache()
            print("      GPU memory is tight: processing one channel at a time", flush=True)
            return np.concatenate([run(batch[i:i + 1]) for i in range(len(batch))])
    return infer


# --------------------------------------------------------------------------- splice

def enhance(infer, audio: np.ndarray, cutoff: float) -> np.ndarray:
    """``audio`` [samples, 2] at 48 kHz -> same shape, with new content only above ``cutoff``."""
    mid, side = (audio[:, 0] + audio[:, 1]) / 2, (audio[:, 0] - audio[:, 1]) / 2
    # A near-mono side channel only holds noise; the model would invent highs there.
    channels = [mid, side] if rms(side) > 0.03 * rms(mid) else [mid]
    total = len(mid)
    out = np.zeros((len(channels), total), dtype=np.float64)
    weight = np.zeros(total, dtype=np.float64)
    starts = chunk_starts(total, CHUNK, OVERLAP)
    for n, start in enumerate(starts):
        pieces = [c[start:start + CHUNK] for c in channels]
        length = len(pieces[0])
        w = crossfade(length, OVERLAP, fade_in=n > 0, fade_out=n < len(starts) - 1)
        weight[start:start + length] += w
        if length < 4096:
            continue   # a sliver at the very end: too short to filter, nothing worth extending
        lo = [band_rms(p, SR, cutoff / 2, cutoff) for p in pieces]
        live = [i for i, p in enumerate(pieces) if np.abs(p).max() >= 1e-4 and lo[i] >= 1e-7]
        if not live:
            continue   # silence: nothing to extend
        batch = np.stack([np.pad(pieces[i], (0, CHUNK - length)) for i in live]).astype(np.float32)
        generated = infer(batch)[:, :length]
        for row, i in zip(generated, live):
            row = row * (lo[i] / max(band_rms(row, SR, cutoff / 2, cutoff), 1e-9))
            out[i, start:start + length] += row * w
    out /= np.maximum(weight, 1e-9)
    hi_mid, hi_side = out[0], (out[1] if len(channels) == 2 else np.zeros(total))
    hi = np.stack([hi_mid + hi_side, hi_mid - hi_side], axis=1)
    return crossover(audio, hi, cutoff, SR).astype(np.float32)


# --------------------------------------------------------------------------- trial

def clips(res: dict) -> dict:
    """{"Before" | "FlashSR" | "AudioSR": audio}, each level-matched to "Before" for a fair listen."""
    out, ref = {}, None
    for label, key in (("Before", "before"), ("FlashSR", "flashsr"), ("AudioSR", "audiosr")):
        if key in res:
            audio = U.read(Path(res[key]))[0]
            ref = U.rms(audio) if ref is None else ref
            out[label] = audio * (ref / max(U.rms(audio), 1e-9))
    return out


def _free_ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def _release_memory() -> None:
    """Give freed model memory back to the system, not just to Python."""
    import ctypes
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def _save(path: Path, audio: np.ndarray) -> str:
    peak = float(np.abs(audio).max())
    U.write(path, audio * (0.999 / peak) if peak > 0.999 else audio, SR, "PCM_24")
    return str(path)


def trial(src: Path, folder: Path, start: float = 45.0, seconds: float = 20.0, cutoff_hz: float = 0.0,
          compare_audiosr: bool = False, audiosr_steps: int = 50, seed: int = 42) -> dict:
    """Restore one excerpt with FlashSR (and optionally AudioSR) and time it.

    Writes ``<folder>/upscaled/flashsr_trial/<song>_{before,flashsr,audiosr}.wav``
    plus a ``.json`` of the numbers. The source file is never touched.
    """
    import torch
    name = U.safe_stem(src)
    out_dir = folder / "upscaled" / "flashsr_trial"
    before = U.excerpt(src, start, seconds, SR).copy()
    clip_s = len(before) / SR
    cutoff = float(cutoff_hz) or detect_cutoff(before, SR)
    if cutoff >= 0.95 * SR / 2:
        raise ValueError(f"'{src.name}' has no lossy shelf in this excerpt, so there is nothing to rebuild. "
                         "Pick an MP3/AAC song, or set CUTOFF_HZ (e.g. 16000) to force it.")
    print(f"🎧 {src.name}: {clip_s:.1f} s excerpt, new highs above {cutoff / 1000:.1f} kHz")
    if cutoff > TRAINED_UP_TO_HZ:
        print(f"   ⚠️ this song already reaches {cutoff / 1000:.1f} kHz. FlashSR was trained on audio cut at "
              f"{TRAINED_UP_TO_HZ / 1000:.0f} kHz or lower, and what is missing here is at the edge of hearing, "
              "so this run says little about its sound. A duller MP3 is a better test.")
    res = {"source": str(src), "start": start, "seconds": round(clip_s, 2), "cutoff_hz": cutoff,
           "before": _save(out_dir / f"{name}_before.wav", before)}
    summary = out_dir / f"{name}_trial.json"

    if compare_audiosr:
        # First, while this notebook's own process is still small: AudioSR needs
        # most of Colab's system RAM just to load, and is killed if FlashSR's
        # leftovers are in the way. A failure here must not cost the FlashSR run.
        print(f"\n   AudioSR on the same excerpt, for comparison… (free system RAM: {_free_ram_gb():.1f} GB)")
        work = U.WORK / "flashsr_trial" / name
        job = {"in": str(U.write(work / "excerpt.wav", before, SR)), "out": str(work / "audiosr.wav"),
               "cutoff_hz": cutoff}
        try:
            python = U.setup_audiosr()
            t = time.time()
            U.audiosr(python, [job], audiosr_steps, 3.5, seed)
            res["audiosr_run_s"] = round(time.time() - t, 1)   # includes loading AudioSR's model
            res["audiosr"] = _save(out_dir / f"{name}_audiosr.wav", U.read(Path(job["out"]))[0])
        except RuntimeError as err:
            res["audiosr_error"] = str(err).splitlines()[0]
            print(f"   ⚠️ AudioSR did not finish: {res['audiosr_error']}"
                  + ("\n      exit -9 means Colab stopped it for using too much system RAM."
                     if "exit -9" in res["audiosr_error"] else "")
                  + "\n      Carrying on with FlashSR alone.\n")

    t = time.time()
    model, device = load(setup())
    res["device"], res["flashsr_load_s"] = device, round(time.time() - t, 1)
    if device == "cpu":
        print("   ⚠️ no GPU: this will be slow. Use Runtime ▸ Change runtime type ▸ T4 GPU")
    else:
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    after = enhance(make_infer(model, device, cutoff, seed), before, cutoff)
    if device != "cpu":
        torch.cuda.synchronize()
        res["flashsr_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    res["flashsr_run_s"] = round(time.time() - t, 1)
    res["flashsr"] = _save(out_dir / f"{name}_flashsr.wav", after)
    del model
    _release_memory()   # so a second run in the same session starts as light as the first

    summary.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    per_min = res["flashsr_run_s"] / clip_s * 60
    print("\n" + "═" * 60)
    print(f"FlashSR: loaded in {res['flashsr_load_s']} s, restored {clip_s:.1f} s of audio in {res['flashsr_run_s']} s"
          f" (≈ {per_min:.0f} s per minute of music)"
          + (f", peak GPU memory {res['flashsr_vram_gb']} GB" if "flashsr_vram_gb" in res else ""))
    if "audiosr_run_s" in res:
        print(f"AudioSR: {res['audiosr_run_s']} s for the same excerpt, model loading included")
    elif "audiosr_error" in res:
        print("AudioSR: did not finish (see the warning above)")
    print(f"Saved in {out_dir}")
    return res
