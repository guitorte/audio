"""AudioSR worker, run inside the isolated Python 3.10 venv.

AudioSR works on mono clips of about 5 s and peak-normalises everything it
returns, so a whole stereo song needs some care:

* stereo is processed as Mid/Side, which keeps generated highs coherent
  between left and right (independent L/R generation smears the image);
* the song is cut into 5.12 s chunks with 0.5 s linear crossfades;
* each chunk's generated highs are level-matched to the band just below the
  cutoff, undoing AudioSR's normalisation;
* only the band ABOVE the cutoff is taken from AudioSR. Everything below
  is the original audio (via a power-complementary
  zero-phase Butterworth crossover); files with no lossy shelf are left
  untouched, since AudioSR would only swap real highs for invented ones.
"""
import argparse
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from upscaler_dsp import band_rms, chunk_starts, crossfade, crossover, detect_cutoff  # noqa: E402

SR = 48_000
CHUNK = int(5.12 * SR)
OVERLAP = int(0.5 * SR)


def to_48k(audio, sr):
    if sr == SR:
        return audio
    g = np.gcd(sr, SR)
    return resample_poly(audio, SR // g, sr // g, axis=0).astype(np.float32)


def enhance_channel(model, super_resolution, signal, cutoff, args, tmp):
    """Run AudioSR over one mono channel and return its above-cutoff band."""
    out = np.zeros_like(signal)
    weight = np.zeros_like(signal)
    starts = chunk_starts(len(signal), CHUNK, OVERLAP)
    for n, start in enumerate(starts):
        piece = signal[start:start + CHUNK]
        w = crossfade(len(piece), OVERLAP, fade_in=n > 0, fade_out=n < len(starts) - 1)
        lo_band = band_rms(piece, SR, cutoff / 2, cutoff)
        if np.abs(piece).max() < 1e-4 or lo_band < 1e-7:
            weight[start:start + len(piece)] += w   # silence: nothing to extend
            continue
        sf.write(tmp, piece, SR, subtype="FLOAT")
        gen = np.asarray(super_resolution(model, tmp, seed=args.seed, ddim_steps=args.steps,
                                          guidance_scale=args.guidance)).reshape(-1)[:len(piece)]
        gen = np.pad(gen, (0, len(piece) - len(gen)))
        gen *= lo_band / max(band_rms(gen, SR, cutoff / 2, cutoff), 1e-9)
        out[start:start + len(piece)] += gen * w
        weight[start:start + len(piece)] += w
        print(f"      chunk {n + 1}/{len(starts)}", end="\r", flush=True)
    print()
    return out / np.maximum(weight, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    jobs = json.load(open(args.jobs))

    warnings.filterwarnings("ignore")
    from audiosr import build_model, super_resolution
    model = build_model(model_name="basic", device="auto")

    tmp = os.path.join(tempfile.mkdtemp(), "chunk.wav")
    for job in jobs:
        audio, sr = sf.read(job["in"], dtype="float32", always_2d=True)
        audio = to_48k(audio, sr)
        cutoff = float(job.get("cutoff_hz") or 0) or detect_cutoff(audio, SR)
        name = os.path.basename(os.path.dirname(job["in"]))
        if cutoff >= 0.95 * SR / 2:
            # Already full band: AudioSR would only replace real highs with generated ones.
            print(f"   {name}: already full band, AudioSR skipped (set CUTOFF_HZ to force it)")
            sf.write(job["out"], audio, SR, subtype="FLOAT")
            with open(os.path.splitext(job["out"])[0] + ".json", "w") as f:
                json.dump({"cutoff_hz": None}, f)
            continue
        print(f"   {name}: crossover at {cutoff / 1000:.1f} kHz")
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        mid, side = (audio[:, 0] + audio[:, 1]) / 2, (audio[:, 0] - audio[:, 1]) / 2
        hi_mid = enhance_channel(model, super_resolution, mid, cutoff, args, tmp)
        # A near-mono side channel only holds noise; AudioSR would invent highs there.
        hi_side = (enhance_channel(model, super_resolution, side, cutoff, args, tmp)
                   if np.sqrt(np.mean(side ** 2)) > 0.03 * np.sqrt(np.mean(mid ** 2)) else np.zeros_like(side))
        hi = np.stack([hi_mid + hi_side, hi_mid - hi_side], axis=1)
        result = crossover(audio, hi, cutoff, SR)
        sf.write(job["out"], result.astype(np.float32), SR, subtype="FLOAT")
        with open(os.path.splitext(job["out"])[0] + ".json", "w") as f:
            json.dump({"cutoff_hz": cutoff}, f)


if __name__ == "__main__":
    main()
