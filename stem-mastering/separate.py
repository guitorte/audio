"""Separate a file with audio-separator, keeping the stems at the input's true level.

audio-separator normalises loud input. Feeding it the file at 0.5 peak with
normalization_threshold=1.0 and dividing the stems back keeps them summing to the input
(same trick as neural-upscaler/upscaler.py::separate_vocals). Stems are float WAVs named
by stem (kick.wav, snare.wav, drums.wav, ...).

usage: python separate.py <input> <model file> <out dir> [--models <cache dir>]
  drum kit:  MDX23C-DrumSep-aufr33-jarredou.ckpt   -> kick, snare, toms, hh, ride, crash
  drums out of a mix: htdemucs_ft.yaml            -> drums, bass, other, vocals
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def separate(src, model, outdir, model_dir=None):
    from audio_separator.separator import Separator
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    g = 0.5 / max(float(np.abs(x).max()), 1e-9)
    tmp = outdir / "_in.wav"
    sf.write(tmp, x * g, sr, subtype="FLOAT")
    kw = {"model_file_dir": str(model_dir)} if model_dir else {}
    s = Separator(output_dir=str(outdir), output_format="WAV", normalization_threshold=1.0, **kw)
    s.load_model(model_filename=model)
    written = {}
    for f in s.separate(str(tmp)):
        p = outdir / Path(f).name
        y, r = sf.read(p, dtype="float32", always_2d=True)
        stem = p.stem.split("(")[-1].split(")")[0].lower()     # "_in_(Kick)_model" -> "kick"
        dst = outdir / f"{stem}.wav"
        sf.write(dst, y / g, r, subtype="FLOAT")
        p.unlink()
        written[stem] = dst
    tmp.unlink()
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("model")
    ap.add_argument("outdir")
    ap.add_argument("--models", help="model cache folder")
    a = ap.parse_args()
    for k, v in separate(a.input, a.model, a.outdir, a.models).items():
        print(k, v)
