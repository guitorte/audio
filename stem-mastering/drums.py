"""Reshape the drums of a song from its stems, then rebuild the mix.

1. Split the drum stem into kick, snare, toms, hh, ride, crash (MDX23C DrumSep, true level),
   unless --kit points at an earlier split.
2. Run each piece through its own chain (dsp.CHAINS: transient shaper, EQ, compression),
   level-matched back to the input piece so the kit balance stays where the song had it.
3. Optionally humanize the hi-hat/shaker layer (--humanize-hh): per-hit levels with an accent on
   the beat, a slow phrase drift and a little randomness, ducking under the vocal, softer top.
4. Rebuild: shaped pieces + the residual (whatever the split missed, untouched). A look-ahead
   limiter keeps the new drum peaks within 1 dB of the original stem's peak.
5. Sum with the other stems: a float mix for master.py, plus 24-bit full / instrumental mixes.

usage: python drums.py <stems dir> <out dir> [--kit <dir>] [--humanize-hh] [--seed N]
The stems dir holds one file per stem; the drum stem's name contains "drum", the vocal's "vocal".
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter1d

import dsp

PIECES = ("kick", "snare", "toms", "hh", "ride", "crash")
AUDIO = (".wav", ".flac", ".aiff", ".aif", ".mp3", ".ogg")


def find_stems(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in AUDIO)
    drums = [p for p in files if "drum" in p.stem.lower()]
    if len(drums) != 1:
        raise SystemExit(f"expected exactly one drum stem in {folder}, found {[p.name for p in drums]}")
    return drums[0], [p for p in files if p != drums[0]]


def read(p, n=None):
    x, sr = sf.read(p, always_2d=True, dtype="float64")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return (x if n is None else fit(x, n)), sr


def fit(x, n):
    return x[:n] if len(x) >= n else np.vstack([x, np.zeros((n - len(x), x.shape[1]))])


def hf_band(x, sr):
    return signal.sosfiltfilt(signal.butter(4, [4000, 16000], "bandpass", fs=sr, output="sos"), x)


def humanize_hh(h, sr, vocals=None, seed=7, cut_db=-3.0, duck_db=-3.0, offbeat_db=-2.5,
                drift_db=1.0, random_db=1.2, top_db=-2.0):
    """Break up a machine-steady hi-hat/shaker. Returns (processed, stats)."""
    import librosa
    hf = hf_band(h.mean(1), sr)
    on = librosa.onset.onset_detect(
        onset_envelope=librosa.onset.onset_strength(y=hf.astype(np.float32), sr=sr, hop_length=256),
        sr=sr, hop_length=256, units="samples", backtrack=False)
    on = on[on > 0]
    if len(on) < 16:
        return h, {"hits": len(on), "note": "too few hits, left unchanged"}
    ioi = np.diff(on) / sr
    t = on / sr
    # the pattern's grid (e.g. eighths): the period the hits line up with best, searched around the
    # median spacing (onset jitter biases the median; counting hit to hit breaks on one missed hit)
    med = float(np.median(ioi[(ioi > 0.08) & (ioi < 0.5)]))
    cands = np.linspace(0.97 * med, 1.03 * med, 601)
    step = float(cands[np.argmax([abs(np.exp(2j * np.pi * t / c).mean()) for c in cands])])
    beat = 2 * step
    ph = (t % beat) / beat
    pk = np.array([np.abs(hf[o:o + int(.03 * sr)]).max() for o in on])
    first = (ph < 0.25) | (ph > 0.75)
    on_beat = first if pk[first].mean() >= pk[~first].mean() else ~first   # louder half = on the beat
    rng = np.random.default_rng(seed)
    phrase = drift_db * np.sin(2 * np.pi * t / (8 * 4 * beat) + 0.7) + 0.5 * drift_db * np.sin(2 * np.pi * t / (3 * 4 * beat))
    hit_db = np.where(on_beat, 0.0, offbeat_db) + phrase + np.clip(rng.normal(0, random_db, len(on)), -2.5, 2.5)
    n = len(h)
    g = np.zeros(n)
    starts = np.maximum(on - int(0.004 * sr), 0)                     # change level just before each hit
    for s, e, v in zip(starts, list(starts[1:]) + [n], hit_db):
        g[s:e] = v
    g = dsp.env(g, sr, 3.0)
    duck = np.zeros(n)
    if vocals is not None:
        v = vocals.mean(1)
        vl = 10 * np.log10(dsp.env(v ** 2, sr, 150) + 1e-12)
        act = np.clip((vl - (np.percentile(vl, 95) - 30)) / 12, 0, 1)   # 0 = silent, 1 = singing
        duck = duck_db * dsp.env(act, sr, 120)
    y = h * (10 ** ((cut_db + g + duck) / 20))[:, None]
    y = dsp.peq(y, sr, 9000, top_db, 0.7, "high")
    stats = {"hits": len(on), "grid_s": round(step, 4),
             "on/off-beat dB": (round(hit_db[on_beat].mean(), 1), round(hit_db[~on_beat].mean(), 1)),
             "per-hit spread dB": round(hit_db.std(), 1),
             "ducking active %": round((duck < -1).mean() * 100),
             "layer change dB": round(10 * np.log10((y ** 2).sum() / (h ** 2).sum()), 1)}
    return y, stats


def peak_cap(bus, ref, sr, over_db=1.0, la_ms=2.0):
    """Look-ahead limiter: the bus may not exceed the original stem's peak + over_db."""
    ceil = np.abs(ref).max() * 10 ** (over_db / 20)
    la = int(la_ms / 1000 * sr)
    pk = maximum_filter1d(np.abs(bus).max(1), size=2 * la + 1)
    g = np.minimum(1, ceil / np.maximum(pk, 1e-12))
    g = np.minimum(dsp.env(g, sr, 1.0), g)
    return bus * g[:, None], -20 * np.log10(g.min())


def main(a):
    out = Path(a.out)
    (out / "kit").mkdir(parents=True, exist_ok=True)
    drum_path, others = find_stems(a.stems)
    d, sr = read(drum_path)
    n = len(d)
    kit = Path(a.kit) if a.kit else out / "kit_split"
    if not a.kit:
        from separate import separate
        separate(str(drum_path), "MDX23C-DrumSep-aufr33-jarredou.ckpt", kit, a.models)
    pieces, shaped = {}, {}
    for k in PIECES:
        x, r = read(kit / f"{k}.wav", n)
        if r != sr:
            raise SystemExit(f"{k}.wav is {r} Hz, the drum stem is {sr} Hz")
        pieces[k] = x
        shaped[k] = dsp.process(k, x, sr)
    resid = d - sum(pieces.values())
    print(f"residual (not captured by the split): {20 * np.log10(np.sqrt((resid ** 2).mean()) / np.sqrt((d ** 2).mean())):.1f} dB below the drums")
    vocal_paths = [p for p in others if "vocal" in p.stem.lower()]
    vocals = read(vocal_paths[0], n)[0] if vocal_paths else None
    for k, y in shaped.items():
        sf.write(out / "kit" / f"{k} (shaped).flac", y, sr, subtype="PCM_24")
    sf.write(out / "kit" / "residual (untouched).flac", resid, sr, subtype="PCM_24")
    if a.humanize_hh:
        shaped["hh"], st = humanize_hh(shaped["hh"], sr, vocals, seed=a.seed)
        sf.write(out / "kit" / "hh (humanized).flac", shaped["hh"], sr, subtype="PCM_24")
        print("hi-hat/shaker humanized:", st)
    bus, gr = peak_cap(sum(shaped.values()) + resid, d, sr)
    crest = lambda z: 20 * np.log10(np.abs(z).max() / np.sqrt((z ** 2).mean()))
    print(f"drum bus: peak cap max {gr:.2f} dB | crest {crest(d):.1f} -> {crest(bus):.1f} dB")
    sf.write(out / "drums (shaped).flac", bus, sr, subtype="PCM_24")
    rest = sum(read(p, n)[0] for p in others) if others else np.zeros_like(bus)
    mix = rest + bus
    sf.write(out / "mix.wav", mix, sr, subtype="FLOAT")                 # input for master.py
    variants = [("full", rest, d)]
    if vocals is not None:
        variants.append(("instrumental", rest - vocals, d))
    for name, base, dry in variants:
        new, old = base + bus, base + dry
        s = min(1.0, 0.966 / max(np.abs(new).max(), np.abs(old).max()))   # same gain as the original would need
        sf.write(out / f"{name} (shaped drums).flac", new * s, sr, subtype="PCM_24")
    print("written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems")
    ap.add_argument("out")
    ap.add_argument("--kit", help="folder with an earlier DrumSep split (kick.wav ... crash.wav)")
    ap.add_argument("--models", help="audio-separator model cache folder")
    ap.add_argument("--humanize-hh", action="store_true", help="break up a machine-steady hi-hat/shaker")
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args())
