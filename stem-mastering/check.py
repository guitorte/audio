"""Compare a master against its mix: what changed, measured at equal loudness.

usage: python check.py <mix.wav> <master.wav> [--drums <drum stem>] [--vocals <vocal stem>]

Prints:
- tone change per band (master turned down to the mix's loudness first)
- section balance: how far each 3 s window moved relative to the whole song
- mono loss in the 500 Hz-4 kHz range, and peak-to-loudness ratio (PLR)
- with --drums: median attack rise of strong drum hits (did the punch survive?)
- with --vocals: the vocal's level against everything else, and how much the master's static EQ
  (dsp.MASTER_EQ) moves it (dynamics excluded; they act on the whole mix alike)
"""
import argparse

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal

from dsp import eq, MASTER_EQ


def bp(z, sr, lo, hi):
    return signal.sosfiltfilt(signal.butter(4, [lo, hi], "bandpass", fs=sr, output="sos"), z, axis=0)


def main(a):
    x, sr = sf.read(a.mix, always_2d=True)
    y, _ = sf.read(a.master, always_2d=True)
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    m = pyln.Meter(sr)
    Lx, Ly = m.integrated_loudness(x), m.integrated_loudness(y)
    yl = y * 10 ** ((Lx - Ly) / 20)
    up = np.abs(signal.resample_poly(y, 4, 1, axis=0)).max()
    print(f"mix {Lx:.1f} LUFS -> master {Ly:.1f} LUFS, true peak {20 * np.log10(up):.1f} dBTP, PLR {20 * np.log10(up) - Ly:.1f} dB")
    f, Px = signal.welch(x.mean(1), sr, nperseg=8192)
    _, Py = signal.welch(yl.mean(1), sr, nperseg=8192)
    bands = [40, 80, 200, 500, 1000, 2500, 5000, 10000, 14000]
    print("tone change at equal loudness:", " ".join(
        f"{c}:{10 * np.log10(Py[(f >= c / 1.26) & (f < c * 1.26)].mean() / Px[(f >= c / 1.26) & (f < c * 1.26)].mean()):+.1f}"
        for c in bands))
    dev = []
    for t in range(0, n // sr - 3, 3):
        s = slice(t * sr, (t + 3) * sr)
        lx, ly = m.integrated_loudness(x[s]), m.integrated_loudness(yl[s])
        if np.isfinite(lx) and np.isfinite(ly) and lx > -40:
            dev.append((t, ly - lx))
    d = np.array([v for _, v in dev]); d -= np.median(d)
    worst = dev[int(np.argmax(np.abs(d)))][0]
    print(f"section balance vs mix: 95% of 3 s windows within {np.percentile(np.abs(d), 95):.2f} dB, "
          f"worst {np.abs(d).max():.2f} dB at {worst // 60}:{worst % 60:02d}")
    mono = 10 * np.log10((bp(y.mean(1), sr, 500, 4000) ** 2).mean()
                         / ((bp(y[:, 0], sr, 500, 4000) ** 2 + bp(y[:, 1], sr, 500, 4000) ** 2).mean() / 2))
    print(f"mono loss 500 Hz-4 kHz: {mono:.1f} dB (worry past -3 dB)")
    vi = lambda z: 10 * np.log10((bp(z.mean(1), sr, 1000, 4000) ** 2).mean() / (bp(z.mean(1), sr, 150, 1000) ** 2).mean())
    print(f"presence index (1-4 kHz vs 150 Hz-1 kHz): {vi(x):.2f} -> {vi(y):.2f} dB")
    if a.vocals:
        v, _ = sf.read(a.vocals, always_2d=True); v = v[:n]
        r = x - v
        base = m.integrated_loudness(v) - m.integrated_loudness(r)
        moved = m.integrated_loudness(eq(v, sr, MASTER_EQ)) - m.integrated_loudness(eq(r, sr, MASTER_EQ)) - base
        print(f"vocal vs rest: {base:+.1f} LU in the mix; the master's EQ moves it {moved:+.2f} LU")
    if a.drums:
        dr, _ = sf.read(a.drums, always_2d=True); dr = dr[:n]
        on = np.where(np.diff((np.abs(bp(dr.mean(1), sr, 40, 8000)) > 0.3).astype(int)) == 1)[0]

        def punch(z):
            M = z.mean(1); r = []
            for i in on:
                if sr * 0.02 < i < len(M) - sr * 0.02:
                    pre = np.sqrt((M[i - int(.02 * sr):i - int(.005 * sr)] ** 2).mean())
                    r.append(20 * np.log10(np.abs(M[i:i + int(.005 * sr)]).max() / (pre + 1e-9)))
            return np.median(r) if r else float("nan")
        print(f"drum attack rise ({len(on)} strong hits): mix {punch(x):.1f} dB -> master {punch(yl):.1f} dB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mix"); ap.add_argument("master")
    ap.add_argument("--drums"); ap.add_argument("--vocals")
    main(ap.parse_args())
