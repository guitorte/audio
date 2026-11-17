"""Master a stereo mix: one fixed chain, measured at every stage.

Chain (see README for why each stage is there):
  1. 25 Hz high-pass, zero phase
  2. dynamic cut at 80 Hz (65-100 Hz band, up to 3 dB, only when the band swells)
  3. static EQ: -1.5 dB at 180 Hz, +2.5 dB at 2.5 kHz, +1.5 dB at 5 kHz, +2.5 dB shelf at 11 kHz
  4. de-esser on the centre (mid) channel, 5-9 kHz, up to 3 dB
  5. glue compressor: 1.5:1, 30 ms attack (drum hits pass), 300 ms release, ~1 dB at most
  6. gain ride: sections whose level moved by more than 0.7 dB against the mix (3 s windows)
     are glided back, so the arrangement keeps its proportions
  7. look-ahead brick-wall limiter, detector on 4x oversampled peaks, ceiling -1.2 dBFS
     (lands at about -1 dBTP), output gain iterated to the target LUFS

usage: python master.py <mix.wav> <out.wav> [--target -13] [--ab-level -15]
Writes the 24-bit master, prints what every stage did, and with --ab-level also writes
<out>_AB_mix.wav / <out>_AB_master.wav at the same loudness for a fair comparison.
"""
import argparse
import numpy as np, soundfile as sf, pyloudnorm as pyln
from scipy import signal
from scipy.ndimage import minimum_filter1d
from dsp import peq, env, eq, MASTER_EQ

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("mix"); ap.add_argument("out")
ap.add_argument("--target", type=float, default=-13.0, help="integrated LUFS (default -13)")
ap.add_argument("--ab-level", type=float, help="also write level-matched A/B copies at this LUFS")
args = ap.parse_args()
target, outp = args.target, args.out
x, sr = sf.read(args.mix, always_2d=True, dtype="float64")
m = pyln.Meter(sr)

def band(z, lo, hi, o=2):
    return signal.sosfiltfilt(signal.butter(o, [lo, hi], 'bandpass', fs=sr, output='sos'), z, axis=0)

def dyn_bell(z, f0, q, max_cut, lo, hi, thr_pct=60, att=10, rel=150):
    """Dynamic band cut (parallel band-pass lo-hi): the louder the band gets above its threshold, the more it is cut, 2:1, up to max_cut. f0 and q are kept for reference only."""
    b = band(z.mean(1), lo, hi)
    lvl = 10 * np.log10(env(b ** 2, sr, 5) + 1e-12)
    thr = np.percentile(lvl[lvl > lvl.max() - 60], thr_pct)
    over = np.clip(lvl - thr, 0, None)
    g = -np.minimum(over * 0.5, max_cut)                     # 2:1 above threshold, capped
    a_a, a_r = np.exp(-1 / (sr * att / 1000)), np.exp(-1 / (sr * rel / 1000))
    gs = signal.lfilter([1 - a_r], [1, -a_r], g)             # smooth (release-ish)
    gs = np.minimum(gs, signal.lfilter([1 - a_a], [1, -a_a], g))
    # apply as a parallel band: z + (10^(g/20)-1) * bell-band
    bb = band(z, lo, hi)
    return z + bb * (10 ** (gs / 20) - 1)[:, None], gs

stats = {}
# 1. sub cleanup, zero phase
y = signal.sosfiltfilt(signal.butter(2, 25, 'high', fs=sr, output='sos'), x, axis=0)
# 2. the 80 Hz boom: dynamic, up to 3 dB
y, g80 = dyn_bell(y, 80, 1.2, 3.0, 65, 100)
stats['80 Hz dynamic cut mean/max'] = (-g80.mean(), -g80.min())
# 3. static tone: mud, presence, air
y = eq(y, sr, MASTER_EQ)
# 4. de-esser on the centre (vocal) 5-9 kHz, up to 3 dB
M, S = (y[:, 0] + y[:, 1]) / 2, (y[:, 0] - y[:, 1]) / 2
Mm, gs = dyn_bell(np.stack([M, M], 1), 7000, 1, 3.0, 5000, 9000, thr_pct=97, att=1, rel=60)
M = Mm[:, 0]; stats['de-esser max'] = -gs.min(); stats['de-esser active % (>0.5 dB)'] = (gs < -0.5).mean() * 100
y = np.stack([M + S, M - S], 1)
# 5. glue: wideband, slow attack so the shaped drum hits pass, ~1 dB
lvl = 20 * np.log10(np.sqrt(env(y.mean(1) ** 2, sr, 30)) + 1e-9)
thr = np.percentile(lvl, 80)
gr = np.clip(lvl - thr, 0, None) * (1 - 1 / 1.5)
a_a, a_r = np.exp(-1 / (sr * 0.030)), np.exp(-1 / (sr * 0.300))
gcur = 0.0
dec = 32
grd = gr[::dec]; out = np.zeros_like(grd)
aa, ar = a_a ** dec, a_r ** dec
for i, v in enumerate(grd):
    gcur = aa * gcur + (1 - aa) * v if v > gcur else ar * gcur + (1 - ar) * v
    out[i] = gcur
grs = np.interp(np.arange(len(gr)), np.arange(len(out)) * dec, out)
y = y * (10 ** (-grs / 20))[:, None]
stats['glue GR median/95pct/max'] = (np.median(grs), np.percentile(grs, 95), grs.max())

# 6. macro-dynamics: ride sections back to the level relationship they had in the mix
#    (EQ moves sections with different instrumentation by different amounts)
win, hop = 3 * sr, sr // 2
cents = np.arange(win // 2, len(x) - win // 2, hop)
dev = []
for c in cents:
    a = m.integrated_loudness(x[c - win // 2:c + win // 2]); b = m.integrated_loudness(y[c - win // 2:c + win // 2])
    dev.append(b - a if np.isfinite(a) and np.isfinite(b) and a > -40 else np.nan)
dev = np.array(dev); med = np.nanmedian(dev); dev = np.nan_to_num(dev - med)
corr = -np.where(np.abs(dev) > 0.7, np.clip(dev, -3, 3), 0.0)
corr = np.convolve(corr, np.ones(5) / 5, 'same')            # ~2.5 s glide
ride = np.interp(np.arange(len(y)), cents, corr)
y = y * (10 ** (ride / 20))[:, None]
stats['gain ride min/max dB'] = (ride.min(), ride.max())


def limit(z, ceil_db=-1.2, la_ms=1.5, rel_ms=80):
    """Look-ahead brick-wall limiter; detector on 4x oversampled peaks."""
    up = np.abs(signal.resample_poly(z, 4, 1, axis=0)).max(1)
    pk = up.reshape(-1, 4).max(1)[:len(z)]
    ceil = 10 ** (ceil_db / 20); la = int(la_ms / 1000 * sr)
    need = np.minimum(1, ceil / np.maximum(pk, 1e-12))
    held = minimum_filter1d(need, 2 * la + 1)                 # hold over the look-ahead window
    a = np.exp(-1 / (sr * rel_ms / 1000))
    rel = 1 - signal.lfilter([1 - a], [1, -a], 1 - held, zi=[0])[0]   # gain recovers with release
    g = np.minimum(rel, held)
    g = np.convolve(g, np.ones(la) / la, 'same')              # glide into each peak (stays <= need)
    return z * g[:, None], -20 * np.log10(g)

gain = target - m.integrated_loudness(y)
for _ in range(5):
    out, grd = limit(y * 10 ** (gain / 20))
    L = m.integrated_loudness(out); gain += target - L
w = sr // 100; k = len(grd) // w; f10 = grd[:k * w].reshape(k, w).max(1)
stats['limiter GR 10ms frames: median / 99pct / max'] = (np.median(f10), np.percentile(f10, 99), f10.max())
stats['limiter time >1 dB %'] = (grd > 1).mean() * 100
stats['final LUFS'] = L
sf.write(outp, out, sr, subtype='PCM_24')
for k_, v in stats.items():
    print(k_, np.round(v, 2))
if args.ab_level is not None:
    stem = outp.rsplit('.', 1)[0]
    for z, tag in ((x, 'mix'), (out, 'master')):
        sf.write(f'{stem}_AB_{tag}.wav', z * 10 ** ((args.ab_level - m.integrated_loudness(z)) / 20), sr, subtype='PCM_24')
