"""DSP building blocks for stem-mastering (numpy/scipy only).

Per-piece drum tone: transient shaper + EQ + gentle compression.

Transient shaper: two envelope followers (fast and slow) on the piece's level; where the fast one
rises above the slow one the hit is starting (attack), where it falls below the hit is decaying
(sustain). Gain = attack_db * attack_amount + sustain_db * sustain_amount, smoothed, applied to stereo.
"""
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, lfilter

def env(x, sr, ms):
    """One-pole smoother with time constant `ms`, starting settled at the first value."""
    a = np.exp(-1.0 / (sr * ms / 1000))
    y, _ = lfilter([1 - a], [1, -a], x, zi=[a * x[0]])   # start settled at the first value
    return y

def transient_shaper(x, sr, attack_db=0.0, sustain_db=0.0, fast_ms=1.0, slow_ms=40.0, smooth_ms=2.0):
    mono = np.abs(x).mean(axis=1) + 1e-9
    lvl = 20 * np.log10(np.maximum(np.sqrt(env(mono ** 2, sr, 0.3)), 1e-7))
    f, s = env(lvl, sr, fast_ms), env(lvl, sr, slow_ms)
    d = np.clip((f - s) / 6.0, -1, 1)               # +1 = clear attack, -1 = clear decay (6 dB span)
    g_db = np.where(d > 0, d * attack_db, -d * sustain_db)
    g_db = env(g_db, sr, smooth_ms)
    return x * (10 ** (g_db / 20))[:, None]

def peq(x, sr, f0, gain_db, q=1.0, kind="bell"):
    """RBJ biquad (bell / low shelf / high shelf)."""
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f0 / sr; al = np.sin(w) / (2 * q); c = np.cos(w)
    if kind == "bell":
        b = [1 + al * A, -2 * c, 1 - al * A]; a = [1 + al / A, -2 * c, 1 - al / A]
    else:
        sA = 2 * np.sqrt(A) * al
        if kind == "low":
            b = [A * ((A + 1) - (A - 1) * c + sA), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sA)]
            a = [(A + 1) + (A - 1) * c + sA, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sA]
        else:
            b = [A * ((A + 1) + (A - 1) * c + sA), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sA)]
            a = [(A + 1) - (A - 1) * c + sA, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sA]
    return lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)

def hpf(x, sr, f):
    return sosfiltfilt(butter(2, f, "high", fs=sr, output="sos"), x, axis=0)

def comp(x, sr, thr_db=-18, ratio=2.5, att_ms=8, rel_ms=120):
    lvl = 20 * np.log10(np.abs(x).max(axis=1) + 1e-9)
    over = np.maximum(lvl - thr_db, 0)
    gr = over * (1 - 1 / ratio)
    a_a, a_r = np.exp(-1 / (sr * att_ms / 1000)), np.exp(-1 / (sr * rel_ms / 1000))
    out = np.zeros_like(gr); g = 0.0
    for i, v in enumerate(gr):   # plain loop; comp_fast() runs it at 1/16 rate
        g = a_a * g + (1 - a_a) * v if v > g else a_r * g + (1 - a_r) * v
        out[i] = g
    return x * (10 ** (-out / 20))[:, None]

def comp_fast(x, sr, **kw):
    # run the detector at 1/16 rate, upsample the gain
    step = 16
    dec = np.abs(x).max(axis=1).reshape(-1)[: len(x) // step * step].reshape(-1, step).max(axis=1)
    g = comp(np.stack([dec, dec], 1), sr / step, **kw)[:, 0] / np.maximum(dec, 1e-12)
    g = np.repeat(g, step); g = np.concatenate([g, np.full(len(x) - len(g), g[-1])])
    return x * env(g, sr, 1.0)[:, None]

# piece -> chain. Kept moderate: the goal is punch and clarity, not a different kit.
CHAINS = {
    "kick":  dict(hp=28,  ts=(5.0, -3.0, 1.0, 50.0), eq=[(60, 2.5, 1.0, "bell"), (320, -3.0, 1.2, "bell"), (3800, 3.0, 1.0, "bell")], comp=(-20, 3.0, 10, 90)),
    "snare": dict(hp=90,  ts=(4.0, 1.0, 1.0, 40.0),  eq=[(200, 2.0, 1.2, "bell"), (500, -2.0, 1.5, "bell"), (5500, 2.5, 0.8, "bell"), (11000, 1.5, 0.7, "high")], comp=(-20, 2.5, 6, 120)),
    "toms":  dict(hp=60,  ts=(4.0, 1.5, 1.0, 50.0),  eq=[(450, -2.5, 1.2, "bell"), (4000, 2.0, 1.0, "bell")], comp=(-20, 2.0, 10, 150)),
    "hh":    dict(hp=250, ts=(2.0, -2.0, 0.7, 25.0), eq=[(400, -2.0, 0.8, "bell"), (10000, 1.5, 0.7, "high")], comp=None),
    "ride":  dict(hp=250, ts=(1.5, 0.0, 0.7, 30.0),  eq=[(9000, 1.0, 0.7, "high")], comp=None),
    "crash": dict(hp=200, ts=(1.0, 0.0, 1.0, 40.0),  eq=[(9000, 1.0, 0.7, "high")], comp=None),
}

def process(name, x, sr):
    """Run one kit piece through its chain and level-match it back to its input (RMS)."""
    c = CHAINS[name]
    y = hpf(x, sr, c["hp"])
    a, s, f, sl = c["ts"]
    y = transient_shaper(y, sr, a, s, f, sl)
    for f0, gdb, q, kind in c["eq"]:
        y = peq(y, sr, f0, gdb, q, kind)
    if c["comp"]:
        t, r, at, re = c["comp"]
        y = comp_fast(y, sr, thr_db=t, ratio=r, att_ms=at, rel_ms=re)
    # level-match to the original piece (RMS), so the balance stays where the song had it
    return y * (np.sqrt((x ** 2).mean()) / max(np.sqrt((y ** 2).mean()), 1e-12))


# The master's static EQ (master.py step 3), shared with check.py.
MASTER_EQ = [(180, -1.5, 1.1, "bell"),     # mud
             (2500, 2.5, 0.9, "bell"),     # presence
             (5000, 1.5, 1.0, "bell"),
             (11000, 2.5, 0.7, "high")]    # air


def eq(x, sr, bands):
    for f0, gdb, q, kind in bands:
        x = peq(x, sr, f0, gdb, q, kind)
    return x
