# 🥁 Stem mastering: reshape the drums, then master the song

🇧🇷 [Versão em português](README.pt-BR.md)

*Written to be read cold, months later.* This folder holds the process used on **raio-x** (Nov 2026): take a song's stems, split the drum stem into its kit pieces, give each piece more punch and a cleaner tone, tame a machine-steady shaker, put the song back together and master it.

It's a set of command-line scripts, not a notebook yet. Everything runs on CPU; a GPU only speeds up the separation.

```
stems/ (drums, bass, guitar, other, piano, vocals)
   │
   ├─ drums ──► separate.py (DrumSep) ──► kick · snare · toms · hh · ride · crash  (+ residual)
   │                                         │
   │                          drums.py: each piece → transient shaper → EQ → compressor
   │                                    hh (optional) → humanize: per-hit levels, duck under vocal
   │                                         │
   │                          shaped pieces + residual ──► peak cap ──► drums (shaped)
   │                                                                        │
   └─ the other stems ─────────────────────────────────────────────────────┴──► mix.wav
                                                                                  │
                                              master.py ──► master (−13 LUFS, −1 dBTP) + A/B copies
                                              check.py  ──► what changed, at equal loudness
```

---

## 1. Setup

```bash
pip install "audio-separator[cpu]" audioread librosa pyloudnorm soundfile scipy   # [gpu] on Colab
cd stem-mastering
```

`audioread` is needed because audio-separator imports it without declaring it. Models download on first use (DrumSep is about 400 MB).

## 2. Recipe

**If you only have the song,** get a drum stem first (13 min on 2 CPU cores for a 2.5 min song; seconds on a GPU):

```bash
python separate.py song.flac htdemucs_ft.yaml stems/
```

**With stems** (one file per stem; the drum stem's name contains `drum`, the vocal's `vocal`):

```bash
python drums.py stems/ out/ --humanize-hh          # splits the drums itself (≈10 min on CPU)
python drums.py stems/ out/ --kit out/kit_split    # reuse an earlier split
python master.py out/mix.wav out/master.wav --target -13 --ab-level -15
python check.py out/mix.wav out/master.wav --drums "out/drums (shaped).flac" --vocals stems/vocals.flac
```

Leave out `--humanize-hh` if the hi-hat sounds fine. Encode MP3s from the 24-bit master with `ffmpeg -i master.wav -b:a 320k master.mp3`.

**What you get in `out/`:**

| File | What it is |
|---|---|
| `kit/<piece> (shaped).flac` | each kit piece after its chain; `residual (untouched).flac` is what the split missed |
| `kit/hh (humanized).flac` | the hi-hat/shaker after humanizing (with `--humanize-hh`) |
| `drums (shaped).flac` | the new drum stem |
| `full (shaped drums).flac`, `instrumental (shaped drums).flac` | the song with the new drums, 24-bit, unmastered |
| `mix.wav` | the same full mix in 32-bit float: the input for `master.py` |
| `master.wav`, `master_AB_mix.wav`, `master_AB_master.wav` | the master, and mix and master at the same loudness for a fair comparison |

## 3. The drum chains (`dsp.CHAINS`)

A **transient shaper** follows each piece's level with a fast and a slow envelope. Where the fast one is above the slow one a hit is starting (attack); where it's below, the hit is dying away (sustain). Each part gets its own gain, so a kick can hit harder and ring shorter.

| Piece | Attack / sustain | EQ | Compressor |
|---|---|---|---|
| kick | +5 / −3 dB | +2.5 dB at 60 Hz, −3 at 320 Hz (boxiness), +3 at 3.8 kHz (beater click) | 3:1, 10 ms / 90 ms |
| snare | +4 / +1 dB | +2 at 200 Hz (body), −2 at 500 Hz, +2.5 at 5.5 kHz (crack), +1.5 shelf at 11 kHz | 2.5:1 |
| toms | +4 / +1.5 dB | −2.5 at 450 Hz (mud), +2 at 4 kHz | 2:1 |
| hh | +2 / −2 dB | −2 at 400 Hz, +1.5 shelf at 10 kHz | none |
| ride, crash | +1 to +1.5 dB attack | a touch of air | none |

Every piece is **level-matched back to its input** (RMS), so the kit's balance stays where the song had it: the drums get punchier, not louder. A look-ahead **peak cap** keeps the new drum bus within 1 dB of the original stem's peak.

On raio-x: the residual sat 26.7 dB below the drums, the bus crest factor went from 19.5 to 20.6 dB, and the peak cap worked 1.3 dB at most.

## 4. Humanizing a robotic shaker (`--humanize-hh`)

Generated or heavily quantized tracks often have a shaker or hi-hat playing straight eighths at the same level from start to finish. It sounds like a metronome. On raio-x it ran from 0:00 to 2:08, within about 3 dB the whole time, locked to the grid (14 ms of timing spread).

What the option does to that layer:

- **Per-hit levels.** Hits on the beat stay at 0 dB and hits between beats drop 2.5 dB. A slow drift (±1 dB over 8- and 3-bar cycles) and a small seeded random offset (σ 1.2 dB, capped at ±2.5) go on top. The rhythm is untouched; only the hits' levels change.
- **Ducking:** up to 3 dB lower while the vocal sings (needs a vocal stem).
- **Level and tone:** 3 dB lower overall, −2 dB shelf above 9 kHz.

**How the beat is found:** onsets come from the layer's 4–16 kHz band. The grid is the period the hits line up with best, searched within ±3% of their median spacing. Two simpler methods failed on raio-x:
- **The median spacing alone** came out at 0.1858 s instead of 0.1875 s, because onset jitter biases it.
- **Counting from hit to hit** flips every on/off label after one missed hit.

Check this by ear: if the accents land off the beat, the louder half of the hits wasn't the on-beat one.

On raio-x the layer ended up 7 dB quieter on average, because the vocal sings 83% of the time.

## 5. The master (`master.py`)

| # | Stage | Setting | Why |
|---|---|---|---|
| 1 | high-pass | 25 Hz, zero phase | sub rumble; zero phase so it doesn't make peaks taller |
| 2 | dynamic low cut | 65–100 Hz, 2:1, up to 3 dB | tames a bass note only when it swells |
| 3 | static EQ (`dsp.MASTER_EQ`) | −1.5 dB at 180 Hz, +2.5 at 2.5 kHz, +1.5 at 5 kHz, +2.5 shelf at 11 kHz | mud, presence hole, air (typical of separated stems) |
| 4 | de-esser | centre channel, 5–9 kHz, up to 3 dB | the brightening shouldn't sharpen the "s" sounds |
| 5 | glue compressor | 1.5:1, 30 ms attack, 300 ms release | ~1 dB at most; the slow attack lets the reshaped drum hits through |
| 6 | gain ride | 3 s windows, corrects drifts > 0.7 dB, 2.5 s glides | keeps the arrangement's proportions (see below) |
| 7 | limiter | look-ahead, 4× oversampled detector, −1.2 dBFS | lands at −1 dBTP; output gain iterated to the target |

**Why the gain ride:** EQ moves sections with different instruments by different amounts. On raio-x the outro is a single midrange part with no bass. The 80 Hz cut didn't lower it and the presence boost lifted it in full, so it came out 2.3 dB louder than the rest. The ride puts every section back within about 0.6 dB of its level in the mix.

**Loudness target:** −13 LUFS by default. The other masters made in Nov 2026 sit between −12 and −13.5. After reshaping drums, don't push higher than the limiter can take quietly: a limiter working hard cancels the extra attack (Katz, *Mastering Audio*, ch. 11, "Compromises When Making Hot Masters"). Owsinski suggests no more than 2–3 dB of limiting and a −1 dB ceiling for online delivery (*The Mastering Engineer's Handbook*, "Setting The Limiter").

**raio-x at −13 LUFS** (all at equal loudness):
- **Limiter:** 0.05 dB on the loudest 1% of moments; 6 dB on a few isolated millisecond peaks.
- **Bass and presence:** 80 Hz −2.7 dB, 2.5–5 kHz +2.3 to +2.5 dB.
- **Dynamics:** PLR 11.8 dB, loudness range 2.4 LU (the mix had 2.5).
- **Drums:** attack rise 10.2 → 10.5 dB, so the punch survived.
- **Vocal:** 0.8 LU further back against the instruments, because the brightening favours the hi-hats and guitars.

## 6. Checking the result (`check.py`)

Always compare at equal loudness: the louder version wins any quick A/B. `master.py --ab-level -15` writes both at −15 LUFS. Pick a level below the mix's own if the mix peaks above 0 dBTP, or its copy will clip as an MP3.

`check.py` prints:
- **Tone change** per band, at equal loudness.
- **Section balance:** how far each 3 s window drifted against the mix.
- **Mono loss** at 500 Hz–4 kHz. Worry past −3 dB.
- **PLR.**
- **Drum attack rise**, with `--drums`.
- **How far the master's EQ moves the vocal** against the rest, with `--vocals`.

## 7. Pitfalls already hit

- **Stems don't sum to the mix unless you keep their level.** `separate.py` feeds audio-separator the input at 0.5 peak with `normalization_threshold=1.0`, then divides the stems back (same trick as `neural-upscaler/upscaler.py::separate_vocals`).
- **audio-separator writes 44.1 kHz.** If the source is 48 kHz, resample before subtracting or summing. Mismatched rates made a drum split look 2.8 dB *louder* than the mix it came from.
- **Filters must start settled.** A one-pole smoother starting from zero fades the first milliseconds in, and the peak cap read that as 33 dB of gain reduction. `dsp.env` initialises its state from the first sample.
- **DrumSep has no "percussion" output.** Shakers and tambourines land in `hh`, and low percussion often lands in `toms`. On raio-x, "toms" was the loudest piece.
- **A constant shaker-like hiss is usually a part, not noise.** Check whether it locks to the tempo and stops with the drums before treating it as noise.
