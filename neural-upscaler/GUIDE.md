# 🎚️ Neural Audio Upscaler: the complete guide

*Written to be read cold, months later. Start at the top; every step says exactly what to click and type.*

🇧🇷 [Versão em português](GUIDE.pt-BR.md)

---

## Contents

1. [What this does](#1-what-this-does)
2. [What each step fixes](#2-what-each-step-fixes)
3. [One-time setup](#3-one-time-setup)
4. [How to pick files](#4-how-to-pick-files)
5. [Recipe: upscale your songs](#5-recipe-upscale-your-songs)
6. [Listening and judging the result](#6-listening-and-judging-the-result)
7. [Getting your results](#7-getting-your-results)
8. [Every setting explained](#8-every-setting-explained)
9. [Troubleshooting](#9-troubleshooting)
10. [How it works under the hood](#10-how-it-works-under-the-hood)

---

## 1. What this does

It **repairs the sound** of songs you already have: the damage from MP3/AAC compression and the missing high end. It does **not** recreate or regenerate the song. It's the same recording, cleaned up.

```
song.mp3 ─► ① vocal split ─┬─ vocal ─────────► ② Apollo ─┐
                           └─ instrumental ──► ② Apollo ─┴─► ③ AudioSR ─► ④ Matchering ─► song_upscaled.wav
```

It runs on **Google Colab**, reading songs from a folder in your **Google Drive** and writing the results back there.

---

## 2. What each step fixes

| Step | Fixes | Default | Cost |
|---|---|---|---|
| ① **Vocal split** (BS-RoFormer) | lets the vocal and the instrumental be repaired separately, and lets you rebalance the vocal. The two parts always add back up to the original | **on** | a minute or two |
| ② **Apollo** | compression damage: swirly or "underwater" cymbals, smeared drums, a dull top end | **on** | a few minutes per song |
| ③ **AudioSR** | **rebuilds the high frequencies an MP3 cut off** (e.g. everything above 16 kHz) and outputs 48 kHz | off | **slow**: 10+ minutes per song |
| ④ **Matchering** | mastering: copies the loudness, tone and stereo width of a **reference song** you choose | off | about a minute |

**AudioSR only adds.** It works out where your file's high end was cut off, keeps everything below that exactly as it was, and fills in only what's missing above it. If a song isn't missing any high end (a lossless master, say), AudioSR skips it on its own.

**Which steps should I use?**
- **First try:** the defaults (split + Apollo). Listen.
- **Still sounds dull or muffled?** Turn on AudioSR for that song.
- **Too quiet next to other music, or the tone is off?** Use Matchering with a reference song you like.

---

## 3. One-time setup

### 3.1 Put your songs in Google Drive
- Create a folder in *My Drive*, for example **`áudio`** (that's the default name; any name works).
- Put the songs in it. Any common format works: mp3, wav, flac, m4a, ogg…
- Results will appear in a new subfolder, `upscaled/`, inside the same folder. Your original files are never changed.

### 3.2 Open the notebook
- Open `neural-upscaler/Neural_Audio_Upscaler.ipynb` in the `guitorte/audio` repository on GitHub and click **Open in Colab** at the top, or use the link in the README.
- *File ▸ Save a copy in Drive* if you want to keep your own copy of the settings.

### 3.3 Choose a GPU
- **Runtime ▸ Change runtime type ▸ T4 GPU** (an **L4** is faster, if your plan has one).

> **Colab plan:** in our runs the upscaler worked on Colab's free tier, but Google decides what free sessions may run and can change that. If a run is cut off with a message about the no-cost tier, you'll need paid compute units. (There's no Kaggle version of the upscaler yet; ask if you need one.)

---

## 4. How to pick files

Cell 1 prints your songs with numbers:
```
📁 /content/drive/MyDrive/áudio  (3 songs)
  1. know better.mp3  (5.1 MB)
  2. relaxa.mp3  (7.8 MB)
  3. vida.flac  (31.2 MB)
```

`SONGS` and `MASTER_REFERENCE` accept:

| You type | It means |
|---|---|
| `"all"` | every song (`SONGS` only) |
| `"2"`, `"1, 3"`, `"1-3"` | numbers from the list |
| `"relaxa"` | the song whose name matches |
| `"know; vida"` | several names, separated by `;` or `,` |

An exact name wins, then names that *start* with what you typed, then names that *contain* it. Capitals and accents don't need to match.

---

## 5. Recipe: upscale your songs

1. **Run cell 1** (the ▶ button, or *Shift+Enter*).
   - Google asks for permission to access your Drive. Allow it.
   - Set `DRIVE_FOLDER` first if your folder isn't called `áudio`. A subfolder is written like `"Music/áudio"`.
   - Check that your songs are listed, and that the last line shows a **GPU** (not "⚠️ none").
2. **Cell 2: choose and run.** The defaults are a good first pass:
   ```
   SONGS = "all"            ← or e.g. "2" for one song
   SEPARATE_VOCALS = ✔
   APOLLO = ✔
   AUDIOSR = ☐              ← tick it for dull or muffled songs (slow)
   MASTER_REFERENCE = ""    ← optional: a song to copy the sound of
   ```
   Run it. For each song you'll see:
   ```
   🎧 relaxa.mp3
      input bandwidth ≈ 16.0 kHz (lossy shelf detected)
      ① separating vocals (BS-RoFormer)…
      ② Apollo restoring vocals…
      ② Apollo restoring instrumental…
   ✅ /content/drive/MyDrive/áudio/upscaled/relaxa_upscaled.wav
   ```
   The **first** run of a session spends a few minutes installing things. That's normal.
3. **Cell 3: compare.** See [section 6](#6-listening-and-judging-the-result).
4. **Want more?** Run cell 2 again with `AUDIOSR` ticked (and `OVERWRITE` ticked, so it redoes songs that are already finished).

---

## 6. Listening and judging the result

Cell 3 plays a 30-second excerpt **before** and **after**, at the **same loudness**. Louder always sounds "better", so the comparison only means something when the volumes match. It also draws two **spectrograms**, pictures of the frequencies over time:

- **An MP3 has a flat "ceiling"**: a hard horizontal line, often around 16 kHz, with nothing above it.
- **After AudioSR**, that space above the line is filled in.
- **After Apollo**, cymbals and consonants look crisper and less smeared.

**What to listen for:** cymbals and hi-hats (less swirl), the "s" sounds in vocals (less lisp), and air or openness at the top.
**Warning signs:** hiss or a metallic sheen at the top (AudioSR inventing too much; keep the Apollo-only version), or pumping, which suggests the mastering reference was a poor fit.

Set `START_AT_SECONDS` to jump to the part you care about, such as the chorus.

---

## 7. Getting your results

Everything is written straight into your Drive, in `upscaled/` next to your songs:

| File | What it is |
|---|---|
| `upscaled/<song>_upscaled.wav` | the repaired song (24-bit WAV; 48 kHz after AudioSR, otherwise 44.1 kHz) |
| `upscaled/<song>_upscaled.json` | the settings used, plus notes (bandwidth detected, gain changes…) |
| `upscaled/<song>_parts/vocals.wav` and `instrumental.wav` | the repaired stems, if you ticked `KEEP_PARTS` |

- **Notes printed under each result** explain what happened, for example `input bandwidth ≈ 16.0 kHz`, `AudioSR: new highs above 15.7 kHz`, `lowered 2.1 dB to avoid clipping`.
- "Lowered … dB to avoid clipping" is normal. Decoded MP3s often peak slightly above the maximum, so the song is turned down just enough not to distort.
- A song that's already done is skipped (`already exists`). Tick `OVERWRITE` to redo it.
- Matchering outputs at 44.1 kHz.

---

## 8. Every setting explained

**Cell 1**
| Setting | Default | Meaning |
|---|---|---|
| `DRIVE_FOLDER` | `"áudio"` | your folder inside *My Drive*. Subfolders like `"Music/áudio"` work |

**Cell 2**
| Setting | Default | Meaning / when to change |
|---|---|---|
| `SONGS` | `"all"` | which songs ([section 4](#4-how-to-pick-files)) |
| `SEPARATE_VOCALS` | ✔ | repair the vocal and the instrumental separately. Untick for instrumental music, or to save time |
| `APOLLO` | ✔ | repair compression damage. The main improvement for MP3s |
| `AUDIOSR` | ☐ | rebuild missing high frequencies. Slow; use it on dull songs |
| `AUDIOSR_STEPS` | `50` | AudioSR quality vs speed. 50 is fine; more is slower and rarely better |
| `CUTOFF_HZ` | `0` | `0` = find where the high end was cut automatically. Set e.g. `16000` to force AudioSR to rebuild everything above that frequency (for songs it would otherwise skip) |
| `VOCAL_GAIN_DB` | `0.0` | make the vocal louder (+) or quieter (−). Needs `SEPARATE_VOCALS` |
| `MASTER_REFERENCE` | `""` | a song from the list whose loudness and tone should be copied. Empty = no mastering. Pick a well-mastered song in a similar style |
| `KEEP_PARTS` | ☐ | also save the repaired vocal and instrumental stems |
| `OVERWRITE` | ☐ | redo songs that were already finished |

**Cell 3:** `START_AT_SECONDS` and `CLIP_SECONDS` choose the excerpt. Short songs are handled automatically.

---

## 9. Troubleshooting

**Where to look first:** the lines printed **above** an error explain it. The red traceback at the bottom only shows where it stopped.

| What you see | Why | Fix |
|---|---|---|
| `Folder 'áudio' not found in … Folders there: […]` | the folder name differs, or it's inside another folder | set `DRIVE_FOLDER` to a name from that list, e.g. `"Music/áudio"` |
| `GPU: ⚠️ none` | no GPU selected | *Runtime ▸ Change runtime type ▸ T4 GPU*, then run cell 1 again |
| `No file number 7` / `No file name contains '…'` | a typo, or the wrong number | use a name or number from cell 1's list |
| `already exists (tick OVERWRITE to redo)` | that song was done before | tick `OVERWRITE` |
| AudioSR prints `already full band, AudioSR skipped` | the song isn't missing any high end | nothing to do. To force it, set `CUTOFF_HZ` |
| hiss or a metallic sheen after AudioSR | AudioSR invented highs that don't suit the song | use the version without AudioSR, or raise `CUTOFF_HZ` |
| the first run takes a long time | installing tools, and AudioSR's first-time setup plus a several-GB download | normal, once per session |
| *"Runtime disconnected… no-cost tier"* | Colab's free tier stopped the run | paid compute units |
| an error about `python failed (exit 1)` with lines above it | a helper program failed; the lines above say why | read the lines above it, and send them to Claude if they're unclear |
| an old bug is back after an update | Colab is still running the old code | *Runtime ▸ Disconnect and delete runtime*, then run again |

---

## 10. How it works under the hood

*For whoever has to debug it later. Developer details are in `CLAUDE.md` at the repo root.*

- **Files:**
  - `upscaler.py`: everything the notebook calls.
  - `audiosr_worker.py`: runs AudioSR.
  - `upscaler_dsp.py`: small audio maths (finding the cutoff, the crossover, crossfades).
- **The notebook is thin:** each session it downloads this repository (branch `ccr-8be874fb-rdbsnf`, or the default branch if that's gone) and calls `upscaler.py`.
- **AudioSR needs old library versions**, so it runs in its own Python 3.10 environment, built automatically the first time.
- **Apollo** runs from a pinned copy of its source code.
- **Lossless split:** the instrumental is computed as *song minus vocal*, so the two parts always add back up to exactly the original.
- **AudioSR is careful:**
  - Below the detected cutoff, the result is your original audio.
  - Only the band above it comes from AudioSR, and it's matched in level.
  - Stereo is processed as middle and sides, which keeps the new highs consistent between left and right.
- **Scratch files** go in `/content/upscaler_work` and vanish when the session ends.
