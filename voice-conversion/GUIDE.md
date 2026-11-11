# 🎤 Voice Conversion: the complete guide

*Written to be read cold, months later. Start at the top; every step says exactly what to click and type.*

🇧🇷 [Versão em português](GUIDE.pt-BR.md)

---

## Contents

1. [What this does](#1-what-this-does)
2. [Which way do I go?](#2-which-way-do-i-go)
3. [One-time setup on Kaggle](#3-one-time-setup-on-kaggle)
4. [How to pick files in any cell](#4-how-to-pick-files-in-any-cell)
5. [Recipe A: quick conversion with Seed-VC (no training)](#5-recipe-a-quick-conversion-with-seed-vc-no-training)
6. [Recipe B: train your own RVC voice](#6-recipe-b-train-your-own-rvc-voice)
7. [Recipe C: convert songs with your RVC voice](#7-recipe-c-convert-songs-with-your-rvc-voice)
8. [Recipe D: octave check (optional)](#8-recipe-d-octave-check-optional)
9. [Getting your results out](#9-getting-your-results-out)
10. [Every setting explained](#10-every-setting-explained)
11. [Using Google Colab instead](#11-using-google-colab-instead)
12. [Troubleshooting](#12-troubleshooting)
13. [How it works under the hood](#13-how-it-works-under-the-hood)

---

## 1. What this does

It **replaces the singer** in a song with another voice. Everything else in the song stays as it is.

```
your song ──► split ──┬── vocal ───────► new voice ──► matched to the original volume ──┐
                      └── instrumental (untouched) ───────────────────────────────────────┴──► finished song
```

You also get the new vocal on its own, so you can mix it yourself in a DAW.

There are **two engines** (two different ways of making the new voice):

| | **Seed-VC** | **RVC** |
|---|---|---|
| What you give it | **one short clip** (10–30 s) of the target voice | a **trained voice model** (a `.pth` file) |
| Training | **none**: it works immediately | you train once from ~10+ minutes of recordings (1–3 h of GPU time) |
| Best for | trying a voice out quickly | the most convincing result for a voice you'll use again and again |

> ⚖️ Only convert voices you have the right to use, such as your own or a collaborator's who has agreed.

---

## 2. Which way do I go?

**Platform: use Kaggle.** It's free, and it lets this run. Google Colab's *free* tier has disconnected this notebook mid-run as a "restricted workload". Colab only works with paid compute units (see [section 11](#11-using-google-colab-instead)).

**Then:**

| I want to… | Do this |
|---|---|
| Hear a song in another voice, right now | [Recipe A: Seed-VC](#5-recipe-a-quick-conversion-with-seed-vc-no-training) |
| Make a reusable, high-quality voice | [Recipe B: train RVC](#6-recipe-b-train-your-own-rvc-voice), then [Recipe C](#7-recipe-c-convert-songs-with-your-rvc-voice) |
| I already have an RVC model (`.pth`) | [Recipe C](#7-recipe-c-convert-songs-with-your-rvc-voice) |
| Know whether I need to shift the octave | [Recipe D](#8-recipe-d-octave-check-optional) |

---

## 3. One-time setup on Kaggle

You do this once. After that, every session starts at step 3.5.

### 3.1 Kaggle account
- Sign up at [kaggle.com](https://www.kaggle.com).
- **Verify your phone number** (*Settings ▸ Phone verification*). Without it, the notebook can't use the internet, and it needs the internet to download the AI models.

### 3.2 Put your audio in a Kaggle Dataset
A **Dataset** is Kaggle's word for a folder of files you upload.

1. *Create* (the **+** button) ▸ **New Dataset**.
2. Drag in your files. **Put everything at the top level**, with no folders needed:
   ```
   my-songs/
   ├── relaxa.mp3            ← songs you want to convert
   ├── displicente.wav       ← a voice clip for Seed-VC
   ├── maria take 1.wav      ← recordings for training an RVC voice
   ├── maria take 2.wav
   └── joão.pth              ← (optional) an RVC model you already have, plus joão.index if you have it
   ```
3. Give it a name (e.g. `my-songs`) and click **Create**. Keep it **Private**.

**Tips for the files:**
- Any common format works: mp3, wav, flac, m4a, ogg…
- **Clean, dry vocal stems** (no reverb, no instruments) give the best results, both as voice clips and for training. Full songs also work, because the notebook can isolate the vocal for you.
- **Names matter a little:** you'll pick files by number or by part of their name (see [section 4](#4-how-to-pick-files-in-any-cell)). A shared prefix like `maria take …` makes training sets easy to select.

**Adding files later:** open the dataset ▸ **New Version**, then add the files. A notebook only sees the new files after you re-attach the dataset or restart the session.

### 3.3 Get the notebook into Kaggle
1. Open this file on GitHub: `voice-conversion/Voice_Conversion_Kaggle.ipynb` in the `guitorte/audio` repository.
2. Download it (the **Download raw file** button).
3. On Kaggle: *Create* ▸ **New Notebook** ▸ **File ▸ Import Notebook** ▸ upload the file.

> When this guide or the code changes, repeat this step to get the new version. The notebook downloads the rest of the code fresh each session, but the notebook file itself only updates when you re-import it.

### 3.4 Notebook settings (right-hand panel)
- **Accelerator ▸ GPU T4 x2.** *Not* the P100: RVC doesn't work on it.
- **Internet ▸ On.**

### 3.5 Attach your dataset (every new notebook)
- **Add Input** (right panel) ▸ *Your Datasets* ▸ pick `my-songs`.

You're ready. Go to the recipe you need.

---

## 4. How to pick files in any cell

Cell 1 prints every audio file with a number:
```
📁 /kaggle/input/my-songs  (5 songs)
  1. displicente.wav
  2. maria take 1.wav
  3. maria take 2.wav
  4. relaxa.mp3
  5. vida.mp3
```

Any setting that asks for files (`SONGS`, `REFERENCE`, `DATASET`, `PITCH_REFERENCE`, `SOURCE`, `TARGET`) accepts:

| You type | It means |
|---|---|
| `"4"` | file number 4 |
| `"4, 5"` or `"2-3"` | several files, or a range |
| `"relaxa"` | the file whose name matches |
| `"maria take"` | every file starting with that (here takes 1 and 2) |
| `"all"` | every file (`SONGS` and `DATASET` only) |
| `"relaxa; vida"` | several names (separate them with `;` or `,`) |

**Matching rules:** an exact name wins, then names that *start* with what you typed, then names that *contain* it. So `"male"` picks `male vocal.wav`, not `female vocal.wav`. Capital letters and accents don't need to match exactly. If what you typed matches more than one file where only one is allowed, you get an error saying which files it matched. Just be more specific.

---

## 5. Recipe A: quick conversion with Seed-VC (no training)

**You need:** a song, plus a clip of the target voice, both in your dataset.

1. Run **cell 1** (the ▶ button on the cell, or *Shift+Enter*). Check that your files are listed.
   - The **first** run of a session takes a few minutes, because it installs things. That's normal.
2. Go to **cell 2** and set:
   ```python
   RUN = True
   SONGS = "relaxa"            # the song(s) to convert
   REFERENCE = "displicente"   # the voice clip
   SEMITONES = 0               # see "pitch" below
   ```
   If your song file is **already a clean vocal stem**, also set `SEPARATE_VOCALS = False`. If your voice clip is a clean stem, set `ISOLATE_REFERENCE_VOCAL = False`.
3. Run cell 2. You'll see setup lines, then `🎧 relaxa.mp3`, then a progress bar, then `✅ …/converted/relaxa__displicente__seedvc.wav`.
4. Run **cell 5** to listen: the original, the converted song, and the new vocal on its own.
5. Get the files out: see [section 9](#9-getting-your-results-out).

**Pitch:** if the new voice sounds strained, or like a chipmunk, the two singers sit in different ranges. A man's song sung by a woman's voice usually needs `SEMITONES = 12` (one octave up); the reverse needs `-12`. If you're not sure, set `AUTO_OCTAVE = True` and it decides for you ([Recipe D](#8-recipe-d-octave-check-optional) explains how).

---

## 6. Recipe B: train your own RVC voice

**You need:** about **10 minutes or more** of one person singing, split across any number of files, in your dataset.

**Before you start, the recordings should be:**
- **one singer only**, with no backing vocals or harmonies;
- **dry** (no reverb or echo) and without instruments. Full songs work if you leave `ISOLATE_VOCALS = True`, but clean stems are better;
- any length each. Silence doesn't matter, because training cuts the audio into short pieces.

### Steps
1. Attach your dataset (3.5) and run **cell 1**. **Write down the numbers of your recordings** (e.g. 2, 3, 6, 7).
2. In **cell 4**, set:
   ```python
   RUN = True
   DATASET = "2, 3, 6, 7"   # your recordings: numbers, or a name part they share ("maria take")
   MODEL_NAME = "maria"     # the voice's name. Required when you pick by number
   ISOLATE_VOCALS = False   # False if the recordings are clean stems; True if they're full songs
   EPOCHS = 200             # how long to train; 200 is a good start for 10+ minutes
   SAVE_EVERY = 10          # leave as is
   BATCH_SIZE = 8           # leave as is
   SAMPLE_RATE = 40000      # leave as is
   ```
3. Make sure every other cell still has `RUN = False`.
4. Start training **in the background**: click **Save Version** (top right) ▸ **Save & Run All (Commit)** ▸ **Save**.
   - This keeps running with the browser closed, for up to ~12 hours. A normal interactive session would stop when idle, so always train this way.
   - Expect very roughly **1–3 hours** for 200 epochs. It uses your weekly GPU quota (about 30 h).
5. **Watching progress (optional):** open the notebook's page ▸ *Versions* ▸ the running version ▸ *Logs*. You'll see lines like:
   ```
   maria | epoch=57 | step=1425 | ...
   ```
6. When it finishes, the version's **Output** tab contains:
   ```
   voices/maria.pth      ← your voice
   voices/maria.index    ← makes the voice more accurate
   voices/.training/…    ← checkpoints, for continuing training later
   ```
   Done. Go to [Recipe C](#7-recipe-c-convert-songs-with-your-rvc-voice) to use it.

### Training again or longer
**It stopped early** (Kaggle's time limit, a crash), **or the voice isn't convincing yet:**
1. Open the notebook ▸ **Add Input** ▸ **Notebook Output** ▸ pick the training version.
2. In cell 4, keep the **same `MODEL_NAME`** and set `EPOCHS` to the new **total** (e.g. `300`).
3. **Save & Run All** again. It prints `resuming from checkpoints …` and continues from where it stopped, not from zero.

**How many epochs?**

| What you hear | What to do |
|---|---|
| Sounds like a blend, not quite the singer | train more: continue to 300–400 |
| Metallic, robotic, buzzy artefacts | overtrained: next time, stop earlier (e.g. 150) |
| Good | stop: you're done |

---

## 7. Recipe C: convert songs with your RVC voice

1. **If the voice was trained in an earlier run**, attach it: **Add Input** ▸ **Notebook Output** ▸ the training version. (A `.pth` that's in your dataset needs no extra step.)
2. Run **cell 1**. Your voice appears under **RVC models**:
   ```
   RVC models:
      • maria
   ```
3. In **cell 3**, set:
   ```python
   RUN = True
   SONGS = "relaxa"
   MODEL = "maria"
   PITCH = 0
   ```
   Optional, for an automatic octave check: `AUTO_OCTAVE = True` and `PITCH_REFERENCE = "maria take 1"` (any clip of the trained voice).
   If the song is already a clean vocal stem: `SEPARATE_VOCALS = False`.
4. Run cell 3, then **cell 5** to listen.

---

## 8. Recipe D: octave check (optional)

Voice conversion keeps the melody **at the pitch it was sung**. If a deep male voice sang the song and the new voice is a high female one, she'd be singing an octave below her natural range, and it sounds wrong. The octave check measures both voices and tells you whether to shift by whole octaves.

**Just check** (nothing is converted) with **cell 7** (cell 6 on Colab):
```python
RUN = True
SOURCE = "relaxa"        # the vocal you'll convert
TARGET = "displicente"   # a clip of the target voice
ISOLATE = False          # True only if these are full songs, not vocal stems
```
It prints, for example:
```
   source vocal: median E3 (166 Hz), range C3–A3
   target voice: median E4 (331 Hz), range B3–G#4
   gap +12.0 semitones → +12 semitones (1 octave up)
```
Then put that number in `SEMITONES` (Seed-VC) or `PITCH` (RVC).

**Or let it apply the shift automatically:** set `AUTO_OCTAVE = True` in cell 2 or 3. It adds the shift to whatever `SEMITONES`/`PITCH` you set, separately for each song.

**How it decides:** if the two voices are less than half an octave apart (6 semitones), it doesn't shift. More than that, it rounds to the nearest whole octave. Close to exactly half an octave, it prints **⚠️ borderline**. Either choice can work there, so listen to both.

---

## 9. Getting your results out

Everything is written to **`/kaggle/working/`** (on Colab: inside your Drive folder):

| File | What it is |
|---|---|
| `converted/<song>__<voice>__<engine>.wav` | the finished song (24-bit WAV) |
| `converted/<song>__<voice>__<engine>_vocals.wav` | just the new vocal, the same length as the song, ready for a DAW |
| `converted/<…>.json` | the settings that were used |
| `voices/<name>.pth` + `.index` | voices you trained |
| `results.zip` | everything in `converted/`, zipped (made by cell 6) |

⚠️ **Kaggle deletes `/kaggle/working` when the session ends**, unless you either:
- **download** it: run **cell 6**, then in the right panel open **Output** and download `results.zip`; or
- **save a version**: *Save Version ▸ Save & Run All*. The files stay in that version's **Output** tab and can be attached to later sessions.

Already converted a song with the same voice? It's skipped (`already exists`). Set `OVERWRITE = True` to redo it.

---

## 10. Every setting explained

**Cell 1**
| Setting | Default | Meaning |
|---|---|---|
| `INPUT` | `""` | Kaggle only. Leave empty to find your dataset automatically. If you attached several datasets, type the folder name, e.g. `"my-songs"` |
| `DRIVE_FOLDER` | `"áudio"` | Colab only. Your folder inside *My Drive* |

**Cell 2 · Seed-VC**
| Setting | Default | Meaning / when to change |
|---|---|---|
| `RUN` | `False` | Kaggle only. Set `True` to run this cell |
| `SONGS` | `"1"` | songs to convert ([section 4](#4-how-to-pick-files-in-any-cell)) |
| `REFERENCE` | | the clip of the target voice |
| `SEMITONES` | `0` | pitch shift. `12` = one octave up, `-12` = one octave down |
| `DIFFUSION_STEPS` | `30` | quality vs speed. 30–50 for singing. Higher is slower |
| `AUTO_OCTAVE` | `False` | measure both voices and add an octave shift if needed ([Recipe D](#8-recipe-d-octave-check-optional)) |
| `SINGING_MODEL` | `True` | `False` for speech instead of singing |
| `SEPARATE_VOCALS` | `True` | `False` if the songs are already a cappella stems |
| `ISOLATE_REFERENCE_VOCAL` | `True` | `False` if the voice clip is already a clean stem |
| `VOCAL_GAIN_DB` | `0.0` | make the new vocal louder (+) or quieter (−) in the mix |
| `OVERWRITE` | `False` | redo songs that were already converted |

**Cell 3 · RVC convert**
| Setting | Default | Meaning / when to change |
|---|---|---|
| `MODEL` | | the voice's name (the `.pth` file name without `.pth`) |
| `PITCH` | `0` | like `SEMITONES` above |
| `INDEX_RATE` | `0.5` | 0–1. How strongly to pull towards the trained voice (needs the `.index`). Higher sounds more like the voice but can get less clear |
| `PROTECT` | `0.33` | 0–0.5. Protects breaths and consonants from artefacts. Lower means stronger protection |
| `AUTO_OCTAVE` + `PITCH_REFERENCE` | off | automatic octave check against a clip of the trained voice |
| `SONGS`, `SEPARATE_VOCALS`, `VOCAL_GAIN_DB`, `OVERWRITE` | | same as cell 2 |

**Cell 4 · RVC training**
| Setting | Default | Meaning / when to change |
|---|---|---|
| `DATASET` | | the recordings: numbers or a shared name part |
| `MODEL_NAME` | | the voice's name. Required when `DATASET` is numbers. Reuse it to continue training |
| `ISOLATE_VOCALS` | `True` | `False` for clean vocal stems |
| `EPOCHS` | `200` | total training length ([how many?](#training-again-or-longer)) |
| `SAVE_EVERY` | `10` | how often checkpoints are saved, in epochs. Leave as is |
| `BATCH_SIZE` | `8` | leave as is. Lower it (e.g. 4) only if you get an "out of memory" error |
| `SAMPLE_RATE` | `40000` | leave as is |

**Cell 5 · Listen:** `START_AT_SECONDS` and `CLIP_SECONDS` choose which excerpt plays. Short songs are handled automatically.

---

## 11. Using Google Colab instead

There's a Colab version: `Voice_Conversion.ipynb` (there's an "Open in Colab" link in the README).

- **It only works with paid compute units** (Colab Pro or pay-as-you-go). The free tier disconnects it.
- Your files live in **Google Drive**, in the folder `DRIVE_FOLDER` (default `áudio`). It can be flat like the Kaggle dataset, or use a `voices/` subfolder for clips, models and training folders.
- Settings are form fields (sliders and boxes), and there are no `RUN` switches. Each cell runs when you click it.
- Results are written straight into your Drive folder (`converted/`, `voices/`), so nothing gets lost when the session ends.
- Training checkpoints are copied to `voices/.training/` every few minutes. If Colab disconnects, run cell 4 again with the same settings and it continues.
- **After the code is updated:** *Runtime ▸ Disconnect and delete runtime* before running again, or Colab keeps using the old code.

---

## 12. Troubleshooting

**Where to look first:** the lines printed **above** an error are the real explanation. The red traceback at the bottom is just where it stopped.

| What you see | Why | Fix |
|---|---|---|
| Colab: *"Runtime disconnected… code not allowed in the no-cost tier"* | Colab's free tier blocks this workload | use Kaggle, or paid Colab units |
| `No folder with songs found under /kaggle/input` | the dataset isn't attached | **Add Input** ▸ your dataset |
| `Several folders could hold your songs` | several datasets are attached | set `INPUT = "my-songs"` in cell 1 |
| `… not found. Available: …` / `matches no clip … and no song` | a typo, or the file isn't in the dataset | use the exact name or number from cell 1's list |
| `… is ambiguous: a, b` | your text matches several files | type more of the name, or use the number |
| `Set MODEL_NAME` | you picked training files by number | give the voice a name in `MODEL_NAME` |
| `AUTO_OCTAVE with RVC needs PITCH_REFERENCE` | RVC models contain no audio to measure | set `PITCH_REFERENCE` to a clip of that voice |
| `⚠️ No GPU` / `⚠️ P100` | wrong accelerator | Settings ▸ **GPU T4 x2** |
| errors while cloning or downloading (git, pip, huggingface) | the internet is off | Settings ▸ **Internet On** (needs a verified phone) |
| my trained voice isn't listed in cell 1 | the training run's output isn't attached | **Add Input** ▸ **Notebook Output** ▸ that version |
| training stopped before finishing | Kaggle's ~12 h limit, or an error | attach that output and run cell 4 again with the same `MODEL_NAME` ([details](#training-again-or-longer)) |
| `CUDA out of memory` while training | the batch doesn't fit in GPU memory | `BATCH_SIZE = 4` |
| the voice strains, or sounds like a chipmunk or a giant | the singers sit in different ranges | `SEMITONES`/`PITCH` ±12, or `AUTO_OCTAVE = True` |
| the original singer's backing vocals are still audible | the vocal split only takes the lead vocal | expected. Use songs with a clear lead, or your own stems |
| metallic or robotic RVC voice | overtrained, or `INDEX_RATE` too high | fewer epochs next time, or `INDEX_RATE = 0.3` |
| `already exists (tick OVERWRITE to redo)` | that song and voice were done before | `OVERWRITE = True` |
| the first run takes ages, with many install lines | each engine sets up its own environment and downloads models on first use | normal: a few minutes per engine, once per session |
| `Warning: Skipped loading some keys due to shape mismatch` (Seed-VC) | comes from Seed-VC's own model files | harmless as far as we know. Judge by ear |
| an old bug is back, or new settings are missing | you're using an old copy of the notebook | Kaggle: re-import the notebook (3.3). Colab: *Disconnect and delete runtime* |

---

## 13. How it works under the hood

*For when something breaks and you, or Claude, need to dig in. Developer details are in `CLAUDE.md` at the repo root.*

- **Files:**
  - `vc.py`: everything the notebooks call (setup, file lookup, conversion, training).
  - `pitch.py`: the octave check.
  - It reuses `../neural-upscaler/upscaler.py` for the vocal split (BS-RoFormer), file selection, audio helpers, and running programs with visible output.
- **The notebooks are thin:** each session they download this repository (branch `ccr-8be874fb-rdbsnf`, falling back to the default branch if it's gone) and call `vc.py`.
- **Isolated environments:** Seed-VC and Applio (the RVC toolkit) each need old, conflicting library versions. Each one runs from a pinned copy of its source code, in its own Python environment, built automatically on first use:
  - Seed-VC: Python 3.10 + torch 2.4;
  - Applio: Python 3.12 + torch 2.11 for CUDA 12.8.
- **Scratch space:** temporary files go in `/tmp` (Kaggle) or `/content` (Colab) and vanish with the session.
- **Volume:** both engines normalise their output, so the new vocal is scaled back to the original vocal's level before mixing.
- **Training safety net:**
  - Checkpoints are mirrored to `voices/.training/<name>/` as training runs.
  - A new run looks for them in the output folder and in attached inputs, then resumes.
  - Only the newest voice file and the latest full checkpoint are kept. Each full checkpoint set is about 1.3 GB.
