# 🎤 voice-conversion

Swap the singer in a song with another voice and keep the instrumental, using [Seed-VC](https://github.com/Plachtaa/seed-vc) (zero-shot) or RVC through [Applio](https://github.com/IAHispano/Applio) (convert and train).

[Open the notebook in Colab](https://colab.research.google.com/github/guitorte/audio/blob/ccr-8be874fb-rdbsnf/voice-conversion/Voice_Conversion.ipynb) · **Kaggle:** import `Voice_Conversion_Kaggle.ipynb` (*File ▸ Import Notebook*, or paste its GitHub URL), attach your songs as a Dataset, set *GPU T4 x2* and *Internet On*. Setup steps are in its first cell.

```
song ─► BS-RoFormer ─┬─ vocal ──► Seed-VC | RVC ──► level-matched vocal ─┐
                     └─ instrumental ────────────────────────────────────┴─► converted/<song>__<voice>__<engine>.wav
```

**Flat folders work too**, which is the simplest option on Kaggle: put songs, voice clips, `.pth`/`.index` models and training recordings all at the dataset root. Every audio file is listed with a number, and each cell takes those numbers or parts of names (training: `DATASET = "maria take"` or `"4-9"` with a `MODEL_NAME`).

Or organise it with a `voices/` subfolder (the Drive default, inside the songs folder `áudio`):

| Path | Used by |
|---|---|
| `voices/<clip>.wav` (any audio) | Seed-VC reference. The vocal is isolated and the loudest 25 s are kept |
| `voices/<name>.pth` (+ `<name>.index`) | RVC conversion |
| `voices/<name>/` (folder of recordings) | RVC training. Writes `voices/<name>.pth`/`.index` and mirrors checkpoints to `voices/.training/<name>/` so training resumes after a disconnect |

| File | Role |
|---|---|
| `vc.py` | Setup (pinned clones, isolated `uv` venvs), vocal split/rebuild, the two engines, training with Drive mirroring |
| `Voice_Conversion.ipynb` | Colab form cells: connect Drive, Seed-VC, RVC convert, RVC train, listen |
| `Voice_Conversion_Kaggle.ipynb` | Kaggle version: reads a read-only Dataset, writes to `/kaggle/working`, `RUN` switches per task, zip cell |

Shared Drive, audio, separation and subprocess helpers come from `../neural-upscaler/upscaler.py`.

Colab's free tier has disconnected this notebook mid-run as a restricted workload, so use paid compute units or another GPU machine.

Only convert voices you have the right to use. Model weights keep their own licences: Seed-VC is GPL-3.0 and Applio is MIT, each with its own terms of use.
