# 🎤 voice-conversion

Swap the singer in a song with another voice and keep the instrumental, using [Seed-VC](https://github.com/Plachtaa/seed-vc) (zero-shot) or RVC through [Applio](https://github.com/IAHispano/Applio) (convert and train).

[Open the notebook in Colab](https://colab.research.google.com/github/guitorte/audio/blob/ccr-8be874fb-rdbsnf/voice-conversion/Voice_Conversion.ipynb)

```
song ─► BS-RoFormer ─┬─ vocal ──► Seed-VC | RVC ──► level-matched vocal ─┐
                     └─ instrumental ────────────────────────────────────┴─► converted/<song>__<voice>__<engine>.wav
```

Drive layout (inside the songs folder, default `áudio`):

| Path | Used by |
|---|---|
| `voices/<clip>.wav` (any audio) | Seed-VC reference. The vocal is isolated and the loudest 25 s are kept |
| `voices/<name>.pth` (+ `<name>.index`) | RVC conversion |
| `voices/<name>/` (folder of recordings) | RVC training. Writes `voices/<name>.pth`/`.index` and mirrors checkpoints to `voices/.training/<name>/` so training resumes after a disconnect |

| File | Role |
|---|---|
| `vc.py` | Setup (pinned clones, isolated `uv` venvs), vocal split/rebuild, the two engines, training with Drive mirroring |
| `Voice_Conversion.ipynb` | Form cells: connect Drive, Seed-VC, RVC convert, RVC train, listen |

Shared Drive, audio, separation and subprocess helpers come from `../neural-upscaler/upscaler.py`.

Only convert voices you have the right to use. Model weights keep their own licences: Seed-VC is GPL-3.0 and Applio is MIT, each with its own terms of use.
