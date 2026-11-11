# Vocal Stem Lab

A Colab workbench for one vocal: pull it out of a song, split lead from backing, remove room reverb and noise, rebuild missing highs, turn the melody into MIDI, and cut it into phrases.

[Open the notebook in Colab](https://colab.research.google.com/github/guitorte/audio/blob/ccr-8be874fb-rdbsnf/vocal-lab/Vocal_Stem_Lab.ipynb)

It is a rebuild of the earlier "Vocal Stem Lab v2.0" notebook. The idea is the same (one active file, a router between stages, everything mirrored to `MyDrive/audio/outputs/<project>/`). The logic now lives in `vocal_lab.py`, and several engines were replaced.

## Modules and engines

| Module | Engine | Runs in | Note |
|---|---|---|---|
| 3 · Analyse | [Beat This](https://github.com/CPJKU/beat_this) for tempo, librosa for key, ffmpeg for loudness | Colab's own Python | falls back to librosa's tempo if Beat This fails |
| A · Separate | [audio-separator](https://github.com/nomadkaraoke/python-audio-separator): BS/Mel-RoFormer, Demucs | its own process | lead vs backing now defaults to a Mel-RoFormer karaoke model |
| B · De-reverb | audio-separator: Mel-RoFormer by anvuew | its own process | the older UVR model is still in the menu |
| C · Denoise | audio-separator: Mel-RoFormer by aufr33 | its own process | replaces the `noisereduce` spectral gate |
| D · Upscale | AudioSR through `neural-upscaler`'s worker, or FlashSR (experimental) | AudioSR's Python 3.10 venv / Colab's Python | stereo kept, only the band above the cut is new |
| E · Vocal to MIDI | [GAME](https://github.com/openvpi/GAME) | a Python 3.12 venv, CPU | one sung line at a time; replaces Basic Pitch |
| F · Chopper | librosa | Colab's own Python | unchanged |

## What changed from the earlier notebook, and why

- **The upscaler no longer makes the song mono or replaces the whole signal.** It reuses `neural-upscaler`: Mid/Side, level-matched, and spliced in only above the cutoff.
- **AudioSR and the MIDI engine no longer share one environment.** They used to install conflicting pins into `/content/srenv`.
- **Separation, de-reverb and denoise are one function** (`separate`) with a model menu, since all three call the same tool.
- **Each module hands its result to the next.** The router is only needed to go back.
- **Files are named after the song and its history**, e.g. `Song_vocals_lead_dry_clean_48k.wav`.
- **A project can be reopened in a new session:** its state is kept on Drive.

## Order matters

Run the upscaler **after** de-reverb and denoise. Those models work at 44.1 kHz and would drop the rebuilt highs.

## What was verified

In a CPU sandbox on Python 3.13, the version Colab runs:

- Real runs: audio-separator 0.48 with the anvuew de-reverb model, GAME (small and medium) through its venv, and Beat This.
- Faked models: AudioSR and FlashSR in the upscale path, and the other separation models. Their file names were checked against audio-separator's registry, not by running each one.
- Not verified: anything on a GPU, and the sound quality of any engine.

## Licences

The code here follows the repository. Model weights keep their own terms, and several are for non-commercial use: GAME's weights are CC BY-NC-SA 4.0 according to its maintainers, and FlashSR's repository states no licence. Check each model's page before publishing anything made with it.
