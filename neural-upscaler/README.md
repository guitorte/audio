# 🎚️ Neural Audio Upscaler

> 📖 **New here, or back after a long break? Read [GUIDE.md](GUIDE.md).** It's a step-by-step guide.
> 🇧🇷 **Guia em português: [GUIDE.pt-BR.md](GUIDE.pt-BR.md)**

Restores songs **without regenerating them**. It's the same recording, with lossy-compression damage repaired and the missing high end rebuilt.

[Open the notebook in Colab](https://colab.research.google.com/github/guitorte/audio/blob/ccr-8be874fb-rdbsnf/neural-upscaler/Neural_Audio_Upscaler.ipynb). It reads songs from a Google Drive folder (default `áudio`) and writes 24-bit WAVs to `áudio/upscaled/`.

```
song ─► ① BS-RoFormer ─┬─ vocals ──────────► ② Apollo ─┐
                       └─ mix − vocals ────► ② Apollo ─┴─► ③ AudioSR ─► ④ Matchering
```

| Stage | Tool | Notes |
|---|---|---|
| ① split | [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) · `model_bs_roformer_ep_317_sdr_12.9755` | The instrumental is the exact residual, so vocals + instrumental equal the original mix |
| ② restore | [Apollo](https://github.com/JusperLee/Apollo) (pinned commit) | Repairs lossy-codec artefacts at 44.1 kHz, chunked with padded seams |
| ③ bandwidth | [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) 0.0.7 | Runs in its own Python 3.10 venv (old numpy/transformers pins). It works in Mid/Side, 5.12 s chunks, and is level-matched. Only the band **above** the detected shelf comes from AudioSR, and full-band files skip it |
| ④ master | [Matchering](https://github.com/sergree/matchering) | Optional, against a reference song in the same folder |

Files: `upscaler.py` (pipeline, main env) · `audiosr_worker.py` (AudioSR venv) · `upscaler_dsp.py` (shared helpers: shelf detection, crossover, crossfades).

Model weights keep their own licences. Apollo and AudioSR are research releases; check their terms before any commercial use.
