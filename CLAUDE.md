# CLAUDE.md

Personal music/audio toolbox. Most of it is **Google Colab notebooks backed by small Python packages**, plus a few standalone browser tools. There is no build system, CI, or test suite, and nothing here is a deployable app.

## Layout

| Path | What it is |
|---|---|
| `music-to-midi/` | Main song → stems (Demucs) → multi-track MIDI workflow. `modules/` package, `run.py` CLI, `notebooks/Song_to_Stems_to_MIDI.ipynb` |
| `stem-to-midi/` | Earlier single-stem → MIDI project (Basic Pitch, ADTOF drums), now folded into `music-to-midi/` |
| `audio-restoration-pipeline/` | DSP restoration and mastering (spectral gating, EQ, harmonic "frequency restoration", LUFS) with `modules/` and several notebook versions |
| `neural-upscaler/` | Neural restoration: BS-RoFormer → Apollo → AudioSR → Matchering. It reads a Google Drive folder (default `áudio`) and writes to `upscaled/` inside it |
| Top-level `*.ipynb` | Older standalone Colab notebooks (stem separator, WAV splitter, song→MIDI, LLM lyrics) |
| Top-level `*.html` | Self-contained browser MIDI tools (riff/melody generators). Single file each, Tone.js from cdnjs, no build |
| `upload/`, `images/`, `*.mp3` | User assets. Don't modify, move, or delete them |

## Conventions

- **Colab is the target runtime.** Assume Python 3.12, a T4 GPU, and Drive mounted at `/content/drive/MyDrive`. Notebooks shallow-clone this repo to `/content/audio` and `sys.path.insert` the project folder, then import its `modules`/package. A notebook's clone `BRANCH` must point at a branch that exists. Fall back to the default branch if the clone fails.
- **Logic lives in `.py` modules, and notebooks stay thin.** They hold form cells (`#@title`, `#@param`, `"cellView": "form"`) that call into the package. Put new logic in the module, not inline in the notebook.
- **Generate or edit `.ipynb` files programmatically** (load JSON, edit cells, dump with `ensure_ascii=False`). Don't hand-edit the JSON. Keep outputs empty and `metadata.accelerator = "GPU"`. Include the "Open in Colab" badge pointing at the right branch.
- **Language:** older READMEs and notebooks are in Portuguese (pt-BR) and `neural-upscaler/` is in English. Match the language of the folder you're editing.
- **Drive paths contain accents** (`áudio`). Compare names with `unicodedata.normalize("NFC", ...)` plus `casefold()`, because Drive may return NFD (see `neural-upscaler/upscaler.py::find_folder`).
- Never overwrite a user's source audio. Write results to a sibling output folder and skip existing outputs unless an overwrite flag is set.

## Dependency pitfalls (already solved; don't regress)

- **basic-pitch** on Python 3.12: install with `--no-deps` plus `onnxruntime` (ONNX backend). Plain `pip install basic-pitch` drags in TensorFlow/tflite, which have no 3.12 wheels.
- **ADTOF-pytorch** is GitHub-only: `pip install git+https://github.com/xavriley/ADTOF-pytorch.git`.
- **AudioSR 0.0.7** pins numpy 1.23.5, librosa 0.9.2, and transformers 4.30.2, which can't share Colab's environment. It runs in an isolated `uv` Python 3.10 venv (`neural-upscaler/upscaler.py::AUDIOSR_PINS`) with `torch==2.5.1` (before `torch.load` started defaulting to `weights_only=True`) and `setuptools<70` (librosa 0.9.2 imports `pkg_resources`). `audiosr_worker.py` runs inside that venv. It may import only `upscaler_dsp.py` (numpy/scipy), never `upscaler.py`.
- **Apollo** is cloned at a pinned commit (`APOLLO_COMMIT`), and its `inference.py` must run with `cwd` set to the clone. It requires 44.1 kHz input.
- **audio-separator** peak-normalises stems. `separate_vocals()` scales its input to 0.5 peak and undoes that afterwards, and the instrumental is the residual `mix - vocals`. Keep both, so stems always sum to the original mix.

## Verifying changes

There are no tests and usually no GPU or Hugging Face access in the dev sandbox, so model downloads fail locally. To check changes:
- Unit-check pure helpers directly (e.g. `upscaler_dsp.py`: shelf detection, crossover reconstruction, crossfade weights summing to 1).
- Run pipelines end to end with the model stages monkeypatched (fake separator/Apollo, or a stub `audiosr` package on `PYTHONPATH`), using `ffmpeg` and the repo's sample MP3.
- Smoke-test notebooks by `exec`-ing their code cells with a stubbed `google.colab`.
- Say plainly which parts were only mock-tested. The first real model run happens in Colab.
- `music-to-midi` can also run locally: `cd music-to-midi && python run.py song.mp3 -o output/`.
