# CLAUDE.md

Personal music/audio toolbox. Most of it is **Google Colab notebooks backed by small Python packages**, plus a few standalone browser tools. There is no build system, CI, or test suite, and nothing here is a deployable app.

## Layout

| Path | What it is |
|---|---|
| `music-to-midi/` | Main song → stems (Demucs) → multi-track MIDI workflow. `modules/` package, `run.py` CLI, `notebooks/Song_to_Stems_to_MIDI.ipynb` |
| `stem-to-midi/` | Earlier single-stem → MIDI project (Basic Pitch, ADTOF drums), now folded into `music-to-midi/` |
| `audio-restoration-pipeline/` | DSP restoration and mastering (spectral gating, EQ, harmonic "frequency restoration", LUFS) with `modules/` and several notebook versions |
| `neural-upscaler/` | Neural restoration of songs in a Google Drive folder. See its section below |
| `voice-conversion/` | Swap a song's singer with Seed-VC (zero-shot) or RVC via Applio (convert and train). See its section below |
| `vocal-lab/` | Workbench for one vocal: split, de-reverb, denoise, upscale, vocal-to-MIDI, analysis, phrase chopper. See its section below |
| Top-level `*.ipynb` | Older standalone Colab notebooks (stem separator, WAV splitter, song→MIDI, LLM lyrics) |
| Top-level `*.html` | Self-contained browser MIDI tools (riff/melody generators). Single file each, Tone.js from cdnjs, no build |
| `upload/`, `images/`, `*.mp3` | User assets. Don't modify, move, or delete them |

## Conventions

- **Colab is the target runtime.** Assume a recent Python (3.13 as of Oct 2026; don't hard-code a version), a T4 GPU, and Drive mounted at `/content/drive/MyDrive`.
- **Notebooks shallow-clone this repo** to `/content/audio`, `sys.path.insert` the project folder, and import its package. A notebook's `BRANCH` must exist, and the notebook falls back to the default branch if the clone fails. The clone is skipped when `/content/audio` already exists, so **after pushing a fix, the user must run *Runtime ▸ Disconnect and delete runtime*** before re-running, or they'll keep the old code. Say so whenever you hand back a fix.
- **Logic lives in `.py` modules, and notebooks stay thin.** They hold form cells (`#@title`, `#@param`, `"cellView": "form"`) that call into the package. Put new logic in the module, not inline in the notebook.
- **Generate or edit `.ipynb` files programmatically** (load JSON, edit cells, dump with `ensure_ascii=False`). Don't hand-edit the JSON. Keep outputs empty and `metadata.accelerator = "GPU"`. Include the "Open in Colab" badge pointing at the right branch.
- **Language:** older READMEs and notebooks are in Portuguese (pt-BR) and `neural-upscaler/` is in English. Match the language of the folder you're editing.
- **Drive paths contain accents** (`áudio`). Compare names with `unicodedata.normalize("NFC", ...)` plus `casefold()`, because Drive may return NFD (see `neural-upscaler/upscaler.py::find_folder`).
- **Never overwrite a user's source audio.** Write results to a sibling output folder and skip existing outputs unless an overwrite flag is set.

## Running other programs from a notebook

Two Colab traps have each broken a real run:

1. **Child output is invisible.** Colab shows only what the kernel prints, so a plain `subprocess.run(check=True)` fails with a bare "exit status 1". Use `neural-upscaler/upscaler.py::run`, which streams output into the cell and raises with the last 25 lines.
2. **Kernel environment variables leak into children.** Colab exports `MPLBACKEND=module://matplotlib_inline.backend_inline` and a `PYTHONPATH`. Both break a program running in a *different* Python, such as a venv: matplotlib rejects the unknown backend, and `PYTHONPATH` mixes in the kernel's packages. Launch such programs with `upscaler.py::venv_env()`, which drops `PYTHONPATH`/`PYTHONHOME` and sets `MPLBACKEND=Agg`.

When a user reports a Colab failure, the streamed output above the traceback is where the real error is.

## neural-upscaler

Pipeline: **BS-RoFormer** vocal split → **Apollo** (lossy-codec repair) → **AudioSR** (bandwidth extension, 48 kHz) → **Matchering** (optional reference master). Every stage can be switched off. It reads a Drive folder (default `áudio`) and writes 24-bit WAVs plus a settings `.json` to `upscaled/` inside it. Scratch files go in `/content/upscaler_work`.

| File | Runs in | Role |
|---|---|---|
| `upscaler.py` | Colab kernel | File selection, setup/installs, stages, `process()` |
| `audiosr_worker.py` | AudioSR venv | Chunked Mid/Side AudioSR with the crossover |
| `upscaler_dsp.py` | both | numpy/scipy only: shelf detection, crossover, chunking, crossfades |
| `Neural_Audio_Upscaler.ipynb` | Colab | 3 form cells: connect Drive, choose and run, A/B compare |
| `flashsr_trial.py` + `FlashSR_Trial.ipynb` | Colab kernel | **Experimental**, not called by `process()`. Restores one excerpt with FlashSR (pinned `FLASHSR_COMMIT`, weights from the HF *dataset* repo `jakeoneijk/FlashSR_weights`, about 3.3 GB) using the same Mid/Side splice as the worker, and optionally times AudioSR on the same excerpt. FlashSR needs no venv: it imports on Python 3.13 / torch 2.15 with only `einops` added. It accepts exactly 245,760 samples (5.12 s at 48 kHz) and at most 2 rows per call. Its random-weight checkpoints (`AudioSRUnet` in `DDPM`, `AutoencoderKL`, `SRVocoder`) run the whole path on CPU, about 6 GB RAM for one row. First real run (T4, Nov 2026): FlashSR loaded and ran Mid+Side together, about 0.1 s per chunk for the sampling step. AudioSR started *after* FlashSR in the same session was killed while loading (exit -9, system RAM), so the comparison now runs AudioSR first and treats its failure as non-fatal. FlashSR is trained on inputs cut at 16 kHz or lower; a song that reaches higher is a poor test |

Invariants to keep:
- **Stems sum to the mix.** audio-separator peak-normalises, so `separate_vocals()` scales its input to 0.5 peak and undoes it afterwards. The instrumental is the residual `mix - vocals`.
- **AudioSR only adds highs.** `detect_cutoff()` finds the lossy shelf. Below it the output is the original audio, via a power-complementary zero-phase Butterworth `crossover()`. Files with no shelf skip AudioSR unless the user sets `CUTOFF_HZ`. Each chunk is level-matched to the band just under the cutoff, because AudioSR peak-normalises its output.
- `audiosr_worker.py` may import only `upscaler_dsp.py`, never `upscaler.py`, because the venv lacks the kernel's packages.

## voice-conversion

**`GUIDE.md` and `GUIDE.pt-BR.md` are the user-facing manuals** (the same goes for `neural-upscaler/GUIDE*.md`). Keep both languages in sync whenever cells, settings, defaults or behaviour change, because the user relies on them after months away.

Song → BS-RoFormer split → convert **only the vocal** with Seed-VC or RVC → level-match it to the original vocal → lay it back over the untouched instrumental. Outputs go to `converted/<song>__<voice>__<engine>.wav` plus `_vocals.wav`. Voices live in `<songs folder>/voices/`: audio clips are Seed-VC references, `<name>.pth` (+ `.index`) are RVC models, and subfolders are RVC training sets.

- **Input and output folders are separate.** `scan_voices` / `convert` / `train` take `folder` (songs + `voices/`, may be read-only) and `out` (writable: `converted/`, trained models, `voices/.training/`). `out` defaults to `folder`, which is the Colab/Drive case. Never write into `folder` itself.
- **Flat layout is supported** (the user's Kaggle datasets keep everything at the root): `.pth`/`.index` next to the songs count as models, any listed file can be the Seed-VC reference (`resolve_reference`), and `training_set()` takes a `voices/` subfolder or files picked like `SONGS` (by name or number; numbers need `MODEL_NAME`). The training cache drops recordings removed from the selection.
- **Kaggle** (`Voice_Conversion_Kaggle.ipynb`): the input is a read-only Dataset found by `kaggle_input()`, which skips attached earlier outputs (they contain `converted/`) and anything inside `voices/`. The output is `/kaggle/working`. The repo is cloned to `/tmp/audio` so it isn't saved as output. Every task cell has a `RUN = False` switch, because *Save & Run All* executes all cells. Training resumes from an earlier run's output attached as an input (`_resume_sources` searches `/kaggle/input/**/voices/.training/<model>`). Models trained in an earlier run are found the same way: `scan_voices` also reads `/kaggle/input/**/voices/` (skipping `.training`). The P100's GPU generation isn't supported by Applio's cu128 torch, so use T4.
- **Optional octave check** (`pitch.py`, off by default): librosa pYIN on both vocals (16 kHz, 65–1050 Hz, confidently voiced frames only, at most 2 min sampled), median gap in semitones rounded half-away-from-zero to whole octaves, gaps within 1 st of ±6 flagged as borderline. The shift is added to the user's `semitones`. Seed-VC measures against its reference clip; RVC needs `pitch_reference`, a clip of the model's voice. `pitch_check()` reports without converting. It's meant for clean, dry vocal stems.
- **Name matching** (`U.match_names`, used by `select_files` and `pick`): exact name or stem first, then prefix, then substring, so "male" doesn't also pick "female voice.wav".
- `vc.py` imports `../neural-upscaler/upscaler.py` as `U` for Drive lookup, file selection, decoding, `separate_vocals`, `run`, `venv_env` and `excerpt`. Changes there affect both projects.
- Both engines run from pinned clones with `cwd` set to the clone. Pass them **absolute paths**.
- **Applio exits 0 even when a step failed**, printing "success" after an error. Never trust its exit code: `_expect` / `_expect_files` check that each step wrote its output.
- Training mirrors finished checkpoints to `voices/.training/<name>/` (a file is copied once it's been unchanged for 30 s, and only the newest voice `.pth` is kept). If the local run is gone, `train()` restores the mirror and Applio resumes from `G_/D_`. Always pass `--save-only-latest`: each full checkpoint set is about 1.3 GB.

## vocal-lab

`vocal_lab.py` + `Vocal_Stem_Lab.ipynb` (English). One **active file** (`<work>/active.wav`) flows through the modules; each writes to `<project>/<stage>/`, records its main result in `lab.json`, and by default makes it the new active file. `route()` goes back to a stage. Outputs mirror to `MyDrive/audio/outputs/<project>/`, and `start()` restores `lab.json` from Drive in a new session. Scratch is `/content/vocal_lab`. It imports `../neural-upscaler/upscaler.py` as `U`, like `vc.py`.

- **audio-separator runs as a subprocess** (`run_separator`, through `U.run`), not through its Python API, so its memory is gone before AudioSR loads. One function, `separate()`, covers vocal split, lead/backing, de-reverb and denoise: `MODELS` maps a menu label to `(task, model file)` and `TASKS` maps each model's stem names to ours. Output files are named `<input>_(<Stem>)_<model>.wav`; `_stem()` parses that. Stem names differ per model (`noreverb`, `No Reverb`, `dry`, `other`), and some registry entries list none, so only add a model to `MODELS` after checking its stems. An unknown stem is saved and reported, never guessed.
- **Run the upscaler after the 44.1 kHz cleaning models**, or they drop the rebuilt highs. `separate()` warns when the active file is above 44.1 kHz.
- **Upscale** calls `U.setup_audiosr()` / `U.audiosr()` with one job, so it shares the venv and the Mid/Side splice. A mono input comes back mono. FlashSR goes through `flashsr_trial`.
- **GAME** (`GAME_COMMIT`, weights from the v1.0.0 release zips: small/medium/large) pins `numpy<2`, so it has a `uv` Python 3.12 venv with **CPU torch** (a 30 s clip takes about 20 s on 2 cores). Run `infer.py extract` with `cwd` set to the clone. It writes `<name>.mid` and `<name>.csv` (`onset,offset,pitch` with names like `A#3+7`). Languages: `en`, `ja`, `yue`, `zh`; anything else runs language-neutral. Weights are CC BY-NC-SA 4.0.
- **Beat This** (`pip install beat-this`) runs in the kernel; its checkpoint (77 MB) downloads from JKU's cloud. `analyze()` falls back to librosa's tempo if it fails, and stores the BPM so `transcribe()` saves its MIDI at that tempo.
- **Sandbox checks that work here:** GitHub release assets and JKU's cloud are reachable, so GAME, Beat This and audio-separator's models run for real on CPU (keep clips under about 10 s for RoFormer models). `run_separator`, `run_game` and `find_beats` are the seams to fake.

## Dependency pitfalls (already solved; don't regress)

- **basic-pitch** on Python ≥ 3.12: install with `--no-deps` plus `onnxruntime` (ONNX backend). Plain `pip install basic-pitch` drags in TensorFlow/tflite, which have no wheels for these versions.
- **ADTOF-pytorch** is GitHub-only: `pip install git+https://github.com/xavriley/ADTOF-pytorch.git`.
- **AudioSR 0.0.7** pins numpy 1.23.5, librosa 0.9.2, and transformers 4.30.2, which can't share Colab's environment. It runs in an isolated `uv` Python 3.10 venv. `upscaler.py::AUDIOSR_PINS` adds three pins of its own:
  - `torch==2.5.1`: before `torch.load` defaulted to `weights_only=True`.
  - `setuptools<70`: librosa 0.9.2 imports `pkg_resources`.
  - `matplotlib`: `audiosr.utilities` imports it without declaring it.

  `setup_audiosr()` re-runs the pinned install every time, so venvs built from an older pin list get repaired. The weights (several GB) download from Hugging Face on first use.
- **Seed-VC** (`SEEDVC_COMMIT`) needs Python 3.10, torch 2.4 and numpy 1.26. `vc.py::SEEDVC_PINS` is the inference-only subset of its `requirements.txt`; the full list drags in GUI and eval packages. Weights download from Hugging Face on first use.
- **Applio** (`APPLIO_COMMIT`) needs Python 3.12 and torch 2.11. Install from `--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match`, like its own installer: PyPI's torch 2.11 is a CUDA 13 build that may need a newer driver than Colab has. Copy `assets/config_template.json` to `assets/config.json`, or training silently skips writing the voice `.pth`. Pass `--gpu -` (CPU) when there's no GPU, or extraction silently writes nothing.
- **Apollo** is cloned at a pinned commit (`APOLLO_COMMIT`), and its `inference.py` must run with `cwd` set to the clone. It requires 44.1 kHz input.

## Verifying changes

There are no tests and usually no GPU or Hugging Face access in the dev sandbox, so model downloads fail locally. To check changes:
- Unit-check pure helpers directly (e.g. `upscaler_dsp.py`: shelf detection, crossover reconstruction, crossfade weights summing to 1).
- Run pipelines end to end with the model stages monkeypatched (fake separator/Apollo), using `ffmpeg` and the repo's sample MP3.
- Smoke-test notebooks by `exec`-ing their code cells with a stubbed `google.colab`.
- **AudioSR's real code runs without Hugging Face.** Stub `transformers.RobertaTokenizer.from_pretrained` (return `None`) and `RobertaConfig.from_pretrained` (return `cls()`), then replace `audiosr.build_model` with one that builds `audiosr.pipeline.LatentDiffusion(**default_audioldm_config("basic")["model"]["params"])` on CPU with random weights. Run the worker with `--steps 2`. To inject the stub into a worker launched through `venv_env()`, which strips `PYTHONPATH`, use a `.pth` file in the scratch venv's site-packages, gated by an env var.
- **Seed-VC and Applio also run offline with random weights.** For Seed-VC, a `.pth` hook gated by an env var stubs `hf_utils.load_custom_model_from_hf`, `modules.commons.load_checkpoint`, BigVGAN, and Whisper (random `WhisperConfig` at whisper-small size), and uses the repo's bundled `campplus_cn_common.bin`. Pass `--fp16 False` on CPU. For Applio, save random `rmvpe.pt` (`E2E(4, 1, (2, 2))`) and contentvec (`HubertModelWithFinalProj(HubertConfig(classifier_proj_size=256))`) into `rvc/models/`. Training with `--no-pretrained` then produces `G_/D_` files that can stand in as `rvc/models/pretraineds/hifi-gan/f0G40k.pth` / `f0D40k.pth`. This runs the whole preprocess → extract → train → index → infer flow on CPU.
- **Test launchers in Colab's environment:** set `MPLBACKEND=module://matplotlib_inline.backend_inline` and `PYTHONPATH=/env/python` in the parent process.
- Say plainly which parts were only mock-tested. The first real model run happens in Colab.
- `music-to-midi` can also run locally: `cd music-to-midi && python run.py song.mp3 -o output/`.
