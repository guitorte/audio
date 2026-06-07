# 🚀 Quick Start — Batch Stem → MIDI (5 min setup)

This is a **step-by-step beginner's guide** to convert isolated audio stems (drums, vocals, bass, guitar, piano, etc.) into a multi-track MIDI file that you can edit in any DAW.

---

## What does this do? 🎯

**Input:** A folder with separate audio files:
- `vocals.wav` → MIDI track (vocals)
- `drums.wav` → MIDI track (drums, with kick/snare/hihat notes)
- `bass.wav` → MIDI track (bass)
- `guitar.wav` → MIDI track (guitar)
- `piano.wav` → MIDI track (piano)
- ... etc

**Output:** A single file `song.mid` that plays all tracks together, ready to edit in Ableton, Logic, Cubase, or any DAW.

---

## Where do stems come from? 📁

**If you don't have stems yet:** Use [Demucs](https://github.com/facebookresearch/demucs) to split a full song into stems:

```bash
# Demucs splits any MP3/WAV into 6 stems
python -m demucs.separate -n htdemucs_6s mysong.mp3
# Output: separated/htdemucs_6s/mysong/
#         ├── vocals.wav
#         ├── drums.wav
#         ├── bass.wav
#         ├── guitar.wav
#         ├── piano.wav
#         └── other.wav
```

Then jump to **[Step 3: Run the conversion](#step-3-run-the-conversion)** below.

---

## Step 1: Open the Colab notebook

Click the badge at the top of `stem-to-midi/notebooks/Stem_to_MIDI_Batch.ipynb`:

<a href="https://colab.research.google.com/github/guitorte/audio/blob/claude/stem-midi-converter-OwEoi/stem-to-midi/notebooks/Stem_to_MIDI_Batch.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
</a>

Or paste this URL in your browser:
```
https://colab.research.google.com/github/guitorte/audio/blob/claude/stem-midi-converter-OwEoi/stem-to-midi/notebooks/Stem_to_MIDI_Batch.ipynb
```

**You're now in Google Colab** — a free Jupyter environment that runs in your browser. Everything happens here; you don't install anything locally.

---

## Step 2: Install dependencies (2 cells)

Look for **Cell 1** ("1. Instalar dependências"). Click the ▶️ play button.

This installs:
- `basic-pitch` — converts melodic instruments (vocals, bass, guitar, piano) to MIDI
- `adtof-pytorch` — converts drums to MIDI with kick/snare/hihat notes
- Audio libraries (`librosa`, `soundfile`, `pretty_midi`, etc.)

**⚠️ After it finishes, click `Runtime > Restart Session`** (top menu). This resets Python so new packages load correctly.

Then run **Cell 2** ("2. Imports e clone do repositório") — it downloads this repository and loads the conversion code.

---

## Step 3: Upload your stems

### Option A: From Google Drive (recommended for Colab)

1. Open Google Drive: https://drive.google.com
2. Create a folder: `stem-to-midi/input/mysong/`
3. Drag your audio files into it:
   - `vocals.wav` (or `.mp3`, `.flac`, `.m4a`)
   - `drums.wav`
   - `bass.wav`
   - `guitar.wav`
   - `piano.wav`
   - (any you have — missing ones are skipped)

Your folder now looks like:
```
My Drive/
└── stem-to-midi/
    └── input/
        └── mysong/
            ├── vocals.wav
            ├── drums.wav
            ├── bass.wav
            ├── guitar.wav
            └── piano.wav
```

### Option B: Local upload (if you're not using Drive)

In **Cell 4**, change:
```python
BASE_DIR = '/content/drive/MyDrive/stem-to-midi'  # ← change this
```

to a local path, then run **Cell 3** to mount Drive, or skip mounting entirely and use `/tmp` or another local folder.

---

## Step 4: Configure paths and filenames

In **Cell 4** ("4. Configurar paths"), you'll see:

```python
TRACK_NAME = 'minha-faixa'  # ← change to your song name
```

Change `minha-faixa` to something like `mysong` or `track1`. This is just a label — stems are auto-discovered by name (`vocals.wav`, `drums.wav`, etc.).

If your files have weird names like `lead_vox.wav` instead of `vocals.wav`, add a rename mapping:

```python
STEM_RENAMES = {'lead_vox': 'vocals', 'kit': 'drums'}
```

Run the cell. You should see:
```
Stems dir   : /content/drive/MyDrive/stem-to-midi/input/mysong
Output dir  : /content/drive/MyDrive/stem-to-midi/output/mysong
Merged MIDI : /content/drive/MyDrive/stem-to-midi/output/mysong/mysong.mid

Arquivos de audio em STEMS_DIR (5):
  bass.wav              5.23 MB
  drums.wav             4.15 MB
  guitar.wav            3.87 MB
  vocals.wav            2.34 MB
  piano.wav             1.56 MB
```

If you see `(vazio — coloque os stems...)`, your stems folder is empty — go back to **Step 3** and upload files.

---

## Step 5: Run the batch conversion

In **Cell 5** ("5. Batch → multi-track MIDI"), click ▶️.

This takes 2–10 minutes depending on:
- Number of stems (6 stems = longer)
- Length of audio (longer songs = longer)
- Which engines are used (drums = fastest, vocals = slower)

You'll see output like:
```
Found 5 stem(s):
  bass     <- bass.wav
  drums    <- drums.wav
  guitar   <- guitar.wav
  piano    <- piano.wav
  vocals   <- vocals.wav

[bass] -> /content/drive/MyDrive/stem-to-midi/output/mysong/bass.mid
  engine=basic_pitch notes=47 dur=180.5s

[drums] -> /content/drive/MyDrive/stem-to-midi/output/mysong/drums.mid
  engine=adtof notes=342 dur=180.5s

[guitar] -> /content/drive/MyDrive/stem-to-midi/output/mysong/guitar.mid
  engine=basic_pitch notes=156 dur=180.5s

[piano] -> /content/drive/MyDrive/stem-to-midi/output/mysong/piano.mid
  engine=basic_pitch notes=89 dur=180.5s

[vocals] -> /content/drive/MyDrive/stem-to-midi/output/mysong/vocals.mid
  engine=basic_pitch notes=24 dur=180.5s

Merged -> /content/drive/MyDrive/stem-to-midi/output/mysong/mysong.mid
  tracks=5 total_notes=658 duration=180.5s
```

✅ **Done!** Your MIDI is ready.

---

## Step 6: Review the results

### Multi-track piano roll

Run **Cell 6** to see a visual breakdown:
- **Top graph:** All pitched instruments (vocals, bass, guitar, piano) as colored bars
- **Bottom graph:** Drums with kick/snare/hihat labeled

### Audio preview

Run **Cell 7** to hear a synthesized preview of the merged MIDI:
- Pitched instruments: sine wave tone
- Drums: noise bursts per hit

This is **not** the original audio — just the MIDI playing back so you can verify notes were captured correctly.

---

## Step 7: Download your files

Run **Cell 8**.

You'll see two types of files:

**Merged file** (the main output):
- `mysong.mid` — all 5 tracks merged, ready to import into your DAW

**Per-stem files** (optional, for fine-tuning):
- `vocals.mid` — just the vocal track
- `drums.mid` — just the drum notes
- `bass.mid` — just the bass track
- (etc.)

Click the download link, or grab them from Google Drive.

---

## Step 8: Open in your DAW

1. Open **Ableton**, **Logic**, **Cubase**, **Studio One**, or any DAW
2. `File > Import MIDI` or drag `mysong.mid` into the arrangement
3. The DAW creates one track per stem with the correct GM (General MIDI) program assigned:
   - Vocals → Voice Oohs (program 53)
   - Bass → Electric Bass (program 33)
   - Guitar → Electric Guitar (program 27)
   - Piano → Acoustic Grand Piano (program 0)
   - Drums → GM drum channel (kick, snare, hihat labeled)

4. Hit play — your stems now play as MIDI! 🎵

---

## Troubleshooting 🔧

| Problem | Solution |
|---------|----------|
| **"No recognizable stems found"** | Your files are named wrong. Check they match `vocals.wav`, `drums.wav`, `bass.wav`, `guitar.wav`, `piano.wav`, or `other.wav`. Use `STEM_RENAMES = {...}` if needed. |
| **Very few notes detected** | The audio might be too quiet or noisy. Try running Demucs with a different model (`htdemucs` instead of `htdemucs_6s`). |
| **Drums are missing** | ADTOF (the drum engine) only works on isolated drum stems. If your drums.wav still has some bass bleeding, it may fail. Try again with cleaner separation. |
| **Installation errors in Cell 1** | Restart runtime after installation. Some packages need a fresh Python session to load. |
| **"Drive not mounted"** | Run Cell 3 again to mount Google Drive. |

---

## Next steps 🚀

- **Edit in your DAW:** Use the per-stem MIDI files to tweak individual instruments
- **Quantize & tempo-sync:** Most DAWs have quantize tools — snap notes to a grid
- **Export MIDI:** Save as `.mid` to share the transcription
- **Re-import originals:** Keep both MIDI + original audio stems for reference

---

## How accurate is it?

- **Pitched instruments** (vocals, bass, guitar, piano): ~85–95% accurate, good enough for editing
- **Drums:** ~90% accurate for kick/snare/hihat; toms/cymbals less reliable
- **Polyphony:** Can struggle with thick chords or complex harmonies — you may need to clean up overlapping notes in the DAW

The output is **a starting point**, not perfect transcription. Always review in the DAW and adjust by ear.

---

## Questions? 💬

- **Stems are too short/incomplete?** Demucs may have truncated them. Try a shorter song first.
- **One instrument sounds wrong?** Drums and vocals are hardest to get right. Try increasing the audio quality or using a cleaner source.
- **Want to transcribe a single stem (not batch)?** Use `notebooks/Stem_to_MIDI_MVP.ipynb` instead — it's for one file at a time.

---

**That's it!** You've converted audio stems to editable MIDI. 🎉
