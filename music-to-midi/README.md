# 🎵→🎼 music-to-midi

Workflow **único** que pega uma música inteira, separa em stems com
**Demucs** e transcreve cada stem para **MIDI multi-track** — numa só passagem.

Unifica os dois cadernos de Colab que antes eram etapas separadas:

| Antes | Etapa | Agora |
|---|---|---|
| `Stems_4` (`stem_separator`) | separação música → stems | `modules/separate.py` |
| `Stem_to_MIDI_Batch` (`stem-to-midi/`) | stems → MIDI multi-track | `modules/batch.py` + `transcribe.py` |
| — | cola dos dois | `modules/pipeline.py` → `song_to_midi()` |

```
song.mp3
  └─► Demucs htdemucs_6s ─┬─► vocals  ─► Basic Pitch ──► vocals.mid
                          ├─► drums   ─► ADTOF ───────► drums.mid
                          ├─► bass    ─► Basic Pitch ──► bass.mid
                          ├─► guitar  ─► Basic Pitch ──► guitar.mid
                          ├─► piano   ─► Basic Pitch ──► piano.mid
                          └─► other   ─► Basic Pitch ──► other.mid
                                                            │
                                                            ▼
                                              merge → <track>.mid (multi-track)
```

## Saída organizada

Tudo cai em `<output_root>/<track>/`:

```
<track>/
├── stems/        vocals.wav, drums.wav, bass.wav, guitar.wav, piano.wav, other.wav
├── midi/         vocals.mid, drums.mid, ...   (um .mid por stem, p/ editar na DAW)
└── <track>.mid   MIDI multi-track consolidado
```

## Engines por stem

| Stem | Engine | GM program / canal |
|---|---|---|
| `vocals` | Basic Pitch (ONNX) | 53 — Voice Oohs |
| `bass`   | Basic Pitch (ONNX) | 33 — Electric Bass |
| `guitar` | Basic Pitch (ONNX) | 27 — Electric Guitar (clean) |
| `piano`  | Basic Pitch (ONNX) |  0 — Acoustic Grand Piano |
| `other`  | Basic Pitch (ONNX) |  0 — Acoustic Grand Piano |
| `drums`  | ADTOF-pytorch      | canal 10 (GM drum map) |

O roteamento de engine e os presets por instrumento vêm do projeto irmão
[`stem-to-midi`](../stem-to-midi/) (mesmo dispatcher e `STEM_PRESETS`).

## Uso no Colab (recomendado)

Abra `notebooks/Song_to_Stems_to_MIDI.ipynb` (badge no topo do notebook),
rode a célula 1, **reinicie o runtime**, aponte `SONG_PATH` para a música e
rode o resto. O MIDI consolidado sai em `OUTPUT_ROOT/<track>/<track>.mid`.

## Uso local (CLI)

```bash
cd music-to-midi
pip install -r requirements.txt
pip install git+https://github.com/xavriley/ADTOF-pytorch.git   # bateria

python run.py "minha-musica.mp3" -o output/
# qualidade maior (mais lento), igual ao notebook Stems_4:
python run.py "minha-musica.mp3" -o output/ --shifts 5 --overlap 0.75
```

## Uso via Python

```python
from modules import song_to_midi

result = song_to_midi(
    "minha-musica.mp3",
    "output/",
    model="htdemucs_6s",     # 6 stems; use "htdemucs" p/ 4 stems mais rápido
    shifts=1, overlap=0.25,
)
print(result.merged_midi)                      # output/minha-musica/minha-musica.mid
print(result.transcription.total_notes)
for stem, r in result.transcription.stem_results.items():
    print(stem, r.engine, r.n_notes)
```

Etapas isoladas também estão expostas:

```python
from modules import separate_stems, batch_stems_to_midi

sep = separate_stems("song.mp3", "out/song/stems", model="htdemucs_6s")
res = batch_stems_to_midi("out/song/stems", "out/song/song.mid")
```

## Nota sobre Python 3.12 / Colab

`basic-pitch` 0.4.0 declara suporte só até Python 3.11 e, em Linux, puxa
`tensorflow<2.15.1` / `tflite-runtime` — sem wheel para 3.12 (que é o que o
Colab roda hoje). Workaround usado no notebook e no `requirements.txt`:
instalar `basic-pitch --no-deps` + `onnxruntime` e cair no backend **ONNX**
(modelo `nmp.onnx` já embutido). Ver
[spotify/basic-pitch#188](https://github.com/spotify/basic-pitch/issues/188).

## Licenças das engines

- **Demucs:** MIT
- **Basic Pitch:** Apache-2.0 (permissivo)
- **ADTOF-pytorch:** porte de [ADTOF](https://github.com/MZehren/ADTOF)
  (AGPL-3.0). Use o stem `drums` apenas se aceita a AGPL.

## Estrutura

```
music-to-midi/
├── README.md
├── requirements.txt
├── run.py                                  # CLI local
├── modules/
│   ├── __init__.py
│   ├── separate.py                         # Demucs: música → stems
│   ├── transcribe.py                       # 1 stem → MIDI (Basic Pitch / ADTOF)
│   ├── batch.py                            # stems → MIDI multi-track
│   └── pipeline.py                         # song_to_midi() (cola tudo)
└── notebooks/
    └── Song_to_Stems_to_MIDI.ipynb         # workflow único (Colab)
```
