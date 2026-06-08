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
├── stems/        vocals.wav, drums.wav, ...   (separados pelo Demucs)
├── stems_clean/  vocals.wav, drums.wav, ...   (mono + denoise + normalize; transcritos)
├── midi/         vocals.mid, drums.mid, ...   (um .mid por stem, p/ editar na DAW)
├── analysis/     <stem>_pianoroll.png + <stem>.txt   (imagem + texto p/ IA)
└── <track>.mid   MIDI multi-track consolidado
```

## Limpeza dos stems (do `Stems_4`)

Antes de transcrever, cada stem passa pela limpeza do caderno `Stems_4` (ligada
por padrão, `clean_stems=True`):

1. **mono true-mid** — down-mix `(L + R) / 2` (não descarta canal);
2. **denoise leve** — `noisereduce` com `prop_decrease=0.15`;
3. **normalização segura** — pico em `0.90`.

Os stems crus ficam em `stems/` e os limpos (que são os efetivamente
transcritos) em `stems_clean/`. Desligue com `clean_stems=False` (ou `--no-clean`
na CLI) para transcrever o áudio cru do Demucs.

## Escolher a música

Coloque os arquivos em `music-to-midi/input/` (qualquer wav/mp3/flac/m4a/ogg).
No notebook, a célula 4 lista o conteúdo dessa pasta e mostra um **menu suspenso**
para você escolher a track — sem digitar nome nenhum.

## Análise: piano roll + texto legível por IA

Para cada stem (e para o MIDI consolidado), o pipeline grava em `analysis/`:

- **`<stem>_pianoroll.png`** — o piano roll daquela faixa.
- **`<stem>.txt`** — uma representação textual, linha a linha, de todas as notas
  (tempo, duração, pitch/nota, velocity), com um cabeçalho auto-explicativo. É um
  formato que uma IA consegue ler para "sacar" a melodia/padrão **e regenerar o
  MIDI**. O round-trip é provado por `text_to_midi()`:

  ```python
  from modules import text_to_midi
  text_to_midi("output/minha-musica/analysis/minha-musica.txt", "recriado.mid")
  ```

  Trecho do `.txt`:

  ```
  # MIDI-TEXT v1 — representação textual de um MIDI, recriável por IA.
  TEMPO_BPM: 120.00
  DURATION_S: 1.950
  N_TRACKS: 2
  TRACK_BEGIN
  NAME: vocals
  PROGRAM: 53
  IS_DRUM: false
  N_NOTES: 4
  PITCH_RANGE: C4..C5
  # NOTE  start_s  dur_s  pitch  midi  vel
  NOTE	0.000	0.450	C4	60	90
  NOTE	0.500	0.450	E4	64	91
  ...
  TRACK_END
  END
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
# sem a limpeza mono/denoise (transcreve o áudio cru do Demucs):
python run.py "minha-musica.mp3" -o output/ --no-clean
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
├── input/                                  # solte aqui as músicas (notebook lista p/ escolher)
├── modules/
│   ├── __init__.py
│   ├── separate.py                         # Demucs: música → stems
│   ├── cleanup.py                          # mono + denoise + normalize (do Stems_4)
│   ├── transcribe.py                       # 1 stem → MIDI (Basic Pitch / ADTOF)
│   ├── batch.py                            # stems → MIDI multi-track
│   ├── analyze.py                          # piano roll PNG + MIDI-TEXT (+ text_to_midi)
│   └── pipeline.py                         # song_to_midi() (cola tudo)
└── notebooks/
    └── Song_to_Stems_to_MIDI.ipynb         # workflow único (Colab)
```
