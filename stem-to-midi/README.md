# 🎼 stem-to-midi

Pipeline de conversão **Stem → MIDI**: complementa o `stem_separator.ipynb` do
repositório, pegando cada stem isolado pelo Demucs e transcrevendo em uma faixa
MIDI. Projeto irmão do `audio-restoration-pipeline/`.

## Status

| Fase | Escopo | Estado |
|---|---|---|
| **MVP** | 1 stem → 1 arquivo MIDI usando Basic Pitch (universal) | ✅ pronto |
| **Fase 2a** | Engine dedicada para drums (ADTOF-pytorch) + dispatcher por stem_type | ✅ pronto |
| Fase 2b | Engines dedicadas para piano (Bytedance) e vocals (ROSVOT) | ⏳ planejado |
| **Fase 3** | Batch sobre todos os stems de uma faixa Demucs + arquivo `.mid` multi-track unificado | ✅ pronto |
| Fase 4 | Quantização, transposição, exportação MusicXML | ⏳ planejado |

## Nota sobre Python 3.12 / Colab

basic-pitch 0.4.0 (única release estável, ago/2024) declara suporte só até
Python 3.11 e, em Linux com `python>=3.11`, puxa `tensorflow<2.15.1` e
`tflite-runtime` — nenhum dos dois tem wheel para Python 3.12. O Colab agora
roda Python 3.12, então `pip install basic-pitch` direto **quebra**
(o resolver do pip backtraca até basic-pitch 0.2.6 e tenta compilar
numpy 1.23 do source). Ver [issue #188](https://github.com/spotify/basic-pitch/issues/188).

**Workaround usado no notebook:** instalar `basic-pitch --no-deps` e o backend
**ONNX** (`onnxruntime`, com wheel para Python 3.12). O modelo `nmp.onnx` já
vem embutido no pacote, e `predict()` escolhe o primeiro backend importável na
ordem `TF → CoreML → TFLite → ONNX` — sem TF/TFLite instalados, ele usa ONNX
automaticamente.

```bash
pip install basic-pitch --no-deps onnxruntime
pip install -r requirements.txt
```

## MVP — uso rápido

1. Abrir `notebooks/Stem_to_MIDI_MVP.ipynb` no Google Colab (badge no topo do
   notebook).
2. Colocar o arquivo do stem em
   `/content/drive/MyDrive/stem-to-midi/input/stem.wav` (`.mp3`, `.flac`,
   `.m4a` também funcionam).
3. Selecionar o `STEM_TYPE` correto na célula 4 (`bass`, `vocals`, `guitar`,
   `piano`, `drums` ou `other`).
4. Rodar as células sequencialmente — o MIDI sai em
   `/content/drive/MyDrive/stem-to-midi/output/stem.mid`.

O dispatcher escolhe a engine automaticamente:

- `drums` → **ADTOF-pytorch** (kick/snare/hat/tom/cymbals, GM drum map em
  canal 10)
- demais → **Basic Pitch** com preset por instrumento (`onset_threshold`,
  `frame_threshold`, faixa de frequência) — ver `modules/transcribe.py` →
  `STEM_PRESETS`.

### Nota sobre licenças

- **Basic Pitch:** Apache-2.0 (permissivo)
- **ADTOF-pytorch:** porte PyTorch de [ADTOF](https://github.com/MZehren/ADTOF)
  (AGPL-3.0). O fork [xavriley/ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch)
  não publicou um LICENSE explícito — o termo aplicável é, por inferência,
  o AGPL do upstream. Use o stem `drums` somente se aceita a AGPL.

## Uso via Python (fora do Colab)

```python
from modules import stem_to_midi

# Pitched: usa Basic Pitch com preset 'bass'
result = stem_to_midi(
    input_path='/path/to/bass.wav',
    output_path='/path/to/bass.mid',
    stem_type='bass',
)
print(f'{result.engine}: {result.n_notes} notas em {result.duration_s:.1f}s')

# Drums: dispatcha para ADTOF-pytorch
result = stem_to_midi(
    input_path='/path/to/drums.wav',
    output_path='/path/to/drums.mid',
    stem_type='drums',
)
```

## Batch — múltiplos stems → MIDI multi-track (Fase 3)

`batch_stems_to_midi()` aceita uma pasta de stems (layout do Demucs:
arquivos `vocals.wav`, `drums.wav`, `bass.wav`, `guitar.wav`, `piano.wav`,
`other.wav`), transcreve cada um com a engine certa e consolida tudo em
um único MIDI multi-track.

```python
from modules import batch_stems_to_midi

result = batch_stems_to_midi(
    stems_dir='/path/to/demucs/separated/htdemucs_6s/song',
    output_path='/path/to/output/song.mid',
)
print(f'{len(result.stem_results)} stems, {result.total_notes} notas, '
      f'{result.duration_s:.1f}s')
for stem_type, r in result.stem_results.items():
    print(f'  {stem_type:8s} {r.engine:14s} {r.n_notes:5d} notas')
```

Produz, ao lado de `output/song.mid`, um arquivo `<stem>.mid` por stem
processado (útil para edição isolada em DAW). Atribui um GM program por
faixa (`vocals`=53 Voice Oohs, `bass`=33 Electric Bass, `guitar`=27
Clean Guitar, `piano`=0 Acoustic Grand, `other`=0 Acoustic Grand,
`drums`=canal 10). Nomes não-padronizados podem ser remapeados via
`stem_types={'lead_vox': 'vocals'}`.

Notebook Colab dedicado: `notebooks/Stem_to_MIDI_Batch.ipynb`.

## Levantamento técnico — ferramentas Audio→MIDI (maio/2026)

A pesquisa abaixo embasou a escolha do Basic Pitch como engine do MVP e o
roadmap dos transcritores especializados.

### Por que **Basic Pitch** no MVP

- Licença Apache 2.0 (sem contaminação).
- Já validado no `Song_to_MIDI_Converter.ipynb` deste repositório.
- Modelo único de ~17 MB, roda em CPU em tempo razoável.
- Cobre `vocals / bass / guitar / piano / other` com qualidade decente.
- API estável (`basic_pitch.inference.predict`).

Limitações conhecidas: não transcreve drums, perde nuances de piano denso, e
velocity é apenas aproximada.

### Tabela de recomendações por stem (`htdemucs_6s`)

| Stem | Recomendação principal | Alternativa | Licença |
|---|---|---|---|
| **vocals** | [ROSVOT](https://github.com/RickyL-2000/ROSVOT) (ACL 2024) | CREPE + [CREPE Notes](https://arxiv.org/pdf/2311.08884); YourMT3+ | MIT |
| **drums**  | [ADTOF](https://github.com/MZehren/ADTOF) (5–7 classes + velocity) | OaF Drums (Magenta) | AGPL-3.0 |
| **bass**   | CREPE + CREPE Notes (mono, pitch limpo) | Basic Pitch | MIT |
| **guitar** | Basic Pitch | [YourMT3+](https://github.com/mimbres/YourMT3) | Apache-2.0 |
| **piano**  | [Bytedance `piano_transcription_inference`](https://github.com/qiuqiangkong/piano_transcription_inference) (SOTA com pedal+velocity) | [Sony hFT-Transformer](https://github.com/sony/hFT-Transformer) (F1 96.7%) | Apache-2.0 / MIT |
| **other**  | Basic Pitch | YourMT3+ | Apache-2.0 |

### Engines avaliadas — resumo

| Engine | Foco | Lic. | Mantida | Observação |
|---|---|---|---|---|
| Spotify Basic Pitch 0.4.0 | Pitched geral | Apache-2.0 | sim (jan/2026) | Coringa universal |
| Bytedance Piano Transcription | Piano | Apache-2.0 | inferência mantida (qiuqiangkong) | Repo original arquivado dez/2025 |
| Sony hFT-Transformer | Piano | MIT | sim | Sem pacote pip — clonar repo |
| Magenta MT3 / Onsets & Frames | Multi / Piano / Drums | Apache-2.0 | parado (último commit 2023) | Stack TF1/JAX desatualizada |
| YourMT3+ (MLSP 2024) | Multi + vocals | **GPL-3.0** | sim | Modelo único cobre tudo, mas GPL |
| ADTOF | Drums (5–7 classes) | **AGPL-3.0** | sim ([arXiv 2509.24853](https://arxiv.org/abs/2509.24853)) | Único drum-AMT moderno open |
| CREPE + CREPE Notes | Mono pitch (vocais / baixo) | MIT / ISC | sim | Pós-processador segmenta em notas |
| ROSVOT | Vocais cantadas | MIT | sim (ACL 2024) | Robusto a ruído de acompanhamento |

### Pipeline-alvo (Fase 2)

```
song.mp3
  └─► Demucs htdemucs_6s ─┬─► vocals.wav ─► ROSVOT ─────────► vocals.mid
                          ├─► drums.wav  ─► ADTOF ──────────► drums.mid
                          ├─► bass.wav   ─► CREPE + Notes ──► bass.mid
                          ├─► guitar.wav ─► Basic Pitch ────► guitar.mid
                          ├─► piano.wav  ─► Bytedance ──────► piano.mid
                          └─► other.wav  ─► Basic Pitch ────► other.mid
                                                                 │
                                                                 ▼
                                                            merge → song.mid (multi-track)
```

## Estrutura

```
stem-to-midi/
├── README.md                              # este arquivo
├── requirements.txt
├── modules/
│   ├── __init__.py
│   └── transcribe.py                      # stem_to_midi() + batch_stems_to_midi()
└── notebooks/
    ├── Stem_to_MIDI_MVP.ipynb             # 1 stem → 1 MIDI
    └── Stem_to_MIDI_Batch.ipynb           # N stems Demucs → 1 MIDI multi-track
```

## Referências

- [spotify/basic-pitch](https://github.com/spotify/basic-pitch)
- [bytedance/piano_transcription](https://github.com/bytedance/piano_transcription) + [pip inference](https://pypi.org/project/piano-transcription-inference/)
- [sony/hFT-Transformer](https://github.com/sony/hFT-Transformer)
- [magenta/mt3](https://github.com/magenta/mt3) · [OaF Drums](https://magenta.tensorflow.org/oaf-drums)
- [mimbres/YourMT3](https://github.com/mimbres/YourMT3) · [arXiv 2407.04822](https://arxiv.org/html/2407.04822v1)
- [MZehren/ADTOF](https://github.com/MZehren/ADTOF) · [arXiv 2509.24853](https://arxiv.org/abs/2509.24853)
- [marl/crepe](https://github.com/marl/crepe) · [CREPE Notes arXiv 2311.08884](https://arxiv.org/pdf/2311.08884)
- [RickyL-2000/ROSVOT](https://github.com/RickyL-2000/ROSVOT)
