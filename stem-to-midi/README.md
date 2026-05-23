# 🎼 stem-to-midi

Pipeline de conversão **Stem → MIDI**: complementa o `stem_separator.ipynb` do
repositório, pegando cada stem isolado pelo Demucs e transcrevendo em uma faixa
MIDI. Projeto irmão do `audio-restoration-pipeline/`.

## Status

| Fase | Escopo | Estado |
|---|---|---|
| **MVP** | 1 stem → 1 arquivo MIDI usando Basic Pitch (universal) | ✅ pronto |
| Fase 2 | Roteamento por tipo de stem (drums dedicado, piano dedicado) | ⏳ planejado |
| Fase 3 | Batch sobre todos os stems de uma faixa Demucs + arquivo `.mid` multi-track unificado | ⏳ planejado |
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
   `piano` ou `other`).
4. Rodar as células sequencialmente — o MIDI sai em
   `/content/drive/MyDrive/stem-to-midi/output/stem.mid`.

Os presets de `onset_threshold`, `frame_threshold`, `minimum_note_length` e
faixa de frequência são ajustados por tipo de stem (ver `modules/transcribe.py`
→ `STEM_PRESETS`).

> **Aviso:** este MVP usa Basic Pitch, que **não transcreve drums**. Para o
> stem `drums` use a Fase 2 (ADTOF).

## Uso via Python (fora do Colab)

```python
from modules import stem_to_midi

result = stem_to_midi(
    input_path='/path/to/bass.wav',
    output_path='/path/to/bass.mid',
    minimum_frequency=30.0,
    maximum_frequency=350.0,
)
print(f'{result.n_notes} notas em {result.duration_s:.1f}s → {result.midi_path}')
```

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
│   └── transcribe.py                      # stem_to_midi() + STEM_PRESETS
└── notebooks/
    └── Stem_to_MIDI_MVP.ipynb             # notebook MVP (Colab)
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
