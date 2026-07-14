# 🎵🔎 track-digest

Duas ferramentas para **pós-processar** uma pasta `<track>/` gerada pelo
[`music-to-midi`](../music-to-midi): um **resumo compacto p/ IA** e um **cortador
de stems em lote**. Feito para rodar bem no Colab pelo celular.

```
music-to-midi/output/<track>/          (entrada — já existe)
├── stems/        vocals.wav, drums.wav, bass.wav, ...
├── midi/         vocals.mid, drums.mid, ...
├── analysis/     <track>.txt, <stem>.txt (MIDI-TEXT completo)
└── <track>.mid
        │
        ├─► digest  ─► analysis/<track>_digest.txt   (resumo curto p/ IA)
        └─► crop    ─► <track>/crops/<stem>_<ini>-<fim>s.wav
```

## Por que existe

O `.txt` de análise do `music-to-midi` (MIDI-TEXT) lista **cada nota** — ótimo p/
recriar o MIDI, mas caro demais em tokens p/ colar num assistente de IA. E não há
como recortar um trecho dos stems e exportar `.wav`. Este projeto resolve os dois.

---

## Feature 1 — `DIGEST v1` (resumo compacto p/ IA)

Um resumo determinístico (sem LLM, sem API key) que combina:

- **Do MIDI** (`midi/*.mid`): andamento, tom/escala estimados, extensão de pitch,
  densidade de notas (por segundo e por compasso), caráter mono ↔ acordal, intervalos
  dominantes, durações rítmicas dominantes, grade rítmica (16/8/quarto + síncope) e
  dinâmica de velocity. Bateria vira padrão de groove (ex.: *four-on-floor + backbeat*).
- **Do áudio** (`stems/*.wav`, via `librosa`): **timbre** — brilho (centróide
  espectral → "dark/low-pass" vs "airy"), textura (flatness → "tonal" vs "distorted"),
  dinâmica (crest → "compressed" vs "punchy") e harmonicidade (HPSS).
  *O timbre não existe no MIDI — só o áudio informa "guitarra grave com breakup".*

Um track de 6 stems cabe em ~250 tokens (vs. milhares do dump nota-a-nota).

### Exemplo de saída

```
DIGEST v1 | track: Emi Grace - Know Better | 4/4 | 118 bpm (audio) | 183.4s | key: C# minor (conf +0.79)
UNITS: pitch=sci-name | dens=notes/s | vel=0-127 | centroid=Hz | crest=dB | h=harmonic-ratio
OVERVIEW: 6 stems | polyphonic | busy | dynamic

[vocals] mono | E3..A4 | 214 notes 1.2/s ~3.9/bar | key-fit C#min | intervals M3,P5
  rhythm: 8th grid, some syncopation, tight 0.88 | durs 1/8,1/4 | vel 54-112 wide
  timbre: bright, tonal, dynamic  (centroid 2740 · flat 0.03 · crest 14.2 · h 0.84 · onset 1.5/s)
[drums] kick+snare+hihat | 3.1 hits/s | four-on-floor, backbeat, 16th hats
  timbre: dark, noisy, punchy  (centroid 1900 · flat 0.21 · crest 20.6 · h 0.18 · onset 3.0/s)
```

---

## Feature 2 — cortar stems em lote → `.wav`

Escolhe uma região `[início, fim]` e corta **todos os stems de uma vez** (ex.: pegar
o refrão em vocals/drums/bass/guitar juntos), exportando `.wav` PCM_24 em
`<track>/crops/`. O nome preserva a região (`vocals_012p50-030p00s.wav`, sem pontos).
O `fim` é limitado à duração de cada stem.

---

## Uso — Colab (celular)

Abra `notebooks/Digest_and_Crop.ipynb`, rode as células em ordem:
instalar deps → montar o Drive → escolher a track num menu → gerar o digest / cortar.
Nada de digitar caminhos.

## Uso — CLI

```bash
cd track-digest

# resumo p/ IA -> analysis/<track>_digest.txt
python run.py digest "../music-to-midi/output/My Song"
python run.py digest "../music-to-midi/output/My Song" --timbre-from raw --bpm 120

# cortar todos os stems -> <track>/crops/*.wav
python run.py crop "../music-to-midi/output/My Song" --start 12.5 --end 30.0
python run.py crop "../music-to-midi/output/My Song" --start 12.5 --end 30.0 --include vocals,bass
```

## Uso — Python

```python
from modules import build_track_digest, render_digest, write_digest, crop_stems

td = build_track_digest("output/My Song", timbre_from="raw")
print(render_digest(td))
write_digest(td, "output/My Song/analysis/My Song_digest.txt")

crop_stems("output/My Song", start_s=12.5, end_s=30.0)   # -> crops/*.wav
```

## Instalação

```bash
pip install -r requirements.txt
```

Stack leve (`librosa`, `soundfile`, `pretty_midi`, `numpy`, `scipy`, `ipywidgets`) —
sem `demucs`/`basic-pitch`/`ADTOF`, pois só consome os `.mid`/`.wav` já gerados.

## Precisão / limitações

- **Andamento é a peça-chave.** MIDIs do Basic Pitch trazem 120 BPM default, o que
  quebraria notas/compasso e a grade rítmica. Por isso o BPM é estimado do **áudio**
  (`beat_track` na bateria; fallback: outro stem → MIDI). O cabeçalho mostra a fonte
  (`audio`/`midi`/`manual`); passe `--bpm`/`bpm=` se a estimativa errar a oitava.
- **Tom estimado** por perfis Krumhansl, ponderado por duração p/ resistir a ruído de
  transcrição; vem com `conf`. O tom por stem (bass/vocals) pode divergir do track — é
  esperado. Baixa confiança = tentativo.
- **Timbre** é lido dos stems **crus** (`stems/`) por padrão; `stems_clean/` é
  denoised+normalizado e distorce centróide/flatness/crest (`timbre_from="clean"` p/ forçar).
- **Bateria** usa o mapa GM (`is_drum`); pitch/tom/intervalos não se aplicam.
- **Colab no celular:** análise usa `sr=22050` e uma janela de `analyze_seconds`
  (padrão 60s) por stem p/ ser rápida; o corte é sempre no áudio completo.

## Estrutura

```
track-digest/
├── README.md · requirements.txt · .gitignore · run.py
├── input/README.txt
├── modules/
│   ├── common.py          list_audio_files, canonical_stem_type, resolve_track_paths
│   ├── theory.py          note_name, GM_DRUMS, drum_group, estimate_key (Krumhansl)
│   ├── descriptors.py     bandas de limiar → palavras (brilho, textura, dinâmica…)
│   ├── midi_features.py    densidade, polifonia, intervalos, grade rítmica, groove
│   ├── audio_features.py   centróide/rolloff/flatness/crest/HPSS/onset via librosa
│   ├── digest.py          orquestra + renderiza o DIGEST v1
│   └── crop.py            corte em lote dos stems → WAV
└── notebooks/Digest_and_Crop.ipynb
```
