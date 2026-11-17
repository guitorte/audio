# 🥁 Masterização por stems: remodelar a bateria e masterizar a música

🇺🇸 [English version](README.md)

*Escrito para ser lido a frio, meses depois.* Esta pasta guarda o processo usado em **raio-x** (nov. 2026): pegar os stems de uma música, dividir o stem de bateria nas peças do kit, dar a cada peça mais punch e um timbre mais limpo, domar um shaker robótico, remontar a música e masterizá-la.

São scripts de linha de comando, ainda sem notebook. Tudo roda em CPU; a GPU só acelera a separação.

```
stems/ (drums, bass, guitar, other, piano, vocals)
   │
   ├─ drums ──► separate.py (DrumSep) ──► kick · snare · toms · hh · ride · crash  (+ resíduo)
   │                                         │
   │                          drums.py: cada peça → transient shaper → EQ → compressor
   │                                    hh (opcional) → humanizar: nível por batida, ducking sob a voz
   │                                         │
   │                          peças processadas + resíduo ──► teto de pico ──► drums (shaped)
   │                                                                              │
   └─ os outros stems ───────────────────────────────────────────────────────────┴──► mix.wav
                                                                                        │
                                              master.py ──► master (−13 LUFS, −1 dBTP) + cópias A/B
                                              check.py  ──► o que mudou, com volume igualado
```

---

## 1. Instalação

```bash
pip install "audio-separator[cpu]" audioread librosa pyloudnorm soundfile scipy   # [gpu] no Colab
cd stem-mastering
```

O `audioread` é necessário porque o audio-separator o importa sem declarar. Os modelos baixam no primeiro uso (o DrumSep tem cerca de 400 MB).

## 2. Receita

**Se você só tem a música,** tire primeiro um stem de bateria (13 min em 2 núcleos de CPU para uma música de 2,5 min; segundos em GPU):

```bash
python separate.py musica.flac htdemucs_ft.yaml stems/
```

**Com os stems** (um arquivo por stem; o nome do stem de bateria contém `drum`, o da voz contém `vocal`):

```bash
python drums.py stems/ out/ --humanize-hh          # divide a bateria sozinho (≈10 min em CPU)
python drums.py stems/ out/ --kit out/kit_split    # reaproveita uma divisão anterior
python master.py out/mix.wav out/master.wav --target -13 --ab-level -15
python check.py out/mix.wav out/master.wav --drums "out/drums (shaped).flac" --vocals stems/vocals.flac
```

Deixe de fora o `--humanize-hh` se o chimbal estiver soando bem. Gere MP3s a partir do master de 24 bits com `ffmpeg -i master.wav -b:a 320k master.mp3`.

**O que sai em `out/`:**

| Arquivo | O que é |
|---|---|
| `kit/<peça> (shaped).flac` | cada peça do kit depois da sua cadeia; `residual (untouched).flac` é o que a divisão não pegou |
| `kit/hh (humanized).flac` | o chimbal/shaker humanizado (com `--humanize-hh`) |
| `drums (shaped).flac` | o novo stem de bateria |
| `full (shaped drums).flac`, `instrumental (shaped drums).flac` | a música com a bateria nova, 24 bits, sem master |
| `mix.wav` | a mesma mixagem completa em float de 32 bits: a entrada do `master.py` |
| `master.wav`, `master_AB_mix.wav`, `master_AB_master.wav` | o master, e mix e master no mesmo volume para uma comparação justa |

## 3. As cadeias da bateria (`dsp.CHAINS`)

Um **transient shaper** acompanha o nível de cada peça com um envelope rápido e um lento. Onde o rápido está acima do lento, uma batida está começando (ataque); onde está abaixo, ela está morrendo (sustentação). Cada parte recebe seu próprio ganho, então um bumbo pode bater mais forte e soar mais curto.

| Peça | Ataque / sustentação | EQ | Compressor |
|---|---|---|---|
| kick | +5 / −3 dB | +2,5 dB em 60 Hz, −3 em 320 Hz (som de caixa), +3 em 3,8 kHz (clique do batedor) | 3:1, 10 ms / 90 ms |
| snare | +4 / +1 dB | +2 em 200 Hz (corpo), −2 em 500 Hz, +2,5 em 5,5 kHz (estalo), +1,5 de shelf em 11 kHz | 2,5:1 |
| toms | +4 / +1,5 dB | −2,5 em 450 Hz (embolado), +2 em 4 kHz | 2:1 |
| hh | +2 / −2 dB | −2 em 400 Hz, +1,5 de shelf em 10 kHz | nenhum |
| ride, crash | +1 a +1,5 dB de ataque | um pouco de ar | nenhum |

Cada peça volta **ao nível de entrada** (RMS), então o equilíbrio do kit fica onde a música tinha: a bateria ganha punch, não volume. Um **teto de pico** com look-ahead mantém o novo barramento de bateria até 1 dB acima do pico do stem original.

Em raio-x: o resíduo ficou 26,7 dB abaixo da bateria, o fator de crista do barramento subiu de 19,5 para 20,6 dB e o teto de pico trabalhou no máximo 1,3 dB.

## 4. Humanizar um shaker robótico (`--humanize-hh`)

Faixas geradas ou muito quantizadas costumam ter um shaker ou chimbal tocando colcheias retas, no mesmo nível do início ao fim. Soa como um metrônomo. Em raio-x ele ia de 0:00 a 2:08, variando no máximo uns 3 dB, preso à grade (14 ms de variação de tempo).

O que a opção faz nessa camada:

- **Nível por batida.** As batidas no tempo ficam em 0 dB e as do contratempo caem 2,5 dB. Por cima entram uma variação lenta (±1 dB em ciclos de 8 e 3 compassos) e um pequeno desvio aleatório com semente fixa (σ 1,2 dB, limitado a ±2,5). O ritmo não muda; só o nível das batidas.
- **Ducking:** até 3 dB a menos enquanto a voz canta (precisa do stem de voz).
- **Nível e timbre:** 3 dB mais baixo no geral, shelf de −2 dB acima de 9 kHz.

**Como o tempo é encontrado:** os ataques vêm da faixa de 4–16 kHz da camada. A grade é o período em que as batidas mais se alinham, procurado dentro de ±3% do espaçamento mediano entre elas. Dois métodos mais simples falharam em raio-x:
- **Só o espaçamento mediano** deu 0,1858 s em vez de 0,1875 s, porque a imprecisão dos ataques o distorce.
- **Contar de batida em batida** inverte todos os rótulos de tempo/contratempo depois de uma única batida perdida.

Confira de ouvido: se os acentos caírem no contratempo, a metade mais forte das batidas não era a do tempo.

Em raio-x a camada ficou em média 7 dB mais baixa, porque a voz canta 83% do tempo.

## 5. O master (`master.py`)

| # | Etapa | Ajuste | Por quê |
|---|---|---|---|
| 1 | passa-altas | 25 Hz, fase zero | ronco de subgrave; fase zero para não aumentar os picos |
| 2 | corte dinâmico de grave | 65–100 Hz, 2:1, até 3 dB | doma uma nota de baixo só quando ela incha |
| 3 | EQ fixo (`dsp.MASTER_EQ`) | −1,5 dB em 180 Hz, +2,5 em 2,5 kHz, +1,5 em 5 kHz, +2,5 de shelf em 11 kHz | embolado, buraco de presença, ar (típico de stems separados) |
| 4 | de-esser | canal central, 5–9 kHz, até 3 dB | o brilho extra não deve afiar os "s" |
| 5 | compressor de cola | 1,5:1, ataque de 30 ms, release de 300 ms | ~1 dB no máximo; o ataque lento deixa passar as batidas remodeladas |
| 6 | ajuste de nível por trecho | janelas de 3 s, corrige desvios > 0,7 dB, transições de 2,5 s | mantém as proporções do arranjo (veja abaixo) |
| 7 | limiter | look-ahead, detector com oversampling de 4×, −1,2 dBFS | chega a −1 dBTP; ganho de saída iterado até o alvo |

**Por que o ajuste por trecho:** o EQ muda trechos com instrumentos diferentes em quantidades diferentes. Em raio-x o final é uma única parte de médios, sem grave. O corte em 80 Hz não o abaixou e o realce de presença o levantou inteiro, então ele ficou 2,3 dB mais alto que o resto. O ajuste devolve cada trecho para cerca de 0,6 dB do nível que tinha na mixagem.

**Alvo de volume:** −13 LUFS por padrão. Os outros masters feitos em nov. 2026 ficam entre −12 e −13,5. Depois de remodelar a bateria, não suba além do que o limiter aguenta em silêncio: um limiter trabalhando muito cancela o ataque extra (Katz, *Mastering Audio*, cap. 11, "Compromises When Making Hot Masters"). Owsinski sugere no máximo 2–3 dB de limitação e teto de −1 dB para distribuição online (*The Mastering Engineer's Handbook*, "Setting The Limiter").

**raio-x a −13 LUFS** (tudo com volume igualado):
- **Limiter:** 0,05 dB nos 1% de momentos mais altos; 6 dB em alguns picos isolados de milissegundos.
- **Grave e presença:** 80 Hz −2,7 dB, 2,5–5 kHz +2,3 a +2,5 dB.
- **Dinâmica:** PLR de 11,8 dB, faixa de loudness de 2,4 LU (a mixagem tinha 2,5).
- **Bateria:** subida do ataque de 10,2 → 10,5 dB, ou seja, o punch sobreviveu.
- **Voz:** 0,8 LU mais atrás em relação aos instrumentos, porque o brilho favorece o chimbal e as guitarras.

## 6. Conferindo o resultado (`check.py`)

Compare sempre com o volume igualado: a versão mais alta vence qualquer A/B rápido. `master.py --ab-level -15` grava as duas a −15 LUFS. Escolha um nível abaixo do da própria mixagem se ela passar de 0 dBTP, senão a cópia vai clipar em MP3.

O `check.py` mostra:
- **Mudança de timbre** por banda, com volume igualado.
- **Equilíbrio dos trechos:** quanto cada janela de 3 s se afastou da mixagem.
- **Perda em mono** em 500 Hz–4 kHz. Preocupe-se a partir de −3 dB.
- **PLR.**
- **Subida do ataque da bateria**, com `--drums`.
- **Quanto o EQ do master move a voz** em relação ao resto, com `--vocals`.

## 7. Armadilhas já encontradas

- **Os stems não somam a mixagem se o nível deles não for preservado.** O `separate.py` entrega a entrada ao audio-separator a 0,5 de pico com `normalization_threshold=1.0` e depois divide os stems de volta (o mesmo truque de `neural-upscaler/upscaler.py::separate_vocals`).
- **O audio-separator grava em 44,1 kHz.** Se a fonte é 48 kHz, reamostre antes de subtrair ou somar. Taxas diferentes fizeram uma divisão de bateria parecer 2,8 dB *mais alta* que a mixagem de onde veio.
- **Os filtros precisam começar assentados.** Um suavizador de um polo que começa do zero faz um fade-in nos primeiros milissegundos, e o teto de pico leu isso como 33 dB de redução de ganho. O `dsp.env` inicia seu estado com a primeira amostra.
- **O DrumSep não tem saída de "percussão".** Shakers e pandeiros caem em `hh`, e percussão grave costuma cair em `toms`. Em raio-x, "toms" foi a peça mais alta.
- **Um chiado constante tipo shaker costuma ser uma parte do arranjo, não ruído.** Veja se ele segue o tempo e para junto com a bateria antes de tratá-lo como ruído.
