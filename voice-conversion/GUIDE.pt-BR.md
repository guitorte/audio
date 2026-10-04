# 🎤 Conversão de voz: o guia completo

*Escrito para ser lido do zero, meses depois. Comece pelo início: cada passo diz exatamente onde clicar e o que digitar.*

🇬🇧 [English version](GUIDE.md)

---

## Sumário

1. [O que isto faz](#1-o-que-isto-faz)
2. [Por onde eu começo?](#2-por-onde-eu-começo)
3. [Configuração única no Kaggle](#3-configuração-única-no-kaggle)
4. [Como escolher arquivos em qualquer célula](#4-como-escolher-arquivos-em-qualquer-célula)
5. [Receita A: conversão rápida com Seed-VC (sem treino)](#5-receita-a-conversão-rápida-com-seed-vc-sem-treino)
6. [Receita B: treinar sua própria voz RVC](#6-receita-b-treinar-sua-própria-voz-rvc)
7. [Receita C: converter músicas com sua voz RVC](#7-receita-c-converter-músicas-com-sua-voz-rvc)
8. [Receita D: verificação de oitava (opcional)](#8-receita-d-verificação-de-oitava-opcional)
9. [Como pegar seus resultados](#9-como-pegar-seus-resultados)
10. [Todas as opções explicadas](#10-todas-as-opções-explicadas)
11. [Usando o Google Colab](#11-usando-o-google-colab)
12. [Solução de problemas](#12-solução-de-problemas)
13. [Como funciona por dentro](#13-como-funciona-por-dentro)

---

## 1. O que isto faz

Ele **troca o cantor** de uma música por outra voz. Todo o resto da música fica como está.

```
sua música ──► separação ──┬── voz ────────────► nova voz ──► mesmo volume da voz original ──┐
                           └── instrumental (intacto) ───────────────────────────────────────┴──► música pronta
```

Você também recebe a nova voz sozinha, para mixar você mesmo numa DAW.

Há **dois motores** (duas formas diferentes de gerar a nova voz):

| | **Seed-VC** | **RVC** |
|---|---|---|
| O que você fornece | **um trecho curto** (10–30 s) da voz desejada | um **modelo de voz treinado** (arquivo `.pth`) |
| Treino | **nenhum**: funciona na hora | você treina uma vez com ~10+ minutos de gravações (1–3 h de GPU) |
| Melhor para | experimentar uma voz rapidamente | o resultado mais convincente, para uma voz que você vai usar muitas vezes |

> ⚖️ Converta apenas vozes que você tem direito de usar, como a sua ou a de alguém que concordou.

---

## 2. Por onde eu começo?

**Plataforma: use o Kaggle.** É gratuito e permite rodar isto. O plano *gratuito* do Google Colab já desconectou este notebook no meio da execução, como "uso não permitido". O Colab só funciona com unidades de computação pagas (veja a [seção 11](#11-usando-o-google-colab)).

**Depois:**

| Eu quero… | Faça isto |
|---|---|
| Ouvir uma música com outra voz, agora | [Receita A: Seed-VC](#5-receita-a-conversão-rápida-com-seed-vc-sem-treino) |
| Criar uma voz reutilizável e de alta qualidade | [Receita B: treinar RVC](#6-receita-b-treinar-sua-própria-voz-rvc), depois a [Receita C](#7-receita-c-converter-músicas-com-sua-voz-rvc) |
| Já tenho um modelo RVC (`.pth`) | [Receita C](#7-receita-c-converter-músicas-com-sua-voz-rvc) |
| Saber se preciso mudar de oitava | [Receita D](#8-receita-d-verificação-de-oitava-opcional) |

---

## 3. Configuração única no Kaggle

Você faz isto uma vez só. Depois, toda sessão começa no passo 3.5.

### 3.1 Conta no Kaggle
- Crie uma conta em [kaggle.com](https://www.kaggle.com).
- **Verifique seu número de celular** (*Settings ▸ Phone verification*). Sem isso, o notebook não pode usar a internet, e ele precisa dela para baixar os modelos de IA.

### 3.2 Coloque seus áudios num Dataset do Kaggle
**Dataset** é o nome que o Kaggle dá a uma pasta de arquivos que você envia.

1. *Create* (o botão **+**) ▸ **New Dataset**.
2. Arraste seus arquivos. **Deixe tudo no nível principal**, sem precisar de pastas:
   ```
   minhas-musicas/
   ├── relaxa.mp3            ← músicas que você quer converter
   ├── displicente.wav       ← um trecho de voz para o Seed-VC
   ├── maria take 1.wav      ← gravações para treinar uma voz RVC
   ├── maria take 2.wav
   └── joão.pth              ← (opcional) um modelo RVC que você já tem, mais joão.index se tiver
   ```
3. Dê um nome (ex.: `minhas-musicas`) e clique em **Create**. Mantenha **Private**.

**Dicas sobre os arquivos:**
- Qualquer formato comum funciona: mp3, wav, flac, m4a, ogg…
- **Stems de voz limpos e secos** (sem reverb, sem instrumentos) dão os melhores resultados, tanto como trecho de referência quanto para treino. Músicas completas também funcionam, porque o notebook consegue isolar a voz para você.
- **Os nomes importam um pouco:** você vai escolher arquivos pelo número ou por parte do nome (veja a [seção 4](#4-como-escolher-arquivos-em-qualquer-célula)). Um começo de nome em comum, como `maria take …`, facilita selecionar o conjunto de treino.

**Para adicionar arquivos depois:** abra o dataset ▸ **New Version** e adicione os arquivos. Um notebook só enxerga os arquivos novos depois que você reconecta o dataset ou reinicia a sessão.

### 3.3 Leve o notebook para o Kaggle
1. Abra este arquivo no GitHub: `voice-conversion/Voice_Conversion_Kaggle.ipynb`, no repositório `guitorte/audio`.
2. Baixe o arquivo (botão **Download raw file**).
3. No Kaggle: *Create* ▸ **New Notebook** ▸ **File ▸ Import Notebook** ▸ envie o arquivo.

> Quando este guia ou o código mudarem, repita este passo para pegar a versão nova. O notebook baixa o resto do código a cada sessão, mas o arquivo do notebook em si só se atualiza quando você o importa de novo.

### 3.4 Configurações do notebook (painel da direita)
- **Accelerator ▸ GPU T4 x2.** *Não* a P100: o RVC não funciona nela.
- **Internet ▸ On.**

### 3.5 Conecte seu dataset (em todo notebook novo)
- **Add Input** (painel da direita) ▸ *Your Datasets* ▸ escolha `minhas-musicas`.

Pronto. Vá para a receita que você precisa.

---

## 4. Como escolher arquivos em qualquer célula

A célula 1 lista todos os arquivos de áudio com um número:
```
📁 /kaggle/input/minhas-musicas  (5 songs)
  1. displicente.wav
  2. maria take 1.wav
  3. maria take 2.wav
  4. relaxa.mp3
  5. vida.mp3
```

Toda opção que pede arquivos (`SONGS`, `REFERENCE`, `DATASET`, `PITCH_REFERENCE`, `SOURCE`, `TARGET`) aceita:

| Você digita | Significa |
|---|---|
| `"4"` | o arquivo número 4 |
| `"4, 5"` ou `"2-3"` | vários arquivos, ou um intervalo |
| `"relaxa"` | o arquivo cujo nome corresponde |
| `"maria take"` | todos os arquivos que começam assim (aqui, os takes 1 e 2) |
| `"all"` | todos os arquivos (só em `SONGS` e `DATASET`) |
| `"relaxa; vida"` | vários nomes (separe com `;` ou `,`) |

**Regras de correspondência:** um nome exato vence; depois, nomes que *começam* com o que você digitou; depois, nomes que *contêm*. Então `"male"` escolhe `male vocal.wav`, não `female vocal.wav`. Maiúsculas e acentos não precisam bater exatamente. Se o que você digitou corresponder a mais de um arquivo onde só um é permitido, aparece um erro dizendo quais arquivos ele encontrou. É só ser mais específico.

---

## 5. Receita A: conversão rápida com Seed-VC (sem treino)

**Você precisa de:** uma música e um trecho da voz desejada, ambos no seu dataset.

1. Rode a **célula 1** (o botão ▶ da célula, ou *Shift+Enter*). Confira se seus arquivos aparecem.
   - A **primeira** execução de cada sessão leva alguns minutos, porque instala coisas. É normal.
2. Vá para a **célula 2** e defina:
   ```python
   RUN = True
   SONGS = "relaxa"            # a(s) música(s) a converter
   REFERENCE = "displicente"   # o trecho de voz
   SEMITONES = 0               # veja "tom" abaixo
   ```
   Se o arquivo da música **já for um stem de voz limpo**, defina também `SEPARATE_VOCALS = False`. Se o trecho de voz for um stem limpo, defina `ISOLATE_REFERENCE_VOCAL = False`.
3. Rode a célula 2. Você verá linhas de instalação, depois `🎧 relaxa.mp3`, depois uma barra de progresso e então `✅ …/converted/relaxa__displicente__seedvc.wav`.
4. Rode a **célula 5** para ouvir: a original, a música convertida e a nova voz sozinha.
5. Pegue os arquivos: veja a [seção 9](#9-como-pegar-seus-resultados).

**Tom:** se a nova voz soar forçada, ou como esquilo, os dois cantores ficam em regiões diferentes. Uma música cantada por homem e convertida para voz feminina normalmente precisa de `SEMITONES = 12` (uma oitava acima); o contrário precisa de `-12`. Se não tiver certeza, defina `AUTO_OCTAVE = True` e ele decide por você (a [Receita D](#8-receita-d-verificação-de-oitava-opcional) explica como).

---

## 6. Receita B: treinar sua própria voz RVC

**Você precisa de:** uns **10 minutos ou mais** de uma pessoa cantando, em quantos arquivos quiser, no seu dataset.

**Antes de começar, as gravações devem ter:**
- **um cantor só**, sem backing vocals ou harmonias;
- som **seco** (sem reverb ou eco) e sem instrumentos. Músicas completas funcionam se você deixar `ISOLATE_VOCALS = True`, mas stems limpos são melhores;
- qualquer duração cada uma. Silêncio não atrapalha, porque o treino corta o áudio em pedaços curtos.

### Passos
1. Conecte seu dataset (3.5) e rode a **célula 1**. **Anote os números das suas gravações** (ex.: 2, 3, 6, 7).
2. Na **célula 4**, defina:
   ```python
   RUN = True
   DATASET = "2, 3, 6, 7"   # suas gravações: números, ou um trecho de nome em comum ("maria take")
   MODEL_NAME = "maria"     # o nome da voz. Obrigatório quando você escolhe por número
   ISOLATE_VOCALS = False   # False se as gravações forem stems limpos; True se forem músicas completas
   EPOCHS = 200             # quanto tempo treinar; 200 é um bom começo para 10+ minutos
   SAVE_EVERY = 10          # deixe como está
   BATCH_SIZE = 8           # deixe como está
   SAMPLE_RATE = 40000      # deixe como está
   ```
3. Confira se todas as outras células continuam com `RUN = False`.
4. Comece o treino **em segundo plano**: clique em **Save Version** (canto superior direito) ▸ **Save & Run All (Commit)** ▸ **Save**.
   - Assim ele continua rodando com o navegador fechado, por até ~12 horas. Uma sessão interativa normal pararia quando ficasse parada, então treine sempre assim.
   - Espere algo em torno de **1–3 horas** para 200 épocas. Isso consome sua cota semanal de GPU (umas 30 h).
5. **Para acompanhar (opcional):** abra a página do notebook ▸ *Versions* ▸ a versão em execução ▸ *Logs*. Você verá linhas como:
   ```
   maria | epoch=57 | step=1425 | ...
   ```
6. Quando terminar, a aba **Output** da versão terá:
   ```
   voices/maria.pth      ← sua voz
   voices/maria.index    ← deixa a voz mais fiel
   voices/.training/…    ← checkpoints, para continuar o treino depois
   ```
   Pronto. Vá para a [Receita C](#7-receita-c-converter-músicas-com-sua-voz-rvc) para usá-la.

### Treinar de novo ou por mais tempo
**Se parou antes do fim** (limite de tempo do Kaggle, um erro) **ou se a voz ainda não convence:**
1. Abra o notebook ▸ **Add Input** ▸ **Notebook Output** ▸ escolha a versão do treino.
2. Na célula 4, mantenha o **mesmo `MODEL_NAME`** e coloque em `EPOCHS` o novo **total** (ex.: `300`).
3. **Save & Run All** de novo. Ele mostra `resuming from checkpoints …` e continua de onde parou, não do zero.

**Quantas épocas?**

| O que você ouve | O que fazer |
|---|---|
| Parece uma mistura, não exatamente o cantor | treine mais: continue até 300–400 |
| Artefatos metálicos, robóticos, chiados | treinou demais: da próxima vez, pare antes (ex.: 150) |
| Está bom | pare: terminou |

---

## 7. Receita C: converter músicas com sua voz RVC

1. **Se a voz foi treinada numa execução anterior**, conecte-a: **Add Input** ▸ **Notebook Output** ▸ a versão do treino. (Um `.pth` que está no seu dataset não precisa de nada extra.)
2. Rode a **célula 1**. Sua voz aparece em **RVC models**:
   ```
   RVC models:
      • maria
   ```
3. Na **célula 3**, defina:
   ```python
   RUN = True
   SONGS = "relaxa"
   MODEL = "maria"
   PITCH = 0
   ```
   Opcional, para a verificação automática de oitava: `AUTO_OCTAVE = True` e `PITCH_REFERENCE = "maria take 1"` (qualquer trecho da voz treinada).
   Se a música já for um stem de voz limpo: `SEPARATE_VOCALS = False`.
4. Rode a célula 3, depois a **célula 5** para ouvir.

---

## 8. Receita D: verificação de oitava (opcional)

A conversão de voz mantém a melodia **na altura em que foi cantada**. Se uma voz masculina grave cantou a música e a nova voz é feminina e aguda, ela cantaria uma oitava abaixo da sua região natural, e isso soa errado. A verificação de oitava mede as duas vozes e diz se é preciso mudar em oitavas inteiras.

**Só verificar** (nada é convertido) com a **célula 7** (célula 6 no Colab):
```python
RUN = True
SOURCE = "relaxa"        # a voz que você vai converter
TARGET = "displicente"   # um trecho da voz desejada
ISOLATE = False          # True só se forem músicas completas, não stems de voz
```
Ela mostra, por exemplo:
```
   source vocal: median E3 (166 Hz), range C3–A3
   target voice: median E4 (331 Hz), range B3–G#4
   gap +12.0 semitones → +12 semitones (1 octave up)
```
Ou seja: voz de origem com mediana em Mi3, voz desejada em Mi4, diferença de +12 semitons, uma oitava acima. Depois coloque esse número em `SEMITONES` (Seed-VC) ou `PITCH` (RVC).

**Ou deixe aplicar automaticamente:** defina `AUTO_OCTAVE = True` na célula 2 ou 3. O ajuste é somado ao `SEMITONES`/`PITCH` que você definiu, separadamente para cada música.

**Como ele decide:** se as duas vozes estão a menos de meia oitava (6 semitons) uma da outra, ele não muda nada. Acima disso, arredonda para a oitava inteira mais próxima. Perto de exatamente meia oitava, ele mostra **⚠️ borderline** (no limite). As duas escolhas podem funcionar, então ouça as duas.

---

## 9. Como pegar seus resultados

Tudo é gravado em **`/kaggle/working/`** (no Colab: dentro da sua pasta no Drive):

| Arquivo | O que é |
|---|---|
| `converted/<música>__<voz>__<motor>.wav` | a música pronta (WAV 24 bits) |
| `converted/<música>__<voz>__<motor>_vocals.wav` | só a nova voz, com a mesma duração da música, pronta para uma DAW |
| `converted/<…>.json` | as configurações usadas |
| `voices/<nome>.pth` + `.index` | vozes que você treinou |
| `results.zip` | tudo de `converted/`, compactado (feito pela célula 6) |

⚠️ **O Kaggle apaga `/kaggle/working` quando a sessão termina**, a não ser que você:
- **baixe**: rode a **célula 6** e, no painel da direita, abra **Output** e baixe o `results.zip`; ou
- **salve uma versão**: *Save Version ▸ Save & Run All*. Os arquivos ficam na aba **Output** daquela versão e podem ser conectados em sessões futuras.

Já converteu uma música com a mesma voz? Ela é pulada (`already exists`). Defina `OVERWRITE = True` para refazer.

---

## 10. Todas as opções explicadas

**Célula 1**
| Opção | Padrão | Significado |
|---|---|---|
| `INPUT` | `""` | só no Kaggle. Deixe vazio para achar seu dataset automaticamente. Se conectou vários datasets, digite o nome da pasta, ex.: `"minhas-musicas"` |
| `DRIVE_FOLDER` | `"áudio"` | só no Colab. Sua pasta dentro de *Meu Drive* |

**Célula 2 · Seed-VC**
| Opção | Padrão | Significado / quando mudar |
|---|---|---|
| `RUN` | `False` | só no Kaggle. Coloque `True` para rodar esta célula |
| `SONGS` | `"1"` | músicas a converter ([seção 4](#4-como-escolher-arquivos-em-qualquer-célula)) |
| `REFERENCE` | | o trecho da voz desejada |
| `SEMITONES` | `0` | mudança de tom. `12` = uma oitava acima, `-12` = uma oitava abaixo |
| `DIFFUSION_STEPS` | `30` | qualidade x velocidade. 30–50 para canto. Mais é mais lento |
| `AUTO_OCTAVE` | `False` | mede as duas vozes e soma uma mudança de oitava se precisar ([Receita D](#8-receita-d-verificação-de-oitava-opcional)) |
| `SINGING_MODEL` | `True` | `False` para fala em vez de canto |
| `SEPARATE_VOCALS` | `True` | `False` se as músicas já forem stems a cappella |
| `ISOLATE_REFERENCE_VOCAL` | `True` | `False` se o trecho de voz já for um stem limpo |
| `VOCAL_GAIN_DB` | `0.0` | deixa a nova voz mais alta (+) ou mais baixa (−) na mixagem |
| `OVERWRITE` | `False` | refaz músicas já convertidas |

**Célula 3 · conversão RVC**
| Opção | Padrão | Significado / quando mudar |
|---|---|---|
| `MODEL` | | o nome da voz (o nome do arquivo `.pth`, sem o `.pth`) |
| `PITCH` | `0` | igual ao `SEMITONES` acima |
| `INDEX_RATE` | `0.5` | 0–1. Quanto puxar o timbre para a voz treinada (precisa do `.index`). Mais alto fica mais parecido com a voz, mas às vezes menos nítido |
| `PROTECT` | `0.33` | 0–0,5. Protege respirações e consoantes de artefatos. Mais baixo protege mais |
| `AUTO_OCTAVE` + `PITCH_REFERENCE` | desligado | verificação automática de oitava contra um trecho da voz treinada |
| `SONGS`, `SEPARATE_VOCALS`, `VOCAL_GAIN_DB`, `OVERWRITE` | | iguais à célula 2 |

**Célula 4 · treino RVC**
| Opção | Padrão | Significado / quando mudar |
|---|---|---|
| `DATASET` | | as gravações: números ou um trecho de nome em comum |
| `MODEL_NAME` | | o nome da voz. Obrigatório quando `DATASET` são números. Use o mesmo nome para continuar treinando |
| `ISOLATE_VOCALS` | `True` | `False` para stems de voz limpos |
| `EPOCHS` | `200` | duração total do treino ([quantas?](#treinar-de-novo-ou-por-mais-tempo)) |
| `SAVE_EVERY` | `10` | a cada quantas épocas salvar um checkpoint. Deixe como está |
| `BATCH_SIZE` | `8` | deixe como está. Baixe para 4 só se der erro de "out of memory" |
| `SAMPLE_RATE` | `40000` | deixe como está |

**Célula 5 · ouvir:** `START_AT_SECONDS` e `CLIP_SECONDS` escolhem o trecho que toca. Músicas curtas são tratadas automaticamente.

---

## 11. Usando o Google Colab

Existe uma versão para Colab: `Voice_Conversion.ipynb` (há um link "Open in Colab" no README).

- **Só funciona com unidades de computação pagas** (Colab Pro ou pagamento por uso). O plano gratuito a desconecta.
- Seus arquivos ficam no **Google Drive**, na pasta `DRIVE_FOLDER` (padrão `áudio`). Ela pode ser plana como o dataset do Kaggle, ou ter uma subpasta `voices/` para trechos, modelos e pastas de treino.
- As opções são campos de formulário (controles deslizantes e caixas), e não há chaves `RUN`. Cada célula roda quando você clica nela.
- Os resultados vão direto para a sua pasta no Drive (`converted/`, `voices/`), então nada se perde quando a sessão termina.
- Os checkpoints do treino são copiados para `voices/.training/` a cada poucos minutos. Se o Colab desconectar, rode a célula 4 de novo com as mesmas opções e ele continua.
- **Depois que o código for atualizado:** *Ambiente de execução ▸ Desconectar e excluir ambiente de execução* (Runtime ▸ Disconnect and delete runtime) antes de rodar de novo, senão o Colab continua usando o código antigo.

---

## 12. Solução de problemas

**Onde olhar primeiro:** as linhas impressas **acima** de um erro são a explicação de verdade. O traceback vermelho no fim só mostra onde parou.

| O que aparece | Por quê | Solução |
|---|---|---|
| Colab: *"Ambiente de execução desconectado… código não permitido no nível sem custo"* | o plano gratuito do Colab bloqueia este uso | use o Kaggle, ou unidades pagas do Colab |
| `No folder with songs found under /kaggle/input` | o dataset não está conectado | **Add Input** ▸ seu dataset |
| `Several folders could hold your songs` | há vários datasets conectados | defina `INPUT = "minhas-musicas"` na célula 1 |
| `… not found. Available: …` / `matches no clip … and no song` | erro de digitação, ou o arquivo não está no dataset | use o nome ou número exato da lista da célula 1 |
| `… is ambiguous: a, b` | o texto corresponde a vários arquivos | digite mais do nome, ou use o número |
| `Set MODEL_NAME` | você escolheu os arquivos de treino por número | dê um nome à voz em `MODEL_NAME` |
| `AUTO_OCTAVE with RVC needs PITCH_REFERENCE` | modelos RVC não contêm áudio para medir | coloque em `PITCH_REFERENCE` um trecho daquela voz |
| `⚠️ No GPU` / `⚠️ P100` | acelerador errado | Settings ▸ **GPU T4 x2** |
| erros ao clonar ou baixar (git, pip, huggingface) | a internet está desligada | Settings ▸ **Internet On** (precisa do celular verificado) |
| minha voz treinada não aparece na célula 1 | a saída do treino não está conectada | **Add Input** ▸ **Notebook Output** ▸ aquela versão |
| o treino parou antes de terminar | limite de ~12 h do Kaggle, ou um erro | conecte aquela saída e rode a célula 4 de novo com o mesmo `MODEL_NAME` ([detalhes](#treinar-de-novo-ou-por-mais-tempo)) |
| `CUDA out of memory` no treino | o lote não cabe na memória da GPU | `BATCH_SIZE = 4` |
| a voz soa forçada, ou como esquilo ou gigante | os cantores ficam em regiões diferentes | `SEMITONES`/`PITCH` ±12, ou `AUTO_OCTAVE = True` |
| ainda dá para ouvir os backing vocals do cantor original | a separação só pega a voz principal | esperado. Use músicas com voz principal clara, ou seus próprios stems |
| voz RVC metálica ou robótica | treinou demais, ou `INDEX_RATE` alto demais | menos épocas da próxima vez, ou `INDEX_RATE = 0.3` |
| `already exists (tick OVERWRITE to redo)` | aquela música com aquela voz já foi feita | `OVERWRITE = True` |
| a primeira execução demora muito, com várias linhas de instalação | cada motor monta seu próprio ambiente e baixa modelos no primeiro uso | normal: alguns minutos por motor, uma vez por sessão |
| `Warning: Skipped loading some keys due to shape mismatch` (Seed-VC) | vem dos próprios arquivos de modelo do Seed-VC | inofensivo, até onde sabemos. Julgue pelo ouvido |
| um bug antigo voltou, ou faltam opções novas | você está usando uma cópia antiga do notebook | Kaggle: importe o notebook de novo (3.3). Colab: *Desconectar e excluir ambiente de execução* |

---

## 13. Como funciona por dentro

*Para quando algo quebrar e você, ou o Claude, precisarem investigar. Os detalhes para desenvolvedores estão em `CLAUDE.md`, na raiz do repositório.*

- **Arquivos:**
  - `vc.py`: tudo o que os notebooks chamam (instalação, busca de arquivos, conversão, treino).
  - `pitch.py`: a verificação de oitava.
  - Ele reaproveita `../neural-upscaler/upscaler.py` para a separação da voz (BS-RoFormer), a seleção de arquivos, utilidades de áudio e a execução de programas com a saída visível.
- **Os notebooks são finos:** a cada sessão eles baixam este repositório (branch `ccr-8be874fb-rdbsnf`, ou o branch padrão se esse não existir mais) e chamam o `vc.py`.
- **Ambientes isolados:** o Seed-VC e o Applio (o kit de ferramentas do RVC) precisam de versões antigas e conflitantes de bibliotecas. Cada um roda a partir de uma cópia fixa do seu código-fonte, no seu próprio ambiente Python, montado automaticamente no primeiro uso:
  - Seed-VC: Python 3.10 + torch 2.4;
  - Applio: Python 3.12 + torch 2.11 para CUDA 12.8.
- **Arquivos temporários:** ficam em `/tmp` (Kaggle) ou `/content` (Colab) e somem com a sessão.
- **Volume:** os dois motores normalizam a saída, então a nova voz é ajustada de volta para o volume da voz original antes da mixagem.
- **Rede de segurança do treino:**
  - Os checkpoints são espelhados em `voices/.training/<nome>/` durante o treino.
  - Uma nova execução procura por eles na pasta de saída e nas entradas conectadas, e então continua.
  - Só ficam guardados o arquivo de voz mais recente e o último checkpoint completo. Cada conjunto completo tem uns 1,3 GB.
