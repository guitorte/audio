# 🎚️ Neural Audio Upscaler: o guia completo

*Escrito para ser lido do zero, meses depois. Comece pelo início: cada passo diz exatamente onde clicar e o que digitar.*

🇬🇧 [English version](GUIDE.md)

---

## Sumário

1. [O que isto faz](#1-o-que-isto-faz)
2. [O que cada etapa corrige](#2-o-que-cada-etapa-corrige)
3. [Configuração única](#3-configuração-única)
4. [Como escolher arquivos](#4-como-escolher-arquivos)
5. [Receita: restaurar suas músicas](#5-receita-restaurar-suas-músicas)
6. [Ouvindo e avaliando o resultado](#6-ouvindo-e-avaliando-o-resultado)
7. [Pegando seus resultados](#7-pegando-seus-resultados)
8. [Todas as opções explicadas](#8-todas-as-opções-explicadas)
9. [Solução de problemas](#9-solução-de-problemas)
10. [Como funciona por dentro](#10-como-funciona-por-dentro)

---

## 1. O que isto faz

Ele **conserta o som** de músicas que você já tem: os danos da compressão MP3/AAC e os agudos que faltam. Ele **não** recria nem gera a música de novo. É a mesma gravação, limpa.

```
musica.mp3 ─► ① separa a voz ─┬─ voz ──────────► ② Apollo ─┐
                              └─ instrumental ─► ② Apollo ─┴─► ③ AudioSR ─► ④ Matchering ─► musica_upscaled.wav
```

Ele roda no **Google Colab**, lendo as músicas de uma pasta no seu **Google Drive** e gravando os resultados de volta lá.

---

## 2. O que cada etapa corrige

| Etapa | Corrige | Padrão | Custo |
|---|---|---|---|
| ① **Separação da voz** (BS-RoFormer) | permite consertar a voz e o instrumental separadamente, e ajustar o volume da voz. As duas partes sempre somam exatamente o original | **ligada** | um ou dois minutos |
| ② **Apollo** | danos da compressão: pratos "aquáticos" ou rodopiando, bateria borrada, agudos abafados | **ligada** | alguns minutos por música |
| ③ **AudioSR** | **reconstrói os agudos que o MP3 cortou** (ex.: tudo acima de 16 kHz) e gera em 48 kHz | desligada | **lento**: mais de 10 minutos por música |
| ④ **Matchering** | masterização: copia o volume, o timbre e a largura estéreo de uma **música de referência** que você escolher | desligada | cerca de um minuto |

**O AudioSR só acrescenta.** Ele descobre onde os agudos do seu arquivo foram cortados, mantém tudo abaixo disso exatamente como era e preenche só o que falta acima. Se uma música não tem agudos faltando (um master sem perdas, por exemplo), o AudioSR pula essa música sozinho.

**Quais etapas usar?**
- **Primeira tentativa:** o padrão (separação + Apollo). Ouça.
- **Ainda soa opaca ou abafada?** Ligue o AudioSR para aquela música.
- **Baixa demais perto de outras músicas, ou com o timbre estranho?** Use o Matchering com uma música de referência de que você goste.

---

## 3. Configuração única

### 3.1 Coloque suas músicas no Google Drive
- Crie uma pasta em *Meu Drive*, por exemplo **`áudio`** (esse é o nome padrão; qualquer nome funciona).
- Coloque as músicas nela. Qualquer formato comum funciona: mp3, wav, flac, m4a, ogg…
- Os resultados aparecem numa subpasta nova, `upscaled/`, dentro da mesma pasta. Seus arquivos originais nunca são alterados.

### 3.2 Abra o notebook
- Abra `neural-upscaler/Neural_Audio_Upscaler.ipynb` no repositório `guitorte/audio` do GitHub e clique em **Open in Colab** no topo, ou use o link do README.
- *Arquivo ▸ Salvar uma cópia no Drive* (File ▸ Save a copy in Drive) se quiser guardar uma cópia sua com as configurações.

### 3.3 Escolha uma GPU
- **Ambiente de execução ▸ Alterar o tipo de ambiente de execução ▸ GPU T4** (Runtime ▸ Change runtime type ▸ T4 GPU). Uma **L4** é mais rápida, se o seu plano tiver.

> **Plano do Colab:** nas nossas execuções, o upscaler funcionou no plano gratuito do Colab, mas o Google decide o que sessões gratuitas podem rodar e pode mudar isso. Se uma execução for interrompida com uma mensagem sobre o "nível sem custo", você vai precisar de unidades de computação pagas. (Ainda não existe versão do upscaler para o Kaggle; peça se precisar.)

---

## 4. Como escolher arquivos

A célula 1 lista suas músicas com números:
```
📁 /content/drive/MyDrive/áudio  (3 songs)
  1. know better.mp3  (5.1 MB)
  2. relaxa.mp3  (7.8 MB)
  3. vida.flac  (31.2 MB)
```

`SONGS` e `MASTER_REFERENCE` aceitam:

| Você digita | Significa |
|---|---|
| `"all"` | todas as músicas (só em `SONGS`) |
| `"2"`, `"1, 3"`, `"1-3"` | números da lista |
| `"relaxa"` | a música cujo nome corresponde |
| `"know; vida"` | vários nomes, separados por `;` ou `,` |

Um nome exato vence; depois, nomes que *começam* com o que você digitou; depois, nomes que *contêm*. Maiúsculas e acentos não precisam bater.

---

## 5. Receita: restaurar suas músicas

1. **Rode a célula 1** (o botão ▶, ou *Shift+Enter*).
   - O Google pede permissão para acessar seu Drive. Permita.
   - Antes, ajuste `DRIVE_FOLDER` se sua pasta não se chamar `áudio`. Uma subpasta se escreve como `"Musicas/áudio"`.
   - Confira se suas músicas aparecem e se a última linha mostra uma **GPU** (e não "⚠️ none").
2. **Célula 2: escolha e rode.** O padrão é uma boa primeira passada:
   ```
   SONGS = "all"            ← ou ex.: "2" para uma música só
   SEPARATE_VOCALS = ✔
   APOLLO = ✔
   AUDIOSR = ☐              ← marque para músicas opacas ou abafadas (lento)
   MASTER_REFERENCE = ""    ← opcional: uma música para copiar o som
   ```
   Rode. Para cada música, você verá:
   ```
   🎧 relaxa.mp3
      input bandwidth ≈ 16.0 kHz (lossy shelf detected)
      ① separating vocals (BS-RoFormer)…
      ② Apollo restoring vocals…
      ② Apollo restoring instrumental…
   ✅ /content/drive/MyDrive/áudio/upscaled/relaxa_upscaled.wav
   ```
   A primeira linha diz que os agudos do arquivo foram cortados por volta de 16 kHz. A **primeira** execução de cada sessão passa alguns minutos instalando coisas. É normal.
3. **Célula 3: compare.** Veja a [seção 6](#6-ouvindo-e-avaliando-o-resultado).
4. **Quer mais?** Rode a célula 2 de novo com `AUDIOSR` marcado (e `OVERWRITE` marcado, para refazer as músicas que já estão prontas).

---

## 6. Ouvindo e avaliando o resultado

A célula 3 toca um trecho de 30 segundos **antes** e **depois**, no **mesmo volume**. O mais alto sempre parece "melhor", então a comparação só vale com volumes iguais. Ela também desenha dois **espectrogramas**, imagens das frequências ao longo do tempo:

- **Um MP3 tem um "teto" reto**: uma linha horizontal nítida, muitas vezes por volta de 16 kHz, sem nada acima.
- **Depois do AudioSR**, esse espaço acima da linha fica preenchido.
- **Depois do Apollo**, pratos e consoantes ficam mais nítidos e menos borrados.

**O que ouvir:** pratos e chimbal (menos rodopio), os "s" da voz (menos chiado ou língua presa) e o "ar" ou abertura nos agudos.
**Sinais de alerta:** chiado ou brilho metálico nos agudos (o AudioSR inventando demais; fique com a versão só com Apollo), ou volume "bombeando", sinal de que a referência de masterização não combinou.

Ajuste `START_AT_SECONDS` para pular para a parte que importa, como o refrão.

---

## 7. Pegando seus resultados

Tudo vai direto para o seu Drive, em `upscaled/`, ao lado das suas músicas:

| Arquivo | O que é |
|---|---|
| `upscaled/<música>_upscaled.wav` | a música restaurada (WAV 24 bits; 48 kHz depois do AudioSR, senão 44,1 kHz) |
| `upscaled/<música>_upscaled.json` | as configurações usadas, mais observações (banda detectada, ajustes de volume…) |
| `upscaled/<música>_parts/vocals.wav` e `instrumental.wav` | os stems restaurados, se você marcou `KEEP_PARTS` |

- **As observações abaixo de cada resultado** explicam o que aconteceu, por exemplo:
  - `input bandwidth ≈ 16.0 kHz`: os agudos do arquivo de entrada vão até cerca de 16 kHz;
  - `AudioSR: new highs above 15.7 kHz`: o AudioSR criou agudos novos acima de 15,7 kHz;
  - `lowered 2.1 dB to avoid clipping`: o volume foi baixado 2,1 dB para não distorcer.
- O "lowered … dB" é normal. MP3s decodificados costumam passar um pouco do máximo, então a música é baixada só o suficiente para não distorcer.
- Uma música já pronta é pulada (`already exists`). Marque `OVERWRITE` para refazer.
- O Matchering gera em 44,1 kHz.

---

## 8. Todas as opções explicadas

**Célula 1**
| Opção | Padrão | Significado |
|---|---|---|
| `DRIVE_FOLDER` | `"áudio"` | sua pasta dentro de *Meu Drive*. Subpastas como `"Musicas/áudio"` funcionam |

**Célula 2**
| Opção | Padrão | Significado / quando mudar |
|---|---|---|
| `SONGS` | `"all"` | quais músicas ([seção 4](#4-como-escolher-arquivos)) |
| `SEPARATE_VOCALS` | ✔ | conserta a voz e o instrumental separadamente. Desmarque para música instrumental, ou para ganhar tempo |
| `APOLLO` | ✔ | conserta os danos da compressão. A principal melhoria para MP3s |
| `AUDIOSR` | ☐ | reconstrói os agudos que faltam. Lento; use em músicas opacas |
| `AUDIOSR_STEPS` | `50` | qualidade x velocidade do AudioSR. 50 está bom; mais é mais lento e raramente melhor |
| `CUTOFF_HZ` | `0` | `0` = descobre sozinho onde os agudos foram cortados. Coloque ex.: `16000` para forçar o AudioSR a reconstruir tudo acima dessa frequência (para músicas que ele pularia) |
| `VOCAL_GAIN_DB` | `0.0` | deixa a voz mais alta (+) ou mais baixa (−). Precisa de `SEPARATE_VOCALS` |
| `MASTER_REFERENCE` | `""` | uma música da lista cujo volume e timbre serão copiados. Vazio = sem masterização. Escolha uma música bem masterizada, de estilo parecido |
| `KEEP_PARTS` | ☐ | salva também os stems restaurados de voz e instrumental |
| `OVERWRITE` | ☐ | refaz músicas já prontas |

**Célula 3:** `START_AT_SECONDS` e `CLIP_SECONDS` escolhem o trecho. Músicas curtas são tratadas automaticamente.

---

## 9. Solução de problemas

**Onde olhar primeiro:** as linhas impressas **acima** de um erro explicam o que houve. O traceback vermelho no fim só mostra onde parou.

| O que aparece | Por quê | Solução |
|---|---|---|
| `Folder 'áudio' not found in … Folders there: […]` | o nome da pasta é outro, ou ela está dentro de outra pasta | coloque em `DRIVE_FOLDER` um nome daquela lista, ex.: `"Musicas/áudio"` |
| `GPU: ⚠️ none` | nenhuma GPU selecionada | *Ambiente de execução ▸ Alterar o tipo de ambiente de execução ▸ GPU T4*, e rode a célula 1 de novo |
| `No file number 7` / `No file name contains '…'` | erro de digitação, ou número errado | use um nome ou número da lista da célula 1 |
| `already exists (tick OVERWRITE to redo)` | aquela música já foi feita | marque `OVERWRITE` |
| O AudioSR mostra `already full band, AudioSR skipped` | a música não tem agudos faltando | nada a fazer. Para forçar, use `CUTOFF_HZ` |
| chiado ou brilho metálico depois do AudioSR | o AudioSR inventou agudos que não combinam com a música | use a versão sem AudioSR, ou aumente `CUTOFF_HZ` |
| a primeira execução demora muito | instalação das ferramentas, e a primeira preparação do AudioSR com um download de vários GB | normal, uma vez por sessão |
| *"Ambiente de execução desconectado… nível sem custo"* | o plano gratuito do Colab interrompeu a execução | unidades de computação pagas |
| um erro `python failed (exit 1)` com linhas acima | um programa auxiliar falhou; as linhas acima dizem por quê | leia as linhas acima, e mande-as para o Claude se não estiver claro |
| um bug antigo voltou depois de uma atualização | o Colab ainda está rodando o código antigo | *Ambiente de execução ▸ Desconectar e excluir ambiente de execução*, e rode de novo |

---

## 10. Como funciona por dentro

*Para quem precisar investigar algo depois. Os detalhes para desenvolvedores estão em `CLAUDE.md`, na raiz do repositório.*

- **Arquivos:**
  - `upscaler.py`: tudo o que o notebook chama.
  - `audiosr_worker.py`: roda o AudioSR.
  - `upscaler_dsp.py`: pequenos cálculos de áudio (achar o corte de agudos, o crossover, as transições suaves).
- **O notebook é fino:** a cada sessão ele baixa este repositório (branch `ccr-8be874fb-rdbsnf`, ou o branch padrão se esse não existir mais) e chama o `upscaler.py`.
- **O AudioSR precisa de versões antigas de bibliotecas**, então roda num ambiente Python 3.10 próprio, montado automaticamente na primeira vez.
- **O Apollo** roda a partir de uma cópia fixa do seu código-fonte.
- **Separação sem perdas:** o instrumental é calculado como *música menos voz*, então as duas partes sempre somam exatamente o original.
- **O AudioSR é cuidadoso:**
  - Abaixo do corte detectado, o resultado é o seu áudio original.
  - Só a faixa acima vem do AudioSR, e com o volume ajustado.
  - O estéreo é processado como centro e laterais, o que mantém os agudos novos coerentes entre esquerda e direita.
- **Arquivos temporários** ficam em `/content/upscaler_work` e somem quando a sessão termina.
