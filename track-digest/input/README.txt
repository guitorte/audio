track-digest lê uma pasta de saída do music-to-midi — não músicas cruas.

Aponte OUTPUT_ROOT (notebook) ou o track_dir (CLI) para uma pasta de track já
processada, por exemplo:

    music-to-midi/output/<nome-da-track>/
        ├── stems/        vocals.wav, drums.wav, bass.wav, ...
        ├── stems_clean/  (opcional)
        ├── midi/         vocals.mid, drums.mid, ...
        ├── analysis/     <track>.txt, <stem>.txt, ...
        └── <track>.mid

Esta pasta input/ existe só como placeholder para manter o diretório no git.
