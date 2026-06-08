#!/usr/bin/env python3
"""CLI for the music-to-midi workflow.

Usage:
    python run.py "song.mp3" -o output/
    python run.py "song.mp3" -o output/ --model htdemucs --shifts 5 --overlap 0.75

Runs the full song → stems → multi-track MIDI pipeline and prints a summary.
"""

from __future__ import annotations

import argparse
import sys
import time

from modules import DEMUCS_MODELS, song_to_midi


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Song → stems → multi-track MIDI")
    parser.add_argument("song", help="Path to the input song (wav/mp3/flac/...)")
    parser.add_argument("-o", "--output", default="output",
                        help="Output root directory (default: ./output)")
    parser.add_argument("--model", default="htdemucs_6s",
                        choices=sorted(DEMUCS_MODELS),
                        help="Demucs model (default: htdemucs_6s — 6 stems)")
    parser.add_argument("--shifts", type=int, default=1,
                        help="Demucs test-time shifts; higher = better/slower")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Demucs chunk overlap (default 0.25)")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel jobs")
    parser.add_argument("--device", default=None, help="cuda | cpu (auto-detected)")
    parser.add_argument("--mp3-stems", action="store_true",
                        help="Write stems as mp3 instead of wav")
    parser.add_argument("--force-separation", action="store_true",
                        help="Re-run Demucs even if stems already exist")
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip stem cleanup (mono/denoise/normalize) before transcribing")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Skip piano-roll PNG + MIDI-TEXT generation")
    args = parser.parse_args(argv)

    t0 = time.time()
    result = song_to_midi(
        args.song,
        args.output,
        model=args.model,
        shifts=args.shifts,
        overlap=args.overlap,
        jobs=args.jobs,
        device=args.device,
        mp3_stems=args.mp3_stems,
        skip_separation_if_exists=not args.force_separation,
        clean_stems=not args.no_clean,
        write_analysis=not args.no_analysis,
    )
    dt = time.time() - t0

    tr = result.transcription
    print("\n=== Resumo ===")
    print(f"Faixa             : {result.track_name}")
    print(f"Pasta             : {result.track_dir}")
    print(f"Tempo total       : {dt:.1f}s")
    print(f"Stems transcritos : {len(tr.stem_results)}")
    print(f"Total de notas    : {tr.total_notes}")
    print(f"Duração do MIDI   : {tr.duration_s:.1f}s")
    print(f"MIDI consolidado  : {tr.midi_path}")
    if result.analysis_dir:
        print(f"Análise (PNG+TXT) : {result.analysis_dir}")
    print()
    print(f"{'Stem':10s} {'Engine':14s} {'Notas':>6s} {'Duração':>10s}")
    print("-" * 44)
    for stem_type, r in tr.stem_results.items():
        print(f"{stem_type:10s} {r.engine:14s} {r.n_notes:6d} {r.duration_s:9.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
