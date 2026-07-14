#!/usr/bin/env python3
"""CLI for track-digest.

Two subcommands, both operating on a music-to-midi output track folder:

    # compact AI digest (melody + timbre) -> analysis/<track>_digest.txt
    python run.py digest "output/My Song" --timbre-from raw

    # crop all stems to a region -> <track>/crops/*.wav
    python run.py crop "output/My Song" --start 12.5 --end 30.0
    python run.py crop "output/My Song" --start 12.5 --end 30.0 --include vocals,bass
"""

from __future__ import annotations

import argparse
import sys

from modules import (
    build_track_digest,
    crop_stems,
    default_digest_path,
    render_digest,
    write_digest,
)


def _cmd_digest(args) -> int:
    td = build_track_digest(
        args.track_dir,
        sr=args.sr,
        timbre_from=args.timbre_from,
        tempo_source=args.tempo_source,
        bpm=args.bpm,
        analyze_seconds=args.analyze_seconds,
    )
    text = render_digest(td)
    print(text)
    out = args.out or default_digest_path(args.track_dir)
    write_digest(td, out)
    print("=== Resumo ===")
    print(f"Faixa   : {td.track_name}")
    print(f"BPM     : {td.bpm:g} ({td.bpm_source})")
    print(f"Stems   : {len(td.stems)}")
    print(f"Digest  : {out}")
    return 0


def _cmd_crop(args) -> int:
    include = [s.strip() for s in args.include.split(",")] if args.include else None
    res = crop_stems(
        args.track_dir,
        args.start,
        args.end,
        out_dir=args.out,
        to_mono=args.mono,
        sr=args.sr,
        include=include,
    )
    print("\n=== Resumo ===")
    print(f"Região  : {res.start_s:g}-{res.end_s:g}s")
    print(f"Saída   : {res.out_dir}")
    print(f"Cortados: {len(res.crops)}")
    for stem, path in res.crops.items():
        print(f"  {stem:10s} -> {path}")
    if res.skipped:
        print(f"Pulados : {', '.join(res.skipped)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compact AI digest + batch stem-crop for a music-to-midi track")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("digest", help="Build the compact DIGEST v1 text")
    d.add_argument("track_dir", help="A music-to-midi output track folder")
    d.add_argument("--timbre-from", choices=["raw", "clean"], default="raw",
                   help="Which stems to read audio timbre from (default: raw)")
    d.add_argument("--tempo-source", choices=["audio", "midi"], default="audio",
                   help="Tempo source; audio beat-tracking is more reliable")
    d.add_argument("--bpm", type=float, default=None,
                   help="Override tempo entirely (skips estimation)")
    d.add_argument("--analyze-seconds", type=float, default=60.0,
                   help="Window (s) analysed per stem for timbre (default 60)")
    d.add_argument("--sr", type=int, default=22050, help="Analysis sample rate")
    d.add_argument("--out", default=None, help="Digest output path")
    d.set_defaults(func=_cmd_digest)

    c = sub.add_parser("crop", help="Crop all stems to a time region → WAV")
    c.add_argument("track_dir", help="A music-to-midi output track folder (or stems dir)")
    c.add_argument("--start", type=float, required=True, help="Start time (s)")
    c.add_argument("--end", type=float, required=True, help="End time (s)")
    c.add_argument("--out", default=None, help="Output dir (default <track>/crops)")
    c.add_argument("--mono", action="store_true", help="Down-mix crops to mono")
    c.add_argument("--sr", type=int, default=None, help="Resample (default: native)")
    c.add_argument("--include", default=None,
                   help="Comma-separated stem names to crop (default: all)")
    c.set_defaults(func=_cmd_crop)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
