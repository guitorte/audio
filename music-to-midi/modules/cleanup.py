"""Optional stem cleanup before transcription (from the ``Stems_4`` notebook).

Reproduces that notebook's post-separation cleanup cell:

* **true-mid mono** — down-mix stereo as ``(L + R) / 2`` (not a channel drop);
* **light denoise** — ``noisereduce`` spectral gate with ``prop_decrease=0.15``;
* **safe normalize** — scale so the peak sits at ``0.90`` (avoids clipping).

A cleaned mono stem tends to transcribe more cleanly (Basic Pitch / ADTOF both
work from a single channel), while the original stems stay untouched in
``stems/``. Heavy deps (librosa, soundfile, noisereduce) are imported lazily.
"""

from __future__ import annotations

import os
from typing import Dict, Optional


def clean_stem(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    to_mono: bool = True,
    prop_decrease: float = 0.15,
    normalize_peak: float = 0.90,
    subtype: str = "PCM_24",
    quiet: bool = False,
) -> str:
    """Clean one stem and write the result to ``output_path`` (a WAV).

    Mirrors the ``Stems_4`` cleanup: true-mid down-mix, light ``noisereduce``
    gate, then peak-normalise to ``normalize_peak``. Any of the three steps can
    be disabled (``to_mono=False`` / ``prop_decrease=0`` / ``normalize_peak=0``).
    """
    import numpy as np
    import librosa
    import soundfile as sf

    y, sr = librosa.load(input_path, sr=None, mono=False)

    # TRUE MID down-mix (average channels), not a left/right drop.
    if to_mono and y.ndim > 1:
        y = y.mean(axis=0)

    # LIGHT CLEANUP — light spectral noise gate.
    if prop_decrease and prop_decrease > 0:
        try:
            import noisereduce as nr

            y = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)
        except Exception as exc:  # noqa: BLE001 - denoise is best-effort
            if not quiet:
                print(f"[clean] noisereduce indisponível ({exc}); pulando denoise")

    # SAFE NORMALIZATION — scale peak to normalize_peak.
    if normalize_peak and normalize_peak > 0 and y.size:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak * normalize_peak

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_clean.wav"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    # soundfile expects (frames, channels); librosa gives (channels, frames).
    data = y.T if y.ndim > 1 else y
    sf.write(output_path, data, sr, subtype=subtype)
    return output_path


def clean_stems_folder(
    src_dir: str,
    dst_dir: str,
    *,
    quiet: bool = False,
    **clean_opts,
) -> Dict[str, str]:
    """Clean every audio file in ``src_dir`` into ``dst_dir`` (names preserved).

    Returns ``{stem_basename: cleaned_path}``. Filenames are kept (as ``.wav``)
    so the transcription stage still detects stem types from them.
    """
    from .separate import list_audio_files

    files = list_audio_files(src_dir)
    os.makedirs(dst_dir, exist_ok=True)
    out: Dict[str, str] = {}
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(dst_dir, f"{base}.wav")
        if not quiet:
            print(f"[clean] {os.path.basename(path)} -> {dst}")
        clean_stem(path, dst, quiet=quiet, **clean_opts)
        out[base] = dst
    return out
