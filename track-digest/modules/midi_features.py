"""MIDI-derived musical features for one instrument track.

Everything here works off ``pretty_midi.Instrument`` note objects. Pitched
paths (key, intervals, pitch range) are skipped for drum tracks, which route
through :func:`drum_summary` instead. Tempo must be supplied by the caller —
Basic Pitch MIDIs carry a bogus default 120 BPM, so an audio-derived tempo
(see :func:`digest.estimate_tempo`) keeps notes/bar and the rhythm grid honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .theory import (
    INTERVAL_NAMES,
    KeyEstimate,
    drum_group,
    estimate_key,
    note_name,
)

# Note duration in beats → label; used to name the dominant rhythmic values.
_DUR_LABELS = {
    4.0: "whole", 3.0: "dot-half", 2.0: "1/2", 1.5: "dot-1/4",
    1.0: "1/4", 0.75: "dot-1/8", 0.5: "1/8", 0.375: "dot-1/16",
    0.25: "1/16", 0.125: "1/32",
}


@dataclass
class RhythmProfile:
    grid: str = "?"                # finest subdivision covering most onsets
    offbeat_frac: float = 0.0
    syncopation: str = "on-beat"
    tightness: float = 0.0         # 1.0 = perfectly quantized to the grid
    dominant_durations: List[str] = field(default_factory=list)

    def label(self) -> str:
        durs = ",".join(self.dominant_durations) if self.dominant_durations else "?"
        return (f"{self.grid} grid, {self.syncopation}, tight {self.tightness:.2f} "
                f"| durs {durs}")


@dataclass
class MidiFeatures:
    name: str
    is_drum: bool
    n_notes: int
    duration_s: float
    notes_per_sec: float = 0.0
    notes_per_bar: float = 0.0
    pitch_lo: Optional[int] = None
    pitch_hi: Optional[int] = None
    pitch_lo_name: str = ""
    pitch_hi_name: str = ""
    polyphony: float = 1.0
    poly_word: str = "mono"
    key: Optional[KeyEstimate] = None
    dominant_intervals: List[str] = field(default_factory=list)
    rhythm: Optional[RhythmProfile] = None
    vel_min: int = 0
    vel_max: int = 0
    vel_mean: float = 0.0
    vel_spread: int = 0
    vel_word: str = "flat"
    # drum-only
    drum_pieces: Optional[List[str]] = None
    drum_pattern: Optional[str] = None


def _rhythm_profile(notes, sec_per_beat: float) -> RhythmProfile:
    """Snap onsets to a 16th grid to characterise the rhythmic feel."""
    if not notes or sec_per_beat <= 0:
        return RhythmProfile()
    sec_per_16th = sec_per_beat / 4.0

    residuals = []          # distance from nearest 16th slot, in [0, 0.5]
    slots = []              # slot index mod 16
    covered = {4: 0, 2: 0, 1: 0}  # onsets landing on 16th / 8th / quarter
    for n in notes:
        pos16 = n.start / sec_per_16th
        nearest = round(pos16)
        residuals.append(abs(pos16 - nearest))
        slots.append(nearest % 16)
        if nearest % 4 == 0:
            covered[1] += 1  # on a quarter
        elif nearest % 2 == 0:
            covered[2] += 1  # on an 8th
        else:
            covered[4] += 1  # on a 16th

    n_on = len(notes)
    tightness = max(0.0, 1.0 - (sum(residuals) / n_on) * 2.0)
    # Finest subdivision needed to cover >=90% of onsets.
    on_16 = covered[4]
    on_8 = covered[2]
    if on_16 / n_on > 0.10:
        grid = "16th"
    elif on_8 / n_on > 0.10:
        grid = "8th"
    else:
        grid = "1/4"
    # Off-beat = not landing on a quarter-note slot {0,4,8,12}.
    offbeat = sum(1 for s in slots if s % 4 != 0) / n_on

    from .descriptors import syncopation_word
    prof = RhythmProfile(
        grid=grid,
        offbeat_frac=round(offbeat, 3),
        syncopation=syncopation_word(offbeat),
        tightness=round(tightness, 2),
        dominant_durations=_dominant_durations(notes, sec_per_beat),
    )
    return prof


def _dominant_durations(notes, sec_per_beat: float) -> List[str]:
    """Top-2 note-duration labels by total sounding time."""
    if sec_per_beat <= 0:
        return []
    totals: dict = {}
    edges = sorted(_DUR_LABELS)
    for n in notes:
        beats = (n.end - n.start) / sec_per_beat
        # nearest label in log space (musical durations are multiplicative)
        best = min(edges, key=lambda e: abs((beats or 1e-6) / e - 1.0)
                   if e else 1e9)
        totals[_DUR_LABELS[best]] = totals.get(_DUR_LABELS[best], 0.0) + (n.end - n.start)
    ranked = sorted(totals, key=lambda k: totals[k], reverse=True)
    return ranked[:2]


def _polyphony(notes) -> float:
    """Time-weighted mean number of simultaneously sounding notes."""
    if not notes:
        return 0.0
    events = []  # (time, +1 on / -1 off)
    for n in notes:
        events.append((n.start, 1))
        events.append((n.end, -1))
    events.sort()
    active = 0
    prev_t = events[0][0]
    weighted = 0.0
    total_t = 0.0
    for t, delta in events:
        dt = t - prev_t
        if active > 0 and dt > 0:
            weighted += active * dt
            total_t += dt
        active += delta
        prev_t = t
    return weighted / total_t if total_t > 0 else 1.0


def _dominant_intervals(notes) -> List[str]:
    """Top-2 interval classes among notes sharing an onset (chord hints)."""
    from collections import defaultdict
    by_onset: dict = defaultdict(list)
    for n in notes:
        by_onset[round(n.start, 2)].append(n.pitch)
    tally: dict = {}
    for pitches in by_onset.values():
        if len(pitches) < 2:
            continue
        pitches = sorted(pitches)
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                ic = (pitches[j] - pitches[i]) % 12
                if ic == 0:
                    continue
                name = INTERVAL_NAMES.get(ic, str(ic))
                tally[name] = tally.get(name, 0) + 1
    ranked = sorted(tally, key=lambda k: tally[k], reverse=True)
    return ranked[:2]


def drum_summary(notes, sec_per_beat: float):
    """Return ``(pieces, pattern)`` for a drum track.

    ``pieces`` lists present families; ``pattern`` names common grooves
    (four-on-floor, backbeat) plus hat subdivision.
    """
    if not notes:
        return [], "(sem notas)"
    groups = [drum_group(n.pitch) for n in notes]
    present = []
    for g in ("kick", "snare", "hihat", "tom", "cymbal", "clap", "perc"):
        if g in groups:
            present.append(g)

    labels = []
    if sec_per_beat > 0:
        sec_per_beat_slot = sec_per_beat
        # Beat slot (0..3 within a bar) each hit lands on.
        def beat_slots(family):
            out = set()
            for n, g in zip(notes, groups):
                if g != family:
                    continue
                beat = (n.start / sec_per_beat_slot) % 4
                out.add(round(beat) % 4)
            return out

        kick_beats = beat_slots("kick")
        snare_beats = beat_slots("snare")
        if kick_beats >= {0, 1, 2, 3}:
            labels.append("four-on-floor")
        if {1, 3} <= snare_beats:
            labels.append("backbeat")

        # Hi-hat subdivision from mean inter-onset spacing.
        hats = [n.start for n, g in zip(notes, groups) if g == "hihat"]
        if len(hats) > 2:
            hats.sort()
            iois = [b - a for a, b in zip(hats, hats[1:]) if b > a]
            if iois:
                mean_ioi = sum(iois) / len(iois)
                sub = sec_per_beat / mean_ioi if mean_ioi > 0 else 0
                if sub >= 3.0:
                    labels.append("16th hats")
                elif sub >= 1.5:
                    labels.append("8th hats")
                else:
                    labels.append("1/4 hats")

    pattern = ", ".join(labels) if labels else "irregular"
    return present, pattern


def extract_midi_features(inst, *, bpm: float, name: Optional[str] = None) -> MidiFeatures:
    """Analyse one ``pretty_midi.Instrument`` into a :class:`MidiFeatures`."""
    from .descriptors import poly_word as _poly_word, vel_spread_word

    notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
    name = name or inst.name or "track"
    is_drum = bool(inst.is_drum)
    n = len(notes)
    duration = (max(x.end for x in notes) - min(x.start for x in notes)) if notes else 0.0
    sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5
    sec_per_bar = 4.0 * sec_per_beat  # assume 4/4

    feat = MidiFeatures(
        name=name, is_drum=is_drum, n_notes=n, duration_s=round(duration, 2),
    )
    if n == 0:
        return feat

    span = duration if duration > 0 else 1e-6
    feat.notes_per_sec = round(n / span, 2)
    feat.notes_per_bar = round(n / (span / sec_per_bar), 2)

    vels = [x.velocity for x in notes]
    feat.vel_min, feat.vel_max = min(vels), max(vels)
    feat.vel_mean = round(sum(vels) / n, 1)
    feat.vel_spread = feat.vel_max - feat.vel_min
    feat.vel_word = vel_spread_word(feat.vel_spread)

    feat.rhythm = _rhythm_profile(notes, sec_per_beat)

    if is_drum:
        feat.drum_pieces, feat.drum_pattern = drum_summary(notes, sec_per_beat)
        return feat

    # --- pitched-only features ---
    pitches = [x.pitch for x in notes]
    feat.pitch_lo, feat.pitch_hi = min(pitches), max(pitches)
    feat.pitch_lo_name = note_name(feat.pitch_lo)
    feat.pitch_hi_name = note_name(feat.pitch_hi)
    feat.polyphony = round(_polyphony(notes), 2)
    feat.poly_word = _poly_word(feat.polyphony)

    pc = [0.0] * 12
    for x in notes:
        pc[x.pitch % 12] += (x.end - x.start)  # duration-weighted
    feat.key = estimate_key(pc)
    feat.dominant_intervals = _dominant_intervals(notes)
    return feat
