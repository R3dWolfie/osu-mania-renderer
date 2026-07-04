"""osu!standard → osu!mania converter (server-side).

Conversion happens in two phases:

  1. NOTE GENERATION — a faithful port of osu!lazer's *legacy* std→mania
     beatmap conversion (``ManiaBeatmapConverter`` + the ``Legacy`` pattern
     generators), implemented in ``legacy_mania_convert``. This reproduces
     the EXACT set of mania notes lazer emits per std object: the count, the
     times, and whether each is a tap or a hold. The seed (derived from
     HP/CS/OD/AR) and the shared ``LegacyRandom`` stream are reproduced
     bit-for-bit so the RNG-driven note counts match. The notes carry
     lazer's own column choices at this stage.

  2. COLUMN RECOVERY — when a replay's key-press events are available (the
     normal case for a .osr-driven render), we discard lazer's column
     choices and re-derive each note's column from what the player actually
     pressed (``_assign_columns_to_notes``). Because phase 1 now produces
     lazer's exact note set, every note the player hit has a corresponding
     press, so the rendered chart tracks the player's hand. Notes with no
     matching press (true misses) keep lazer's generated column.

Without a replay, phase 2 is skipped and lazer's columns are used as-is.
Either way the output is a fully-populated BeatmapInfo that the renderer
treats identically to a natively-parsed mania chart.
"""
from __future__ import annotations

from dataclasses import dataclass

from osu_mania_renderer_v2.models import (
    BeatmapInfo,
    HitSample,
    HoldNote,
    KeyEvent,
    Note,
    TimingPoint,
)
from osu_mania_renderer_v2 import legacy_mania_convert as _legacy


# Default key count when a converted chart doesn't specify one. osu!mania
# itself picks the count based on the beatmap's CS + OD; we lock to 4K
# unless the caller forces something else, which matches the lazer default
# for "Convert to 4K" and is the most common community choice.
DEFAULT_CONVERTED_KEY_COUNT = 4


@dataclass(frozen=True)
class PressEvent:
    """A column key-press event extracted from a replay's keys_held
    bitmask deltas. `time_ms` is when bit `column` flipped 0→1.
    `hold_duration_ms` is how long the key stayed down before releasing
    (0 if it never released before the replay ended) — used to bias the
    column matcher toward pairing hold-notes with sustained presses."""
    time_ms: int
    column: int
    hold_duration_ms: int = 0


def extract_press_events(
    key_events: tuple[KeyEvent, ...], key_count: int,
) -> list[PressEvent]:
    """Walk consecutive replay frames, diffing keys_held to turn the
    bitmask timeline into a flat list of (time, column) press events,
    each carrying how long that key was held before release.

    Column placement only needs the press TIME + column; the hold
    duration is a secondary signal the matcher uses to prefer mapping a
    converted hold-note to a press the player actually sustained (rather
    than a quick tap that happened to fall nearby)."""
    out: list[PressEvent] = []
    prev_mask = 0
    # Track the press-down time of each currently-held column.
    down_since: dict[int, int] = {}
    # Index in `out` of the open press per column, so we can backfill its
    # duration on release.
    open_index: dict[int, int] = {}
    for ev in key_events:
        mask = ev.keys_held
        new_presses = mask & ~prev_mask          # 0→1 transitions
        new_releases = prev_mask & ~mask         # 1→0 transitions
        for col in range(key_count):
            bit = 1 << col
            if new_releases & bit and col in down_since:
                out[open_index[col]] = PressEvent(
                    time_ms=down_since[col], column=col,
                    hold_duration_ms=ev.time_ms - down_since[col],
                )
                down_since.pop(col, None)
                open_index.pop(col, None)
            if new_presses & bit:
                open_index[col] = len(out)
                down_since[col] = ev.time_ms
                out.append(PressEvent(time_ms=ev.time_ms, column=col))
        prev_mask = mask
    return out


# Window (ms) for matching a hit object to a press event. osu!mania's
# strictest judgement (320) is ±16 ms; 100/50 stretch out to ~165 ms. Pick
# a value slightly wider than 50ms to capture even pretty late hits while
# rejecting "wrong note pressed seconds later" noise.
_PRESS_MATCH_WINDOW_MS = 150

# Hold-awareness thresholds + penalties for the column matcher. A press
# held for >= _HOLD_PRESS_MIN_MS looks like a sustained hold; one released
# within _TAP_PRESS_MAX_MS looks like a tap. The penalties are deliberately
# small (sub-window) so they only break ties / near-ties between otherwise
# similar-distance candidate pairs, never override a clearly-closer match.
_HOLD_PRESS_MIN_MS = 60
_TAP_PRESS_MAX_MS = 130
_HOLD_MISMATCH_PENALTY = 40.0
_TAP_MISMATCH_PENALTY = 20.0


def _assign_columns_to_notes(
    notes: list[Note | HoldNote],
    presses: list[PressEvent],
    key_count: int,
) -> dict[int, int]:
    """Match presses to the FULL list of mania notes (slider streams +
    chords included), keyed by note index in `notes`. Returns, per note
    index, the column of the replay press that produced it.

    Why this matters: now that the note GENERATION is a faithful port of
    lazer (correct count / times / hold-vs-tap), every note the player hit
    has a corresponding press in the replay. We recover the column the
    player actually used so the rendered chart tracks their hand.

    Matching strategy — GLOBAL CLOSEST-PAIR (not greedy-by-time):

      Enumerate every (note, press) pair whose times fall within the match
      window, sort all candidate pairs by absolute time delta, then claim
      them closest-first; each note and each press is used at most once.

      This is a strict improvement over the old greedy-by-note-time scan,
      which mis-assigned in dense sections: an earlier note could grab a
      press that belonged (much more closely) to a slightly later note,
      starving the later note into a false miss. With the faithful — and
      therefore denser, more hold-heavy — note set, that greedy starvation
      was the dominant source of spurious misses. Closest-pair-first
      assignment resolves each contested press to the note it truly fits,
      which is what the player intended.

    Notes with no press inside the window stay unmatched and fall through
    to lazer's own column in the caller (true misses keep their generated
    column)."""
    if not presses:
        return {}
    presses_sorted = sorted(presses, key=lambda p: p.time_ms)
    press_times = [p.time_ms for p in presses_sorted]

    # Build all in-window (note, press) candidate pairs, scored by a cost.
    # Base cost is |time delta|. A small hold-awareness penalty discourages
    # pairing a converted hold-note with a quick tap (and a tap-note with a
    # long sustained hold) when a better-matching press exists — this
    # resolves the dense slider→hold sections where the player sustained one
    # key while tapping neighbours, which plain time-only matching can
    # mis-pair into false misses.
    from bisect import bisect_left
    candidates: list[tuple[float, int, int]] = []  # (cost, note_idx, press_idx)
    for idx, n in enumerate(notes):
        is_hold = isinstance(n, HoldNote)
        lo = bisect_left(press_times, n.time_ms - _PRESS_MATCH_WINDOW_MS)
        i = lo
        while i < len(press_times):
            p = presses_sorted[i]
            dt = p.time_ms - n.time_ms
            if dt > _PRESS_MATCH_WINDOW_MS:
                break
            cost = float(abs(dt))
            if is_hold and p.hold_duration_ms < _HOLD_PRESS_MIN_MS:
                cost += _HOLD_MISMATCH_PENALTY
            elif (not is_hold) and p.hold_duration_ms > _TAP_PRESS_MAX_MS:
                cost += _TAP_MISMATCH_PENALTY
            candidates.append((cost, idx, i))
            i += 1

    # Claim closest-first (lowest cost). Ties are broken by the stable sort
    # on (note_idx, press_idx), which is deterministic.
    candidates.sort()
    note_claimed = [False] * len(notes)
    press_claimed = [False] * len(presses_sorted)
    assignments: dict[int, int] = {}
    for _dt, note_idx, press_idx in candidates:
        if note_claimed[note_idx] or press_claimed[press_idx]:
            continue
        note_claimed[note_idx] = True
        press_claimed[press_idx] = True
        assignments[note_idx] = presses_sorted[press_idx].column
    return assignments


def _parse_std_objects_for_legacy(
    block: str,
    timing_points: tuple[TimingPoint, ...],
    kiai_points: tuple[tuple[int, bool], ...],
) -> list[_legacy.StdObject]:
    """Parse a standard ``[HitObjects]`` block into the richer
    ``StdObject`` form the lazer port needs (kiai + per-object slider
    velocity + edge hitsounds), reusing the already-parsed timing points
    for SV/BPM lookups.

    Slider line layout:
        x,y,time,type,hitSound,curve,slides,length,edgeSounds,edgeSets,sample
    Spinner line layout:
        x,y,time,type,hitSound,endTime,sample
    """
    out: list[_legacy.StdObject] = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            x = int(float(parts[0]))
            y = int(float(parts[1]))
            t = int(parts[2])
            type_bits = int(parts[3])
            hit_sound = int(parts[4])
        except ValueError:
            continue

        if type_bits & 0x8:
            # Spinner.
            try:
                end_t = int(parts[5])
            except (IndexError, ValueError):
                end_t = t
            out.append(_legacy.StdObject(
                start_time=t, x=x, y=y, kind="spinner", end_time=end_t,
                hit_sound=hit_sound,
            ))
            continue

        if type_bits & 0x2:
            # Slider.
            try:
                slides = max(1, int(parts[6]))
            except (IndexError, ValueError):
                slides = 1
            try:
                pixel_length = float(parts[7])
            except (IndexError, ValueError):
                pixel_length = 0.0
            node_sounds: tuple[int, ...] = ()
            if len(parts) > 8 and parts[8].strip():
                try:
                    node_sounds = tuple(
                        int(z) for z in parts[8].split("|") if z != "")
                except ValueError:
                    node_sounds = ()
            out.append(_legacy.StdObject(
                start_time=t, x=x, y=y, kind="slider", end_time=t,
                hit_sound=hit_sound, span_count=slides,
                pixel_length=pixel_length,
                slider_velocity=_sv_at(timing_points, t),
                has_kiai=_kiai_at(kiai_points, t),
                node_hit_sounds=node_sounds,
            ))
            continue

        # Circle.
        out.append(_legacy.StdObject(
            start_time=t, x=x, y=y, kind="circle", end_time=t,
            hit_sound=hit_sound,
        ))
    return out


def _sv_at(timing_points: tuple[TimingPoint, ...], t_ms: int) -> float:
    """Per-object slider velocity at ``t_ms`` — the most-recent green
    (inherited) TP's multiplier, reset to 1.0 by any red (uninherited) TP."""
    sv = 1.0
    for tp in timing_points:
        if tp.time_ms > t_ms:
            break
        sv = 1.0 if tp.uninherited else tp.sv_multiplier
    return sv


def _beat_length_for_legacy(timing_points: tuple[TimingPoint, ...]):
    """Build a ``beat_length_at(time)`` closure over the uninherited
    (red) timing points, matching ``ControlPointInfo.TimingPointAt``."""
    reds = [tp for tp in timing_points if tp.uninherited]

    def beat_length_at(t_ms: float) -> float:
        active = 500.0
        for tp in reds:
            if tp.time_ms <= t_ms:
                active = tp.beat_length_ms
            else:
                break
        return active

    return beat_length_at


def _kiai_at(kiai_points: tuple[tuple[int, bool], ...], t_ms: int) -> bool:
    """KiaiMode of the effect control point active at ``t_ms``."""
    cur = False
    for (time, kiai) in kiai_points:
        if time > t_ms:
            break
        cur = kiai
    return cur


def _generate_lazer_notes(
    *,
    hit_objects_block: str,
    timing_points: tuple[TimingPoint, ...],
    slider_multiplier: float,
    overall_difficulty: float,
    hp_drain_rate: float | None,
    circle_size: float | None,
    approach_rate: float | None,
    total_break_time_ms: float,
    kiai_points: tuple[tuple[int, bool], ...],
    key_count: int,
) -> tuple[list[Note | HoldNote], int]:
    """Faithful lazer std→mania note generation. Returns (notes, max_time).

    Columns carry lazer's own choices here; the caller re-derives them from
    the replay. If the real difficulty settings weren't threaded through
    (older caller), estimate them from ``overall_difficulty`` so the seed +
    conversionDifficulty stay well-defined (this only affects the RNG-driven
    note count, never correctness of the column-recovery path)."""
    drain_rate = hp_drain_rate if hp_drain_rate is not None else overall_difficulty
    cs = circle_size if circle_size is not None else 4.0
    ar = approach_rate if approach_rate is not None else overall_difficulty

    std_objects = _parse_std_objects_for_legacy(
        hit_objects_block, timing_points, kiai_points)
    beat_length_at = _beat_length_for_legacy(timing_points)

    result = _legacy.convert_legacy(
        std_objects,
        total_columns=key_count,
        drain_rate=drain_rate,
        circle_size=cs,
        overall_difficulty=overall_difficulty,
        approach_rate=ar,
        slider_multiplier=slider_multiplier,
        total_break_time=total_break_time_ms,
        beat_length_at=beat_length_at,
    )

    notes: list[Note | HoldNote] = []
    max_time = 0
    for o in result.objects:
        if o.is_hold:
            notes.append(HoldNote(
                column=o.column, time_ms=o.start_time,
                end_time_ms=o.end_time,
                hit_sound=o.hit_sound, hit_sample=HitSample(),
            ))
            max_time = max(max_time, o.end_time)
        else:
            notes.append(Note(
                column=o.column, time_ms=o.start_time,
                hit_sound=o.hit_sound, hit_sample=HitSample(),
            ))
            max_time = max(max_time, o.start_time)
    return notes, max_time


def convert_standard_to_mania(
    *,
    hit_objects_block: str,
    timing_points: tuple[TimingPoint, ...],
    audio_filename: str | None,
    background_filename: str | None,
    audio_lead_in_ms: int,
    artist: str, title: str, difficulty: str, creator: str,
    beatmap_id: int | None, beatmapset_id: int | None,
    default_sample_set: str,
    slider_multiplier: float = 1.4,
    overall_difficulty: float = 5.0,
    key_count: int = DEFAULT_CONVERTED_KEY_COUNT,
    seed_source: str = "",
    replay_key_events: tuple[KeyEvent, ...] | None = None,
    # --- New: faithful-lazer conversion inputs. All optional so existing
    # callers keep working; beatmap.py now threads the real difficulty
    # settings through so the lazer note GENERATION can be reproduced
    # exactly (the seed + conversionDifficulty depend on every one of
    # these). When hp_drain_rate/circle_size/approach_rate are left at
    # their sentinel (None), we fall back to overall_difficulty-derived
    # estimates so behaviour degrades gracefully.
    hp_drain_rate: float | None = None,
    circle_size: float | None = None,
    approach_rate: float | None = None,
    total_break_time_ms: float = 0.0,
    kiai_points: tuple[tuple[int, bool], ...] = (),
) -> BeatmapInfo:
    """Run the conversion and return a fully-populated BeatmapInfo that
    looks identical (to the renderer) to a natively-parsed mania chart.

    Note GENERATION (which notes exist, at what times, tap vs hold) is a
    faithful port of osu!lazer's legacy std→mania conversion — see
    ``legacy_mania_convert``. The renderer's existing replay-driven column
    recovery then overwrites lazer's column choices wherever the player has
    a matching key-press, so the visual columns track the player's hand.

    When `replay_key_events` is provided we use them to recover the
    column assignments — see module docstring for the rationale."""
    # Faithful lazer note generation (count + times + hold-vs-tap).
    notes, max_time = _generate_lazer_notes(
        hit_objects_block=hit_objects_block,
        timing_points=timing_points,
        slider_multiplier=slider_multiplier,
        overall_difficulty=overall_difficulty,
        hp_drain_rate=hp_drain_rate,
        circle_size=circle_size,
        approach_rate=approach_rate,
        total_break_time_ms=total_break_time_ms,
        kiai_points=kiai_points,
        key_count=key_count,
    )

    # CRITICAL for converted maps: strip inherited (green) timing points.
    # In osu!standard those carry slider-velocity multipliers used for
    # slider geometry; in mania they tell the renderer how fast each
    # note should scroll. Passing std SV through to a converted mania
    # chart causes random scroll-speed jumps mid-song. osu!stable's
    # converter drops them entirely — only the BPM (red TP) info is
    # meaningful for the rendered mania chart.
    mania_timing_points = tuple(
        TimingPoint(
            time_ms=tp.time_ms, sample_set=tp.sample_set,
            custom_index=tp.custom_index, volume=tp.volume,
            sv_multiplier=1.0,         # force constant scroll speed
            uninherited=True,
            beat_length_ms=tp.beat_length_ms,
        )
        for tp in timing_points if tp.uninherited
    )

    # Phase 1 (note generation) already happened above via the faithful
    # lazer port in `_generate_lazer_notes`. `notes` holds the exact set of
    # mania notes lazer would emit (correct count, times, hold-vs-tap),
    # carrying lazer's column choices — which Phase 2 below overwrites from
    # the replay wherever the player has a matching press.

    # Phase 2: replay-driven column recovery. Matching runs against the
    # FULL lazer note set — slider-derived notes and chords included — so
    # each note inherits the column the player actually pressed for it,
    # rather than lazer's RNG column choice. Without this step a converted
    # chart renders score-correct but with notes in different columns than
    # the player's hand (visually broken).
    #
    # IMPORTANT: lazer already placed every note in a specific column via
    # the shared RNG stream (HitCircle/Slider/Spinner pattern generators).
    # Those columns ARE what the player saw and played against, so for a
    # faithful replay simulation we keep them. The press-matching recovery
    # below is a heuristic that mis-fires in dense streams (an earlier note
    # steals a press belonging to a slightly-later note → false miss), which
    # is the source of the spurious misses. It is therefore disabled by
    # default; set OMR_PRESS_RECOVERY=1 to fall back to the old heuristic.
    import os as _os
    _use_press_recovery = _os.environ.get("OMR_PRESS_RECOVERY", "0") == "1"
    if replay_key_events and _use_press_recovery:
        presses = extract_press_events(replay_key_events, key_count)
        note_assignments = _assign_columns_to_notes(notes, presses, key_count)
        if note_assignments:
            rebuilt: list[Note | HoldNote] = []
            for i, n in enumerate(notes):
                col = note_assignments.get(i)
                if col is None or col == n.column:
                    rebuilt.append(n)
                    continue
                if isinstance(n, HoldNote):
                    rebuilt.append(HoldNote(
                        column=col, time_ms=n.time_ms,
                        end_time_ms=n.end_time_ms,
                        hit_sound=n.hit_sound, hit_sample=n.hit_sample,
                    ))
                else:
                    rebuilt.append(Note(
                        column=col, time_ms=n.time_ms,
                        hit_sound=n.hit_sound, hit_sample=n.hit_sample,
                    ))
            notes = rebuilt

    notes_sorted = tuple(sorted(notes, key=lambda n: n.time_ms))

    return BeatmapInfo(
        key_count=key_count,
        notes=notes_sorted,
        audio_filename=audio_filename,
        background_filename=background_filename,
        total_duration_ms=max_time,
        audio_lead_in_ms=audio_lead_in_ms,
        artist=artist,
        title=title,
        difficulty=f"{difficulty} (converted {key_count}K)",
        creator=creator,
        beatmap_id=beatmap_id,
        beatmapset_id=beatmapset_id,
        default_sample_set=default_sample_set,
        timing_points=mania_timing_points,
        overall_difficulty=overall_difficulty,
    )
