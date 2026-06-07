"""osu!standard → osu!mania converter (server-side).

Two-mode operation:

  1. With a replay's key-press events available (the normal case for a
     rendering pipeline driven by a .osr), we **reverse-engineer the
     original ManiaBeatmapConverter's column choices from what the player
     actually pressed**. For each std hit object's timestamp we look at
     the press events within ±150 ms and pick the column the player hit.
     This effectively reproduces osu!stable's conversion exactly for every
     note the player didn't miss — the position-based heuristic fills in
     for the misses.

  2. Without a replay, we fall back to a pragmatic reproduction — a
     deterministic position-based assignment with light RNG perturbation.

Either way the output is a fully-populated BeatmapInfo that the renderer
treats identically to a natively-parsed mania chart.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from osu_mania_renderer_v2.models import (
    BeatmapInfo,
    HitSample,
    HoldNote,
    KeyEvent,
    Note,
    TimingPoint,
)


# Default key count when a converted chart doesn't specify one. osu!mania
# itself picks the count based on the beatmap's CS + OD; we lock to 4K
# unless the caller forces something else, which matches the lazer default
# for "Convert to 4K" and is the most common community choice.
DEFAULT_CONVERTED_KEY_COUNT = 4


@dataclass(frozen=True)
class StandardHitObject:
    """Subset of a standard .osu HitObject the converter actually cares
    about. Spinners and edge cases get filtered out upstream."""
    time_ms: int
    x: int                          # 0-512, used for column seed
    is_slider: bool
    end_time_ms: int                # equals time_ms for circles
    new_combo: bool


def _slider_state_at(
    timing_points: tuple[TimingPoint, ...], t_ms: int,
) -> tuple[float, float]:
    """Active (beat_length_ms, sv_multiplier) at time t_ms. Walks both
    uninherited (red, BPM) and inherited (green, SV) timing points,
    keeping the most-recent of each kind."""
    beat_ms = 500.0
    sv = 1.0
    for tp in timing_points:
        if tp.time_ms > t_ms:
            break
        if tp.uninherited:
            beat_ms = tp.beat_length_ms
            sv = 1.0  # red TPs reset SV
        else:
            sv = tp.sv_multiplier
    return beat_ms, sv


def parse_standard_hit_objects(
    block: str,
    timing_points: tuple[TimingPoint, ...] = (),
    slider_multiplier: float = 1.4,
) -> list[StandardHitObject]:
    """Pull just what the converter needs from a standard [HitObjects]
    section. Standard hit-object types (bitmask in column 4):
       0x1 = circle, 0x2 = slider, 0x4 = new combo, 0x8 = spinner.

    Slider duration math (matches osu!stable):
        duration_ms = (pixel_length / (100 * slider_multiplier * sv))
                      * beat_length_ms * repeats
    With SliderMultiplier=1.4 (typical) and SV=1 at 120 BPM, a 100-pixel
    one-repeat slider lasts 357 ms ≈ 0.71 beats — within range to spawn
    a mini-stream when the BPM is high enough that 4 sub-divisions fit.
    """
    out: list[StandardHitObject] = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            x = int(parts[0])
            t = int(parts[2])
            type_bits = int(parts[3])
        except ValueError:
            continue
        if type_bits & 0x8:
            # Spinners: osu!stable's std→mania converter generates a short
            # stream during a spinner. Approximate with a single tap at
            # the start — the replay matcher handles column placement.
            out.append(StandardHitObject(
                time_ms=t, x=x, is_slider=False, end_time_ms=t,
                new_combo=bool(type_bits & 0x4),
            ))
            continue
        is_slider = bool(type_bits & 0x2)
        end_t = t
        if is_slider:
            try:
                repeats = max(1, int(parts[6]))
                pixel_length = float(parts[7])
            except (IndexError, ValueError):
                repeats = 1
                pixel_length = 100.0
            beat_ms, sv = _slider_state_at(timing_points, t)
            # The osu! slider duration formula:
            #   duration_per_repeat = px / (100 * SliderMultiplier * SV) * beat_ms
            denom = max(0.001, 100.0 * slider_multiplier * sv)
            per_repeat = (pixel_length / denom) * beat_ms
            est_duration = int(per_repeat * repeats)
            end_t = t + max(60, min(15000, est_duration))
        out.append(StandardHitObject(
            time_ms=t, x=x, is_slider=is_slider, end_time_ms=end_t,
            new_combo=bool(type_bits & 0x4),
        ))
    return out


@dataclass(frozen=True)
class PressEvent:
    """A column key-press event extracted from a replay's keys_held
    bitmask deltas. `time_ms` is when bit `column` flipped 0→1."""
    time_ms: int
    column: int


def extract_press_events(
    key_events: tuple[KeyEvent, ...], key_count: int,
) -> list[PressEvent]:
    """Walk consecutive replay frames, diffing keys_held to turn the
    bitmask timeline into a flat list of (time, column) press events.
    Releases are intentionally ignored — note placement only cares about
    where the player started pressing."""
    out: list[PressEvent] = []
    prev_mask = 0
    for ev in key_events:
        # 0→1 transitions only. AND with the complement of prev catches
        # exactly the bits that just turned on.
        new_presses = ev.keys_held & ~prev_mask
        prev_mask = ev.keys_held
        if new_presses == 0:
            continue
        for col in range(key_count):
            if new_presses & (1 << col):
                out.append(PressEvent(time_ms=ev.time_ms, column=col))
    return out


# Window (ms) for matching a hit object to a press event. osu!mania's
# strictest judgement (320) is ±16 ms; 100/50 stretch out to ~165 ms. Pick
# a value slightly wider than 50ms to capture even pretty late hits while
# rejecting "wrong note pressed seconds later" noise.
_PRESS_MATCH_WINDOW_MS = 150


def _assign_columns_to_notes(
    notes: list[Note | HoldNote],
    presses: list[PressEvent],
    key_count: int,
) -> dict[int, int]:
    """Match presses to the FULL list of expanded mania notes (slider
    streams included), keyed by note index in `notes`.

    Why expanded-notes, not std-object-level: a std slider blows up into
    N mania notes via `_expand_slider_to_stream`. The player pressed N
    times during the slider — those presses are in the replay. Earlier
    versions only matched one press per std object, which left the
    in-stream notes random-walked and visibly out of sync with the
    actual hand. Press-matching at the mania-note granularity flows the
    player's column choice into every sub-note of a converted slider.

    Strategy: walk notes in time order. For each note, take the closest
    unclaimed press inside the match window — that's the column. Chords
    claim distinct presses from the same window. Missed notes (no press
    within the window) stay unmatched and fall through to the
    position-based fallback in the caller."""
    if not presses:
        return {}
    presses_sorted = sorted(presses, key=lambda p: p.time_ms)
    claimed = [False] * len(presses_sorted)
    assignments: dict[int, int] = {}

    indexed_sorted = sorted(
        enumerate(notes), key=lambda iv: iv[1].time_ms,
    )
    cursor = 0
    for idx, n in indexed_sorted:
        while (cursor < len(presses_sorted)
                and presses_sorted[cursor].time_ms
                    < n.time_ms - _PRESS_MATCH_WINDOW_MS):
            cursor += 1
        best_i = -1
        best_dt = _PRESS_MATCH_WINDOW_MS + 1
        i = cursor
        while i < len(presses_sorted):
            dt = presses_sorted[i].time_ms - n.time_ms
            if dt > _PRESS_MATCH_WINDOW_MS:
                break
            if not claimed[i] and abs(dt) < best_dt:
                best_dt = abs(dt)
                best_i = i
            i += 1
        if best_i >= 0:
            claimed[best_i] = True
            assignments[idx] = presses_sorted[best_i].column
    return assignments


def _beat_length_at(timing_points: tuple[TimingPoint, ...], t_ms: int) -> float:
    """Active beat-length (ms per beat) at time `t_ms`. Walks uninherited
    (red) timing points only, since green points don't change BPM."""
    active = 500.0  # 120 BPM fallback
    for tp in timing_points:
        if tp.uninherited and tp.time_ms <= t_ms:
            active = tp.beat_length_ms
    return active


def _expand_slider_to_stream(
    obj: StandardHitObject,
    timing_points: tuple[TimingPoint, ...],
    key_count: int,
    rng: random.Random,
    last_column: int,
    overall_difficulty: float = 5.0,
) -> list[Note | HoldNote]:
    """One std slider → N mania notes spaced ~¼ beat apart, matching
    osu!stable's std→mania conversion which emits a stream during long
    sliders rather than a single hold. Short sliders (< 1 beat) become a
    single hold note. The final note in a stream is also a hold spanning
    any remainder so the slider's tail still registers as held.

    osu!stable gates this stream expansion by difficulty: low-difficulty
    maps (Beginner/Easy, OD ≤ 3) keep sliders as single holds because
    streaming them produces an unplayably-dense converted chart. At
    OD ≥ 5 (Hard/Insane source) sliders fully stream. Between we
    linearly interpolate the threshold.

    Columns are chosen here via position + stair walk; the replay-driven
    column matcher upstream overrides them when there's a press to bind
    to, so this only affects unmatched/post-fail notes."""
    beat_ms = max(50.0, _beat_length_at(timing_points, obj.time_ms))
    duration = obj.end_time_ms - obj.time_ms

    # Difficulty gate. OD ≤ 3 (Beginner/Easy) never streams; the entire
    # slider becomes one hold. OD ≥ 5 (Hard+) always streams long sliders.
    # In between, raise the minimum-duration threshold so only the longest
    # sliders stream — keeps medium-difficulty conversions reasonable.
    if overall_difficulty <= 3.0:
        stream_min_duration = float("inf")
    elif overall_difficulty >= 5.0:
        stream_min_duration = beat_ms
    else:
        # Linear interpolation between "never stream" and "stream at 1 beat".
        frac = (overall_difficulty - 3.0) / 2.0
        stream_min_duration = beat_ms / max(0.1, frac)

    if duration < stream_min_duration:
        # Short slider OR low-difficulty source — single hold, like the
        # previous behaviour.
        return [HoldNote(
            column=_pick_col(obj.x, key_count, last_column, rng),
            time_ms=obj.time_ms,
            end_time_ms=obj.end_time_ms,
            hit_sound=0, hit_sample=HitSample(),
        )]

    # Stream out at ¼-beat intervals. The last note is a hold covering
    # the remainder so the slider still releases at the correct time —
    # otherwise a hyper-long slider would end abruptly mid-press.
    step_ms = beat_ms / 4.0
    n_steps = max(2, int(duration / step_ms))
    notes: list[Note | HoldNote] = []
    col = _pick_col(obj.x, key_count, last_column, rng)
    for i in range(n_steps):
        t = int(obj.time_ms + i * step_ms)
        # Stair-walk: each successive stream note nudges one column over
        # so the stream "rolls" across the keyboard the way osu!stable
        # produces. Reverses at the edges.
        if i > 0:
            direction = 1 if (col < key_count - 1 and rng.random() < 0.6) else -1
            col = max(0, min(key_count - 1, col + direction))
        if i == n_steps - 1:
            # Tail hold absorbs the remainder.
            notes.append(HoldNote(
                column=col, time_ms=t,
                end_time_ms=obj.end_time_ms,
                hit_sound=0, hit_sample=HitSample(),
            ))
        else:
            notes.append(Note(
                column=col, time_ms=t,
                hit_sound=0, hit_sample=HitSample(),
            ))
    return notes


def _pick_col(
    x: int, key_count: int, last_column: int, rng: random.Random,
) -> int:
    """Position-based column choice with anti-stack + light shuffle. Used
    only when the replay matcher can't recover the column directly."""
    base = max(0, min(key_count - 1, (x * key_count) // 512))
    col = base
    if rng.random() < 0.30 and key_count > 1:
        delta = rng.choice((-1, 1))
        col = max(0, min(key_count - 1, col + delta))
    if col == last_column and key_count > 1:
        col = (col + 1) % key_count
    return col


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
) -> BeatmapInfo:
    """Run the conversion and return a fully-populated BeatmapInfo that
    looks identical (to the renderer) to a natively-parsed mania chart.

    When `replay_key_events` is provided we use them to recover the
    original column assignments — see module docstring for the rationale.
    Without them, `seed_source` (typically the beatmap id or md5) drives
    a deterministic RNG so the same chart always converts identically."""
    objects = parse_standard_hit_objects(
        hit_objects_block, timing_points, slider_multiplier,
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
    rng = random.Random(
        int(hashlib.md5((seed_source or "default").encode()).hexdigest(), 16),
    )

    # Phase 1: expand every std object into mania notes with **heuristic**
    # columns. These will be overwritten by replay-driven assignments in
    # Phase 2 wherever the player has a matching press. For misses (no
    # press within the window) the heuristic column stays.
    notes: list[Note | HoldNote] = []
    max_time = 0
    last_column = -1
    for obj in objects:
        if obj.is_slider:
            stream = _expand_slider_to_stream(
                obj, timing_points, key_count, rng, last_column,
                overall_difficulty=overall_difficulty,
            )
            notes.extend(stream)
            tail = stream[-1]
            tail_end = tail.end_time_ms if isinstance(tail, HoldNote) else tail.time_ms
            max_time = max(max_time, tail_end)
            last_column = stream[-1].column
        else:
            col = _pick_col(obj.x, key_count, last_column, rng)
            if (notes and obj.time_ms - notes[-1].time_ms < 40
                    and col == last_column):
                col = (col + (1 if col == 0 else -1)) % key_count
            notes.append(Note(
                column=col, time_ms=obj.time_ms,
                hit_sound=0, hit_sample=HitSample(),
            ))
            max_time = max(max_time, obj.time_ms)
            last_column = col

    # Phase 2: replay-driven column recovery. Matching runs against the
    # FULL expanded mania-note list — slider streams included — so each
    # in-stream note inherits the column the player actually pressed at
    # that ¼-beat instant, not the random stair-walk. Without this step
    # converted slider-heavy maps render with notes in totally different
    # columns than the player's hand → score-correct, visually broken.
    if replay_key_events:
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
    )
