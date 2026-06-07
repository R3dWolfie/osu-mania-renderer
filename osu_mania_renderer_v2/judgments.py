"""Determine each note's judgment from the replay's keypress timeline.

Mania timing windows (osu!stable, OD = 8 default):
    ±16.5 ms  → 320 (geki / rainbow 300)
    ±40   ms  → 300
    ±73   ms  → 200 (katu)
    ±103  ms  → 100
    ±127  ms  → 50
    > 127 ms  → miss
"""
from __future__ import annotations

from dataclasses import dataclass

from osu_mania_renderer_v2.models import KeyEvent

# Stable mania uses FIXED hit windows regardless of OD. These are the
# constants we used until we learned osu!web displays lazer-recomputed
# scores. Kept as a fallback when OD isn't available.
WINDOW_320 = 16.5
WINDOW_300 = 40
WINDOW_200 = 73
WINDOW_100 = 103
WINDOW_50 = 127

# Hold-tail release window multiplier — stable is ~1.5× more lenient
# on release timing than on press timing.
_TAIL_MULTIPLIER = 1.5
WINDOW_TAIL_320 = WINDOW_320 * _TAIL_MULTIPLIER
WINDOW_TAIL_300 = WINDOW_300 * _TAIL_MULTIPLIER
WINDOW_TAIL_200 = WINDOW_200 * _TAIL_MULTIPLIER
WINDOW_TAIL_100 = WINDOW_100 * _TAIL_MULTIPLIER
WINDOW_TAIL_50  = WINDOW_50  * _TAIL_MULTIPLIER


def windows_for_od(overall_difficulty: float) -> tuple[float, float, float, float, float]:
    """Lazer-style OD-scaled mania hit windows (in ms).

    Returns `(w_320, w_300, w_200, w_100, w_50)`. osu!web displays
    lazer-recomputed values for mania plays, so we use these windows
    to match the website's tier counts more closely than stable's
    fixed windows would.

    Formula (lazer source):
      320 = 16ms (constant; the 320 / "PERFECT" window doesn't scale)
      300 =  64 - 3 * OD
      200 =  97 - 3 * OD
      100 = 127 - 3 * OD
       50 = 151 - 3 * OD
    """
    return (
        16.0,
        max(1.0, 64.0  - 3.0 * overall_difficulty),
        max(1.0, 97.0  - 3.0 * overall_difficulty),
        max(1.0, 127.0 - 3.0 * overall_difficulty),
        max(1.0, 151.0 - 3.0 * overall_difficulty),
    )


@dataclass(frozen=True)
class JudgmentEvent:
    time_ms: int     # scheduled note time (when it reaches the receptor)
    column: int
    judgment: str    # "geki" | "300" | "katu" | "100" | "50" | "miss"
    hit_offset_ms: float | None = None
    # Signed press-time minus note-time. Negative = early, positive = late.
    # None on a true miss (no press matched). Used to compute UR + avg offset.


@dataclass(frozen=True)
class JudgmentTimeline:
    events: tuple[JudgmentEvent, ...]
    count_geki: int
    count_300: int
    count_katu: int
    count_100: int
    count_50: int
    count_miss: int
    max_combo: int


def compute_judgments(
    notes: tuple,
    events: tuple[KeyEvent, ...],
    key_count: int,
    overall_difficulty: float | None = None,
) -> JudgmentTimeline:
    """Pair each scoring event (tap, hold head, hold tail) with the closest
    matching keypress/release and classify by mania timing window.

    Hold notes contribute TWO scoring events (head ↔ press; tail ↔
    release). Previously we judged each hold as a single event, which left
    our final counts off by `N_holds` from what osu! actually recorded.
    """
    from osu_mania_renderer_v2.models import HoldNote

    if not events:
        return _all_miss(notes)

    presses = _rising_edges(events, key_count)
    releases = _falling_edges(events, key_count)

    # OD-scaled (lazer-style) tap windows when the caller passed an OD.
    # Falls back to stable's fixed windows when None — used by unit
    # tests and any legacy callsite that hasn't been migrated yet.
    if overall_difficulty is not None:
        od_320, od_300, od_200, od_100, od_50 = windows_for_od(overall_difficulty)
    else:
        od_320, od_300, od_200, od_100, od_50 = (
            WINDOW_320, WINDOW_300, WINDOW_200, WINDOW_100, WINDOW_50,
        )

    # Build a unified, time-sorted scoring-event list. Each entry is
    # (scheduled_time_ms, column, kind) — kind is "tap" (matched against a
    # rising edge) or "tail" (matched against a falling edge).
    scoring = []
    for note in notes:
        scoring.append((note.time_ms, note.column, "tap"))
        if isinstance(note, HoldNote):
            scoring.append((note.end_time_ms, note.column, "tail"))
    scoring.sort()

    j_events: list[JudgmentEvent] = []
    counts = {"geki": 0, "300": 0, "katu": 0, "100": 0, "50": 0, "miss": 0}
    combo = 0
    max_combo = 0
    used_press: set[tuple[int, int]] = set()
    used_release: set[tuple[int, int]] = set()

    for target_time, col, kind in scoring:
        # Pick rising edges for tap/head, falling edges for tail.
        is_tail = kind == "tail"
        sources = releases if is_tail else presses
        used = used_release if is_tail else used_press
        # Tails get the wider release windows (1.5× the tap windows).
        if is_tail:
            w_320 = od_320 * _TAIL_MULTIPLIER
            w_300 = od_300 * _TAIL_MULTIPLIER
            w_200 = od_200 * _TAIL_MULTIPLIER
            w_100 = od_100 * _TAIL_MULTIPLIER
            w_50  = od_50  * _TAIL_MULTIPLIER
        else:
            w_320, w_300, w_200, w_100, w_50 = od_320, od_300, od_200, od_100, od_50
        best_idx = -1
        best_delta = w_50 + 1
        for i, src_t in enumerate(sources[col]):
            if (col, i) in used:
                continue
            d = abs(src_t - target_time)
            if d > w_50:
                continue
            if d < best_delta:
                best_delta = d
                best_idx = i
        if best_idx < 0:
            jud = "miss"
            combo = 0
            offset: float | None = None
        else:
            used.add((col, best_idx))
            d = best_delta
            offset = float(sources[col][best_idx] - target_time)
            if d <= w_320:
                jud = "geki"
            elif d <= w_300:
                jud = "300"
            elif d <= w_200:
                jud = "katu"
            elif d <= w_100:
                jud = "100"
            else:
                jud = "50"
            combo += 1
            max_combo = max(max_combo, combo)
        counts[jud] += 1
        j_events.append(JudgmentEvent(
            time_ms=target_time, column=col, judgment=jud,
            hit_offset_ms=offset,
        ))

    return JudgmentTimeline(
        events=tuple(j_events),
        count_geki=counts["geki"],
        count_300=counts["300"],
        count_katu=counts["katu"],
        count_100=counts["100"],
        count_50=counts["50"],
        count_miss=counts["miss"],
        max_combo=max_combo,
    )


def _falling_edges(
    events: tuple[KeyEvent, ...], key_count: int,
) -> list[list[int]]:
    """Per-column key-release times. Symmetric to _rising_edges."""
    releases: list[list[int]] = [[] for _ in range(key_count)]
    prev = 0
    for e in events:
        gone = prev & ~e.keys_held
        for c in range(key_count):
            if gone & (1 << c):
                releases[c].append(e.time_ms)
        prev = e.keys_held
    return releases


def compute_consumed_times(
    notes: tuple, events: tuple[KeyEvent, ...], key_count: int,
    attempt_window_ms: int = 250,
) -> dict[tuple[int, int], int]:
    """For each note, the timestamp of the first key press in its column
    within ±attempt_window_ms of the note's target time — or absent if the
    note was never attempted.

    Wider than the scoring window (±127 ms) on purpose: this captures the
    "miss because clicked too early/late" case. The renderer uses it to hide
    notes that the player actually tried to hit, regardless of judgment.
    Notes with no entry in the returned map are *true* misses (no press at
    all) and stay visible as they scroll past.
    """
    if not events:
        return {}
    presses = _rising_edges(events, key_count)
    out: dict[tuple[int, int], int] = {}
    for note in notes:
        col = note.column
        target = note.time_ms
        best_press: int | None = None
        best_delta = attempt_window_ms + 1
        for press_t in presses[col]:
            d = abs(press_t - target)
            if d <= attempt_window_ms and d < best_delta:
                best_delta = d
                best_press = press_t
        if best_press is not None:
            out[(col, target)] = best_press
    return out


def _all_miss(notes: tuple) -> JudgmentTimeline:
    events = tuple(
        JudgmentEvent(time_ms=n.time_ms, column=n.column, judgment="miss") for n in notes
    )
    return JudgmentTimeline(
        events=events,
        count_geki=0, count_300=0, count_katu=0, count_100=0, count_50=0,
        count_miss=len(events), max_combo=0,
    )


def _rising_edges(
    events: tuple[KeyEvent, ...], key_count: int,
) -> list[list[int]]:
    presses: list[list[int]] = [[] for _ in range(key_count)]
    prev = 0
    for e in events:
        new_pressed = e.keys_held & ~prev
        for c in range(key_count):
            if new_pressed & (1 << c):
                presses[c].append(e.time_ms)
        prev = e.keys_held
    return presses
