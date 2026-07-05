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

from dataclasses import dataclass, replace

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


def _difficulty_range(od: float, od0: float, od5: float, od10: float) -> float:
    """osu!framework IBeatmapDifficultyInfo.DifficultyRange: piecewise-linear
    interpolation of a value across OD 0 / 5 / 10."""
    if od > 5:
        return od5 + (od10 - od5) * (od - 5.0) / 5.0
    if od < 5:
        return od0 + (od5 - od0) * od / 5.0
    return od5


def windows_for_od(overall_difficulty: float) -> tuple[float, float, float, float, float]:
    """osu!mania hit windows (ms), ported from lazer ManiaHitWindows.

    DifficultyRange (OD0, OD5, OD10), interpolated by OD:
      PERFECT(320): 22.4, 19.4, 13.9
      GREAT(300)  : 64,   49,   34
      GOOD(200)   : 97,   82,   67
      OK(100)     : 127,  112,  97
      MEH(50)     : 151,  136,  121
    (The old code hard-coded the 320 window at a flat 16ms — ~0.5ms too wide at
    OD8.5 (true 15.55), which over-counted PERFECTs vs what osu recorded.)
    """
    return (
        _difficulty_range(overall_difficulty, 22.4, 19.4, 13.9),
        _difficulty_range(overall_difficulty, 64.0, 49.0, 34.0),
        _difficulty_range(overall_difficulty, 97.0, 82.0, 67.0),
        _difficulty_range(overall_difficulty, 127.0, 112.0, 97.0),
        _difficulty_range(overall_difficulty, 151.0, 136.0, 121.0),
    )


@dataclass(frozen=True)
class JudgmentEvent:
    time_ms: int     # scheduled note time (when it reaches the receptor)
    column: int
    judgment: str    # "geki" | "300" | "katu" | "100" | "50" | "miss"
    hit_offset_ms: float | None = None
    # Signed press-time minus note-time. Negative = early, positive = late.
    # None on a true miss (no press matched). Used to compute UR + avg offset.
    is_tail: bool = False
    # True for a hold-note tail (release) event. Structural fact set by
    # compute_judgments; used by reconcile_to_counts to detect the replay's
    # hold-counting convention (stable ScoreV1 = 1 judgment per hold).
    scoring: bool = True
    # False when this event must NOT contribute to the accuracy/count tally.
    # Set by reconcile_to_counts on tail events for ScoreV1 replays, where
    # the .osr records ONE judgment per hold — the tail stays in the event
    # stream for visuals (hit light / popup / combo) but is excluded from
    # the running counts so the on-screen accuracy lands on the recorded
    # value exactly.


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


def reconcile_to_counts(
    timeline: "JudgmentTimeline",
    geki: int, c300: int, katu: int, c100: int, c50: int, miss: int,
) -> "JudgmentTimeline":
    """Re-label the simulated per-note judgments so the totals EXACTLY match the
    .osr's recorded counts, ordered by the simulated timing quality.

    Our window/pairing sim isn't byte-identical to osu!'s engine, so its tier
    split drifts a little (e.g. a few extra 320s) — which made the live accuracy
    read ~1% high and then snap to the true value at the end. osu! already tells
    us the authoritative final tally, so we trust it: sort every note by how
    cleanly it was hit (smallest |offset| first; unmatched/true-miss last) and
    hand out the recorded number of each tier in order. The running accuracy and
    counts then track correctly the whole way and land exactly on what osu shows
    — no end-of-song patching.

    Hold-note counting convention (detected arithmetically per replay):
      • sum(counts) == len(events)            → ScoreV2 / lazer: head AND tail
        are each a recorded judgment. Relabel every event (old behaviour).
      • sum(counts) == len(non-tail events)   → stable ScoreV1: ONE judgment
        per hold. Relabel taps + hold heads to the recorded counts and mark
        every tail event scoring=False so the tally excludes it (the tail
        keeps its sim judgment purely for visuals).
    Anything else (unexpected mismatch) returns the timeline unchanged.
    """
    events = timeline.events
    targets = [("geki", geki), ("300", c300), ("katu", katu),
               ("100", c100), ("50", c50), ("miss", miss)]
    total = sum(c for _, c in targets)
    if not events:
        return timeline

    if total == len(events):
        # V2/lazer convention — every event is a recorded judgment.
        idxs = list(range(len(events)))
    else:
        heads = [i for i, e in enumerate(events) if not e.is_tail]
        if total != len(heads):
            return timeline
        # ScoreV1 convention — holds recorded once; tails are visual-only.
        idxs = heads

    inf = float("inf")
    order = sorted(
        idxs,
        key=lambda i: (abs(events[i].hit_offset_ms)
                       if events[i].hit_offset_ms is not None else inf),
    )
    new = list(events)
    pos = 0
    for tier, count in targets:
        for _ in range(count):
            i = order[pos]
            pos += 1
            e = events[i]
            if tier == "miss":
                new[i] = replace(e, judgment="miss", hit_offset_ms=None)
            else:
                # phantom hits (a sim-miss promoted to a hit tier to match the
                # recorded tally) get a 0ms offset so downstream press lookups
                # still resolve.
                off = e.hit_offset_ms if e.hit_offset_ms is not None else 0.0
                new[i] = replace(e, judgment=tier, hit_offset_ms=off)
    if len(idxs) != len(events):
        # ScoreV1: exclude tails from the accuracy/count tally.
        for i, e in enumerate(new):
            if e.is_tail:
                new[i] = replace(e, scoring=False)

    return JudgmentTimeline(
        events=tuple(new), count_geki=geki, count_300=c300, count_katu=katu,
        count_100=c100, count_50=c50, count_miss=miss,
        max_combo=timeline.max_combo,
    )


def compute_judgments(
    notes: tuple,
    events: tuple[KeyEvent, ...],
    key_count: int,
    overall_difficulty: float | None = None,
) -> JudgmentTimeline:
    """Pair each scoring event (tap, hold head, hold tail) with the closest
    matching keypress/release and classify by mania timing window.

    Hold notes contribute TWO events (head ↔ press; tail ↔ release) —
    the ScoreV2/lazer convention. Whether the TAIL counts as a recorded
    judgment depends on the replay: stable ScoreV1 records ONE judgment per
    hold, so reconcile_to_counts flags tails scoring=False in that case
    (detected by comparing len(events) against the .osr count total).
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
            hit_offset_ms=offset, is_tail=is_tail,
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
