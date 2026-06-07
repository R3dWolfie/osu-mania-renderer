"""Per-frame scene state for the renderer. Pure function of beatmap + replay + t."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from osu_mania_renderer_v2.models import HoldNote, KeyEvent, VisualMods


@dataclass(frozen=True)
class VisibleNote:
    column: int
    is_hold: bool
    y_fraction: float          # 0.0 = top of playfield, 1.0 = at receptor
    head_y_fraction: float     # alias of y_fraction for taps
    tail_y_fraction: float     # equals head for taps, separate for holds
    # Note's scheduled hit time (ms). Used by per-note animation phase
    # so each note's animation starts from frame 0 when it spawns,
    # rather than all notes in a column animating in sync.
    time_ms: int = 0


@dataclass(frozen=True)
class JudgmentPopup:
    column: int
    judgment: str       # "geki" | "300" | "katu" | "100" | "50" | "miss"
    age_ms: int         # 0 = just hit, fades over 600ms


@dataclass(frozen=True)
class SceneState:
    t_ms: int
    visible_notes: tuple[VisibleNote, ...]
    keys_held: tuple[bool, ...]
    visual_mods: VisualMods
    active_judgments: tuple[JudgmentPopup, ...] = ()
    score: int = 0
    combo: int = 0
    max_combo: int = 0
    accuracy: float = 100.0
    # Display-ordered mod pill labels for this replay, e.g. ("4K", "HD", "DT").
    # First entry is the key-count badge; rest are gameplay mods.
    mod_acronyms: tuple[str, ...] = ()
    # Post-game results card. When True, the renderer overlays the final
    # grade / score / accuracy / judgment breakdown on top of a dimmed,
    # gameplay-emptied playfield (notes have all scrolled past at this point).
    # 0 = not visible; 0.0–1.0 = fade-in progress.
    results_opacity: float = 0.0
    grade: str = "D"
    judgment_counts: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    # ordered: (geki=320, 300, katu=200, 100, 50, miss)
    # Hit offsets (press minus note time, in ms) for the LAST ~60 hits.
    # Drives the on-screen unstable rate bar — gameplay HUD shows the
    # distribution + the average offset. None = no press for that note.
    recent_offsets: tuple[float, ...] = ()
    # Average hit offset across the play so far, in ms (positive = late).
    avg_hit_offset_ms: float = 0.0
    # Unstable rate (10 × stddev of all signed hit offsets so far).
    unstable_rate: float = 0.0
    # PP values (player's and the map's max-FC PP). Both 0 until computed.
    pp: float = 0.0
    max_pp: float = 0.0
    # Black-fade opacity layered over the whole frame. 1.0 = pure black,
    # 0.0 = no overlay. Used to fade in at song start and fade out into the
    # results card.
    fade_to_black: float = 0.0
    # Per-column hit-light state: age in ms (0..LIGHT_DURATION_MS) since the
    # most recent successful hit landed there. -1 ⇒ no recent hit. Drives a
    # judgment-coloured flash + glow at the receptor row.
    hit_light_age_ms: tuple[int, ...] = ()
    # Per-column hit judgment for the same recent hit (for colour-coding the
    # light). Empty string ⇒ no recent hit.
    hit_light_judgment: tuple[str, ...] = ()
    # Combo pop animation: ms since the last combo increment (0 = just now).
    # The renderer scales the combo number briefly then settles back.
    combo_age_ms: int = 9999
    # Smoothed (tweened) versions of score + accuracy so the on-screen
    # counter rolls up rather than snapping per-frame.
    score_smoothed: int = 0
    accuracy_smoothed: float = 100.0
    # Song-progress fraction in 0..1 (used by the thin progress bar at the
    # very top of the screen).
    song_progress: float = 0.0
    # Real osu!mania HP in 0..1, derived from per-judgment HP deltas rather
    # than the older combo/max_combo proxy.
    hp: float = 1.0
    # Per-column most recent hit's offset (in ms) for the floating "+8 ms"
    # popups near each receptor. None ⇒ no recent hit.
    hit_offset_per_col: tuple[float, ...] = ()
    # Whether we're currently inside a kiai-time section (chorus highlight).
    is_kiai: bool = False
    # Per-column UR (only meaningful on the results card; gameplay uses
    # the playfield-wide scene.unstable_rate). Computed once at end.
    per_column_ur: tuple[float, ...] = ()
    # Per-column age (ms) since the most recent key-down. Drives a bright
    # stage-light strip in each column that fades over ~200 ms.
    key_press_age_ms: tuple[int, ...] = ()
    # Per-column cumulative key-press count up to t_ms (rising edges). Drives
    # the bottom-right key counter (lazer's KeyCounterDisplay).
    key_press_counts: tuple[int, ...] = ()
    # Age (ms) since the most recent combo-break that was large enough to
    # trigger the shake/flash. 9999 = no recent break.
    miss_break_age_ms: int = 9999


def snapshot(
    notes: tuple,
    key_events: tuple[KeyEvent, ...],
    t_ms: int,
    key_count: int,
    approach_ms: int,
    visual_mods: VisualMods,
    consumed_times: dict[tuple[int, int], int] | None = None,
    sv_for_note: dict[tuple[int, int], float] | None = None,
    # SV integration support — when both are provided, scene uses
    # piecewise-constant SV integration for note positioning. This is
    # the lazer-faithful path; if omitted (legacy callers), falls back
    # to the per-note point-sample `sv_for_note` lookup.
    timing_points: tuple | None = None,
    sv_table: tuple[float, ...] | None = None,
    # Optional perf hints — precomputed once per render and passed in
    # so the per-frame loop doesn't rescan all notes in the beatmap.
    # `note_times` is a sequence of n.time_ms aligned with `notes`;
    # `max_hold_dur_ms` is the longest hold body in the map so we know
    # how far back to look for still-active holds.
    note_times: tuple[int, ...] | None = None,
    max_hold_dur_ms: int = 0,
) -> SceneState:
    """Return what's on screen at time t_ms.

    Notes whose start time is within [t_ms, t_ms + approach_ms] are scrolling
    in the playfield. Holds are also visible while their body covers t_ms.
    y_fraction = (t_ms - (note.time_ms - approach_ms)) / approach_ms.

    `consumed_times` maps (column, note_time_ms) → press timestamp of the
    player's attempt on that note. Tap notes are hidden as soon as t_ms
    crosses their consumed timestamp (the player tried to hit it). True
    misses (no entry in the map) keep scrolling past the receptor so the
    viewer can see what slipped through.
    """
    visible: list[VisibleNote] = []
    consumed = consumed_times or {}
    svs = sv_for_note or {}
    # SV-integration path: when both `timing_points` and `sv_table` are
    # given we compute note positions from integrated distance, which
    # is the correct lazer-mania semantics. Without them, fall back to
    # the legacy point-sample lookup (still wired into render.py for
    # callers that haven't migrated).
    use_integration = (
        timing_points is not None and sv_table is not None
        and len(timing_points) > 0
    )
    # Cache the current frame's cumulative-distance — it doesn't change
    # within a frame, so compute once and reuse for every note.
    if use_integration:
        from osu_mania_renderer_v2.beatmap import sv_distance_at as _sv_at
        current_cum = _sv_at(t_ms, timing_points, sv_table)
    else:
        _sv_at = None
        current_cum = 0.0
    # The horizon needs to allow for SV sections where notes may be
    # CLOSER (smaller SV → larger time window for the same playfield
    # distance). For integration path, use the smallest SV in the
    # table (≥0.05 by parser clamp) as the safe floor.
    if use_integration:
        min_sv = min(tp.sv_multiplier for tp in timing_points) if timing_points else 1.0
        horizon = t_ms + int(approach_ms / max(0.05, min_sv))
    else:
        horizon = t_ms + int(approach_ms / 0.25)
    # Un-attempted misses stay visible past the receptor briefly so the
    # viewer can see them slip through. Anything the player tried to hit
    # vanishes the instant it crosses the receptor — feedback is provided
    # by the judgment popup instead.
    MISS_GRACE_MS = 250
    # Window the iteration to just notes that could be on screen. Notes
    # are time-sorted by start. Safe lower bound: any note whose start
    # time is older than (t_ms - longest_hold_in_map - grace) is fully
    # past the screen. Upper bound: anything past the horizon is sorted
    # later and can't be visible yet. Without these bounds a long song
    # rescans the entire note list every frame (14M+ pure-Python checks
    # for a busy map at 30fps).
    if note_times is not None:
        import bisect as _bisect
        lower_t = t_ms - max(max_hold_dur_ms, 0) - MISS_GRACE_MS
        start_idx = _bisect.bisect_left(note_times, lower_t)
    else:
        start_idx = 0
    for n in notes[start_idx:]:
        if n.time_ms > horizon:
            break
        if isinstance(n, HoldNote):
            # Hold disappears the moment its tail reaches the receptor.
            # Anything else (body extending below the receptor row) is the
            # cause of the "notes don't disappear" complaint, since holds
            # are most of the playfield in this map.
            if t_ms >= n.end_time_ms:
                continue
            head_consumed = consumed.get((n.column, n.time_ms))
            if use_integration:
                head_y = _y_integrated(
                    n.time_ms, current_cum, approach_ms,
                    timing_points, sv_table,
                )
                tail_y = _y_integrated(
                    n.end_time_ms, current_cum, approach_ms,
                    timing_points, sv_table,
                )
            else:
                sv = svs.get((n.column, n.time_ms), 1.0)
                head_y = _y(n.time_ms, t_ms, approach_ms, sv)
                tail_y = _y(n.end_time_ms, t_ms, approach_ms, sv)
            # Once the head was consumed AND has reached the receptor,
            # clamp the visual head to the receptor row so the body is
            # anchored at the bottom instead of sliding past. The renderer
            # also hides the head circle when head_y_fraction >= 1.
            if head_consumed is not None and t_ms >= n.time_ms:
                head_y = 1.0
            visible.append(VisibleNote(
                column=n.column, is_hold=True,
                y_fraction=head_y, head_y_fraction=head_y, tail_y_fraction=tail_y,
                time_ms=n.time_ms,
            ))
        else:
            attempt_t = consumed.get((n.column, n.time_ms))
            if attempt_t is not None:
                # Player tried to hit it — disappear AT the receptor (when
                # t_ms first crosses the note's scheduled time). The press
                # timing is reflected by the judgment popup, not the note.
                if t_ms >= n.time_ms:
                    continue
            else:
                # True miss — scroll past for a moment so it's visible.
                if n.time_ms < t_ms - MISS_GRACE_MS:
                    continue
            if use_integration:
                y = _y_integrated(
                    n.time_ms, current_cum, approach_ms,
                    timing_points, sv_table,
                )
            else:
                sv = svs.get((n.column, n.time_ms), 1.0)
                y = _y(n.time_ms, t_ms, approach_ms, sv)
            visible.append(VisibleNote(
                column=n.column, is_hold=False,
                y_fraction=y, head_y_fraction=y, tail_y_fraction=y,
                time_ms=n.time_ms,
            ))

    keys_held = _keys_held_at(key_events, t_ms, key_count)
    return SceneState(
        t_ms=t_ms,
        visible_notes=tuple(visible),
        keys_held=keys_held,
        visual_mods=visual_mods,
    )


def _y(note_time: int, t_ms: int, approach_ms: int,
       sv: float = 1.0) -> float:
    """LEGACY point-sample SV. Position is computed from the SV at the
    note's own time, ignoring SV variation between t_ms and note_time.
    Correct only when SV is constant for the whole map; on charts with
    SV section changes it causes notes to warp at boundaries.

    Kept for back-compat callers that haven't migrated to the
    integration path. New callers should pass `timing_points` +
    `sv_table` to `snapshot()` and use `_y_integrated()`."""
    return 1.0 - (note_time - t_ms) * sv / approach_ms


def _y_integrated(
    note_time: int,
    current_cum: float,
    approach_ms: int,
    timing_points: tuple,
    sv_table: tuple[float, ...],
) -> float:
    """Lazer-faithful: note y-fraction = 1 - (cumDist(note) -
    cumDist(now)) / approach_ms, where cumDist is the piecewise-
    constant integral of SV over time. Notes that cross SV boundaries
    naturally accelerate/decelerate without warping.

    `current_cum` is `sv_distance_at(t_ms, ...)` — computed ONCE per
    frame by the caller (cached for every note in the frame).

    `approach_ms` here means "playfield-distance to cover from spawn
    to receptor at SV=1." So at SV=1 the formula degenerates to the
    legacy one and rendering is identical."""
    from osu_mania_renderer_v2.beatmap import sv_distance_at
    note_cum = sv_distance_at(note_time, timing_points, sv_table)
    return 1.0 - (note_cum - current_cum) / approach_ms


def _keys_held_at(
    events: tuple[KeyEvent, ...], t_ms: int, key_count: int,
) -> tuple[bool, ...]:
    if not events:
        return tuple(False for _ in range(key_count))
    times = [e.time_ms for e in events]
    idx = bisect_right(times, t_ms) - 1
    if idx < 0:
        return tuple(False for _ in range(key_count))
    mask = events[idx].keys_held
    return tuple(bool(mask & (1 << c)) for c in range(key_count))
