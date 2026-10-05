"""Immutable stable presentation facts from notes and replay input, never tallies.

Authority: HitObjectManagerMania.UpdateHitObjects/Hit, HitCircleMania.Hit,
HitCircleManiaLong.HitStart/Hit and ColumnMania_Input.HasLongHitLight.
This reconstructs their presentation calls, independently of score simulation,
header reconciliation and the frozen-head geometry lifecycle.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from osu_mania_renderer_v2.beatmap.models import HoldNote, KeyEvent, Note
from osu_mania_renderer_v2.beatmap.mods import Mod


@dataclass(frozen=True)
class StableManiaWindows:
    geki: float
    great: float
    good: float
    ok: float
    meh: float
    early: float

    @classmethod
    def build(cls, od: float, source_mode: int = 3, mods: int = 0, rate: float = 1):
        # UpdateVariables uses original OD and fixed windows for converts;
        # VariablesCalcu truncates in audio time before conversion to video time.
        if source_mode == 3:
            inverse_od = min(10, max(0, 10 - od))
            raw = (16, *(base + 3 * inverse_od for base in (34, 67, 97, 121, 158)))
        else:
            raw = (16, 34, 67, 97, 121, 158) if round(od) > 4 else (16, 47, 77, 97, 121, 158)
        factor = 1 / 1.4 if mods & Mod.HR else 1.4 if mods & Mod.EZ else 1
        return cls(*(int(value * factor * rate) / rate for value in raw))


@dataclass(frozen=True)
class LegacyHitLightFact:
    """One successful final Hit() that actually calls AddHitLight()."""

    time_ms: float
    column: int
    judgment: str
    kind: str  # tap / ln-final; HitStart and hold breaks cannot enter this tuple


@dataclass(frozen=True)
class LegacyLongLightState:
    animation_age_ms: float = -1
    alpha: float = 0


@dataclass(frozen=True)
class _LongLightChange:
    time_ms: float
    animation_start_ms: float
    start_alpha: float
    enabled: bool

    def alpha_at(self, t_ms: float, rate: float):
        fraction = min(1, max(0, (t_ms - self.time_ms) * rate / (80 if self.enabled else 120)))
        return self.start_alpha + ((1 if self.enabled else 0) - self.start_alpha) * fraction


@dataclass(frozen=True)
class LegacyManiaPresentation:
    normal_hit_facts: tuple[LegacyHitLightFact, ...] = ()
    normal_hit_times: tuple[float, ...] = ()
    long_light_changes: tuple[tuple[_LongLightChange, ...], ...] = ()
    long_light_times: tuple[tuple[float, ...], ...] = ()
    sliding_intervals: tuple[tuple[float, float], ...] = ()
    rate: float = 1

    def long_lights_at(self, t_ms: float):
        result = []
        for changes, times in zip(self.long_light_changes, self.long_light_times):
            index = bisect_right(times, t_ms) - 1
            if index < 0:
                result.append(LegacyLongLightState())
            else:
                change = changes[index]
                result.append(LegacyLongLightState(
                    (t_ms - change.animation_start_ms) * self.rate,
                    change.alpha_at(t_ms, self.rate),
                ))
        return tuple(result)


@dataclass
class _NoteLifecycle:
    note: Note | HoldNote
    pressed: bool = False
    time_press: float = 0
    time_release: float = 0
    finalised: bool = False
    missing: bool = False
    hold_break: bool = False

    def hit(self, now: float, windows: StableManiaWindows):
        note = self.note
        if not isinstance(note, HoldNote):
            self.finalised = True
            if self.pressed:
                diff = abs(self.time_press - note.time_ms)
                for window, tier in zip((windows.geki, windows.great, windows.good, windows.ok, windows.meh),
                                        ("geki", "300", "katu", "100", "50")):
                    if diff <= window:
                        return tier
            return None
        # HitStart returns Ignore. Only Hit() can finalise or report a break.
        self.missing = not self.pressed and now > note.time_ms + windows.meh
        if now < note.time_ms + windows.meh and self.time_press == 0:
            return None
        if now < note.end_time_ms - windows.meh:
            if not self.pressed:
                self.hold_break = True
            return None
        if self.pressed and now < note.end_time_ms + windows.meh:
            return None
        self.finalised = True
        start = (note.end_time_ms - 1 if self.time_press < note.time_ms - windows.meh
                 else note.time_ms + abs(self.time_press - note.time_ms))
        end = now if self.pressed else note.end_time_ms - abs(self.time_release - note.end_time_ms)
        if end < note.end_time_ms - windows.meh:
            return None
        diff_start = abs(start - note.time_ms)
        diff_total = diff_start + abs(end - note.end_time_ms)
        for window, tier in ((windows.geki * 1.2, "geki"), (windows.great * 1.1, "300"),
                             (windows.good, "katu"), (windows.ok, "100")):
            if diff_start <= window and diff_total <= window * 2:
                return "katu" if self.hold_break and tier in ("geki", "300") else tier
        return "50"


def build_legacy_mania_presentation(
    notes: tuple[Note | HoldNote, ...], key_events: tuple[KeyEvent, ...], key_count: int,
    *, od: float = 5, source_mode: int = 3, mods: int = 0, rate: float = 1,
) -> LegacyManiaPresentation:
    """Replay stable's input branches and automatic gates once, in source order.

    Evaluate replay timestamps plus exact source gate crossings, so automatic
    Hit() never relies on a future replay release or a renderer's frame history.
    An active-note window keeps work proportional to input and nearby objects.
    """
    windows = StableManiaWindows.build(od, source_mode, mods, rate)
    ordered = sorted(notes, key=lambda note: note.time_ms)  # stable tie order
    ticks = {}
    for event in key_events:
        ticks.setdefault(event.time_ms, []).append(event.keys_held)
    for note in ordered:
        end = note.end_time_ms if isinstance(note, HoldNote) else note.time_ms
        for boundary in (note.time_ms - windows.early, note.time_ms, end + windows.ok):
            ticks.setdefault(boundary, [])
        if isinstance(note, HoldNote):
            for boundary in (note.time_ms + windows.meh, note.time_ms + windows.meh + 1 / rate,
                             end + windows.meh):
                ticks.setdefault(boundary, [])
    lights = [[] for _ in range(key_count)]
    enabled = [False] * key_count
    facts, intervals = [], []
    sliding_start = None
    held = 0
    next_note = 0
    active = []

    def set_long_light(column, value, now):
        if enabled[column] == value:
            return
        enabled[column] = value
        previous = lights[column][-1] if lights[column] else None
        alpha = previous.alpha_at(now, rate) if previous else 0
        animation_start = now if value else previous.animation_start_ms
        lights[column].append(_LongLightChange(now, animation_start, alpha, value))

    def hit(state, now):
        tier = state.hit(now, windows)
        if tier is not None and not (mods & Mod.PF and tier not in ("geki", "300")):
            note = state.note
            facts.append(LegacyHitLightFact(now, note.column, tier,
                                          "ln-final" if isinstance(note, HoldNote) else "tap"))
            set_long_light(note.column, False, now)

    for now, replay_masks in sorted(ticks.items()):
        while next_note < len(ordered) and ordered[next_note].time_ms - windows.early <= now:
            active.append(_NoteLifecycle(ordered[next_note]))
            next_note += 1
        # At a gate coinciding with a real input frame, process that input in
        # the SAME update. The automatic Hit() branch still precedes input.
        for mask in replay_masks or (held,):
            rising = mask & ~held
            held = mask
            hitted = [False] * key_count
            sliding = False
            for state in active:
                note, column = state.note, state.note.column
                if state.finalised:
                    if note.time_ms > now:
                        hitted[column] = True
                    continue
                if hitted[column]:
                    continue
                end = note.end_time_ms if isinstance(note, HoldNote) else note.time_ms
                if now >= end + windows.ok:
                    hit(state, now)
                    continue
                if rising & (1 << column):
                    if not isinstance(note, HoldNote):
                        state.pressed, state.time_press = True, now
                        hitted[column] = True
                        hit(state, now)
                    elif not state.pressed:
                        state.pressed, state.time_press = True, now
                        hitted[column] = sliding = True
                        set_long_light(column, True, now)
                elif held & (1 << column):
                    if isinstance(note, HoldNote) and state.pressed:
                        hitted[column] = sliding = True
                        set_long_light(column, True, now)
                else:
                    if isinstance(note, HoldNote):
                        if state.pressed:
                            state.pressed, state.time_release = False, now
                            hit(state, now)
                        elif not state.missing:
                            hit(state, now)
                    set_long_light(column, False, now)
            if sliding and sliding_start is None:
                sliding_start = now
            elif not sliding and sliding_start is not None:
                if now > sliding_start:
                    intervals.append((sliding_start, now))
                sliding_start = None
            active = [state for state in active if not state.finalised or state.note.time_ms > now]
    changes = tuple(tuple(column) for column in lights)
    return LegacyManiaPresentation(tuple(facts), tuple(fact.time_ms for fact in facts), changes,
                                   tuple(tuple(change.time_ms for change in column) for column in changes),
                                   tuple(intervals), rate)
