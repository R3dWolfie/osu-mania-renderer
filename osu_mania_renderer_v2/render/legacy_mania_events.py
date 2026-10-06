"""Immutable stable presentation facts from notes and replay input, never tallies.

Authority: HitObjectManagerMania.UpdateHitObjects/Hit, HitCircleMania.Hit,
HitCircleManiaLong.HitStart/Hit and ColumnMania_Input.HasLongHitLight.
This reconstructs their presentation calls, independently of score simulation,
header reconciliation and the frozen-head geometry lifecycle.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

from osu_mania_renderer_v2.beatmap.models import HoldNote, KeyEvent, Note
from osu_mania_renderer_v2.beatmap.mods import Mod
from osu_mania_renderer_v2.render.stable_mania_combo import StableComboFact


@dataclass(frozen=True)
class StableManiaWindows:
    geki: float
    great: float
    good: float
    ok: float
    meh: float
    early: float

    @classmethod
    def build(cls, od: float, source_mode: int = 3, mods: int = 0, rate: float = 1,
              timeline_rate: float | None = None):
        # UpdateVariables uses original OD and fixed windows for converts;
        # VariablesCalcu truncates in audio time before conversion to video time.
        if source_mode == 3:
            inverse_od = min(10, max(0, 10 - od))
            raw = (16, *(base + 3 * inverse_od for base in (34, 67, 97, 121, 158)))
        else:
            raw = (16, 34, 67, 97, 121, 158) if round(od) > 4 else (16, 47, 77, 97, 121, 158)
        factor = 1 / 1.4 if mods & Mod.HR else 1.4 if mods & Mod.EZ else 1
        input_rate = rate if timeline_rate is None else timeline_rate
        return cls(*(int(value * factor * rate) / input_rate for value in raw))


@dataclass(frozen=True)
class LegacyHitLightFact:
    """One successful final Hit() that actually calls AddHitLight()."""

    time_ms: float
    column: int
    judgment: str
    kind: str  # tap / ln-final; HitStart and hold breaks cannot enter this tuple


@dataclass(frozen=True)
class StableSoundFact:
    """An actual PlaySound/SoundStart call, before score/header reconciliation."""

    time_ms: float
    column: int
    object_time_ms: float  # parent start, including final LN calls
    source: str  # tap / head / final / empty-tap / empty-final


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
    stable_combo_facts: tuple[StableComboFact, ...] = ()
    sound_facts: tuple[StableSoundFact, ...] = ()

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
    last_score_time: float | None = None
    sound_at_end: bool = True  # Native Hold.SoundStart turns this off permanently.

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
            return 'miss'
        # HitStart returns Ignore. Only Hit() can finalise or report a break.
        self.last_score_time = None  # Hit() cancels Holding even when it returns Ignore.
        self.missing = not self.pressed and now > note.time_ms + windows.meh
        if now < note.time_ms + windows.meh and self.time_press == 0:
            return None
        if now < note.end_time_ms - windows.meh:
            if not self.pressed:
                self.hold_break = True
                return 'hold-break'
            return None
        if self.pressed and now < note.end_time_ms + windows.meh:
            return None
        self.finalised = True
        start = (note.end_time_ms - 1 if self.time_press < note.time_ms - windows.meh
                 else note.time_ms + abs(self.time_press - note.time_ms))
        end = now if self.pressed else note.end_time_ms - abs(self.time_release - note.end_time_ms)
        if end < note.end_time_ms - windows.meh:
            return 'miss'
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
    include_combo: bool = False, include_audio: bool = False,
    timeline_rate: float | None = None,
) -> LegacyManiaPresentation:
    """Replay stable's input branches and automatic gates once, in source order.

    Evaluate replay timestamps plus exact source gate crossings, so automatic
    Hit() never relies on a future replay release or a renderer's frame history.
    An active-note window keeps work proportional to input and nearby objects.
    """
    # rate is the mod's audio speed; timeline_rate describes the supplied
    # timestamps. Original audio timestamps use 1, video timestamps use rate.
    input_rate = rate if timeline_rate is None else timeline_rate
    windows = StableManiaWindows.build(od, source_mode, mods, rate, timeline_rate)
    ordered = sorted(notes, key=lambda note: note.time_ms)  # stable tie order
    lifecycles = [_NoteLifecycle(note) for note in ordered]
    sound_columns = [[state for state in lifecycles if state.note.column == column]
                     for column in range(key_count)] if include_audio else []
    sound_starts = [[state.note.time_ms for state in column] for column in sound_columns]
    next_sound = [None] * key_count
    ticks = {}
    for event in key_events:
        ticks.setdefault(event.time_ms, []).append(event.keys_held)
    for note in ordered:
        end = note.end_time_ms if isinstance(note, HoldNote) else note.time_ms
        for boundary in (note.time_ms - windows.early, note.time_ms, end + windows.ok):
            ticks.setdefault(boundary, [])
        if isinstance(note, HoldNote):
            for boundary in (note.time_ms + windows.meh, note.time_ms + windows.meh + 1 / input_rate,
                             end + windows.meh):
                ticks.setdefault(boundary, [])
    lights = [[] for _ in range(key_count)]
    enabled = [False] * key_count
    facts, intervals = [], []
    sliding_start = None
    held = 0
    next_note = 0
    active = []
    combo_facts = []
    sound_facts = []
    pending = list(ticks)
    # Presentation update times are unchanged. Extra logical Holding updates
    # only advance the combo clock; they cannot mutate Round 3.5 presentation.
    heapify(pending)
    queued = set(pending)
    combo_interval = 100 / input_rate  # JudgementMania.ComboIntv uses Clocks.Audio.

    def schedule_holding(state, now):
        if not include_combo or state.last_score_time is None:
            return
        when = max(state.note.time_ms, state.last_score_time + combo_interval)
        # Holding adds once per update. Catch up an overdue early-press clock
        # on the next integer audio millisecond, never on a renderer frame.
        if when <= now:
            when = now + 1 / input_rate
        if when <= state.note.end_time_ms and when not in queued:
            heappush(pending, when)
            queued.add(when)

    def holding(state, now):
        if (include_combo and state.last_score_time is not None
                and state.note.time_ms <= now <= state.note.end_time_ms
                and now >= state.last_score_time + combo_interval):
            state.last_score_time += combo_interval
            combo_facts.append(StableComboFact(now, 'increment', 'ln-hold', state.note.column))
        schedule_holding(state, now)

    def set_long_light(column, value, now):
        if enabled[column] == value:
            return
        enabled[column] = value
        previous = lights[column][-1] if lights[column] else None
        alpha = previous.alpha_at(now, input_rate) if previous else 0
        animation_start = now if value else previous.animation_start_ms
        lights[column].append(_LongLightChange(now, animation_start, alpha, value))

    def hit(state, now):
        tier = state.hit(now, windows)
        positive = tier in ("geki", "300", "katu", "100", "50")
        if include_audio:
            note = state.note
            # HitCircleMania plays whenever Pressed, even for a consuming Miss.
            # Converted HitCircleManiaLong plays a positive final result. Native
            # HitCircleManiaHold.SoundStart disables SoundAtEnd instead.
            if not isinstance(note, HoldNote) and state.pressed:
                sound_facts.append(StableSoundFact(now, note.column, note.time_ms, 'tap'))
            elif isinstance(note, HoldNote) and positive and source_mode != 3:
                sound_facts.append(StableSoundFact(now, note.column, note.time_ms, 'final'))
        positive = positive and not (mods & Mod.PF and tier not in ("geki", "300"))
        if include_combo and tier is not None:
            source = ('ln-break' if tier == 'hold-break' else 'ln-final'
                      if isinstance(state.note, HoldNote) else 'tap' if positive else 'miss')
            combo_facts.append(StableComboFact(now, 'increment' if positive else 'reset',
                                              source, state.note.column))
        if positive:
            note = state.note
            facts.append(LegacyHitLightFact(now, note.column, tier,
                                          "ln-final" if isinstance(note, HoldNote) else "tap"))
            set_long_light(note.column, False, now)

    while pending:
        now = heappop(pending)
        if now not in ticks:
            # No replay edge occurs here. Keep source object order, including
            # negative Hit() calls on other lanes, before/after Holding().
            # Presentation toggles still belong to the existing update stream.
            hitted = [False] * key_count
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
                if isinstance(note, HoldNote) and state.pressed and held & (1 << column):
                    hitted[column] = True
                    next_sound[column] = None
                    holding(state, now)
                elif isinstance(note, HoldNote) and not (held & (1 << column)) and not state.missing:
                    hit(state, now)
            continue
        replay_masks = ticks[now]
        while next_note < len(ordered) and ordered[next_note].time_ms - windows.early <= now:
            active.append(lifecycles[next_note])
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
                        next_sound[column] = None
                        hit(state, now)
                    elif not state.pressed:
                        state.pressed, state.time_press = True, now
                        state.last_score_time = now
                        next_sound[column] = None
                        if source_mode == 3:
                            state.sound_at_end = False
                        if include_audio:
                            sound_facts.append(StableSoundFact(now, column, note.time_ms, 'head'))
                        hitted[column] = sliding = True
                        set_long_light(column, True, now)
                        schedule_holding(state, now)
                elif held & (1 << column):
                    if isinstance(note, HoldNote) and state.pressed:
                        hitted[column] = sliding = True
                        next_sound[column] = None
                        holding(state, now)
                        set_long_light(column, True, now)
                else:
                    if isinstance(note, HoldNote):
                        if state.pressed:
                            state.pressed, state.time_release = False, now
                            hit(state, now)
                        elif not state.missing:
                            hit(state, now)
                    set_long_light(column, False, now)
            if include_audio:
                # Stable runs this AFTER the object loop. The cached reference
                # survives automatic misses, releases and finalisation; only
                # the three real press/holding branches above clear it.
                for column in range(key_count):
                    if hitted[column] or not rising & (1 << column):
                        continue
                    if next_sound[column] is None:
                        index = bisect_right(sound_starts[column], now)
                        if index < len(sound_columns[column]):
                            next_sound[column] = sound_columns[column][index]
                    cached = next_sound[column]
                    if cached is None:
                        continue
                    long = isinstance(cached.note, HoldNote)
                    if long and not cached.sound_at_end:
                        continue
                    sound_facts.append(StableSoundFact(now, column, cached.note.time_ms,
                                                       'empty-final' if long else 'empty-tap'))
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
                                   tuple(intervals), input_rate, tuple(combo_facts), tuple(sound_facts))
