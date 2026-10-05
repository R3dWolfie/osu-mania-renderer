"""Lazer Mania combo from ordered input, independent of recorded statistics.

Source: DrawableNote, DrawableHoldNote{,Head,Tail,Body}, OrderedHitPolicy,
ManiaHitWindows, RulesetInputManager and ScoreProcessor in ppy/osu.
The parent hold and a successful body use Ignore results; only head/tail
results and a failed body affect combo. There are no stable hold ticks.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass

from osu_mania_renderer_v2.beatmap.judgments import _difficulty_range
from osu_mania_renderer_v2.beatmap.models import HoldNote, KeyEvent, Note
from osu_mania_renderer_v2.beatmap.mods import Mod
from osu_mania_renderer_v2.render.stable_mania_combo import StableComboFact, StableComboTimeline


@dataclass(frozen=True)
class LazerComboFact(StableComboFact):
    source: str
    object_time_ms: float
    result: str
    cause: str
    input_frame_index: int | None = None


@dataclass(frozen=True)
class LazerComboTimeline(StableComboTimeline):
    # The prefix fold is source-independent; the builders and facts are not.
    facts: tuple[LazerComboFact, ...]


@dataclass(frozen=True)
class LazerManiaWindows:
    windows: tuple[float, ...]

    @classmethod
    def build(cls, od: float, mods: int = 0, rate: float = 1):
        multiplier = rate
        if mods & Mod.HR:
            multiplier /= 1.4
        elif mods & Mod.EZ:
            multiplier *= 1.4
        ranges = ((22.4, 19.4, 13.9), (64, 49, 34), (97, 82, 67),
                  (127, 112, 97), (151, 136, 121), (188, 173, 158))
        # ManiaHitWindows rounds in AUDIO time, including the half-ms border.
        return cls(tuple(math.floor(_difficulty_range(od, *r) * multiplier) + .5 for r in ranges))

    @property
    def meh(self):
        return self.windows[-2]

    @property
    def miss(self):
        return self.windows[-1]

    def result_for(self, offset: float):
        for result, window in zip(('geki', '300', 'katu', '100', '50', 'miss'), self.windows):
            if abs(offset) <= window:
                return result
        return None


@dataclass
class _Object:
    note: Note | HoldNote
    index: int = 0
    head: str | None = None
    tail: str | None = None
    body: str | None = None
    holding: bool = False

    @property
    def end(self):
        return self.note.end_time_ms if isinstance(self.note, HoldNote) else self.note.time_ms

    @property
    def finished(self):
        return self.head is not None and (not isinstance(self.note, HoldNote) or self.tail is not None)


def build_lazer_combo_timeline(
    notes: tuple[Note | HoldNote, ...], inputs: tuple[KeyEvent, ...], key_count: int,
    *, od: float = 5, mods: int = 0, rate: float = 1,
) -> LazerComboTimeline:
    """Simulate original audio-time objects and frames, then scale only facts.

    No replay header or reconciled judgement is accepted by this interface.
    Stable sorting retains original frame indices and within-timestamp order.
    ReplayStateChangeEvent dispatches releases before presses, by action order.
    Logical non-user deadlines run throughout playback, after input on ties.
    """
    windows = LazerManiaWindows.build(od, mods, rate)
    columns: list[list[_Object]] = [[] for _ in range(key_count)]
    for note in sorted(notes, key=lambda n: n.time_ms):
        column = columns[note.column]
        column.append(_Object(note, index=len(column)))
    starts = [[obj.note.time_ms for obj in column] for column in columns]
    force_indices = [0] * key_count
    holding = [dict() for _ in range(key_count)]
    facts: list[LazerComboFact] = []

    def emit(obj, source, result, now, cause, frame):
        scheduled = obj.end if source == 'tail' else obj.note.time_ms
        facts.append(LazerComboFact(now / rate, 'reset' if result in ('miss', 'combo-break') else 'increment',
                                   source, obj.note.column, scheduled / rate, result, cause, frame))

    def force_earlier(column, target, now, frame):
        # OrderedHitPolicy.enumerateHitObjectsUpTo stops at the FIRST parent
        # whose END >= the successful object's start (including overlapping LN).
        col = column[0].note.column
        index = force_indices[col]
        while index < len(column):
            earlier = column[index]
            if earlier.end >= target:
                break
            index += 1
            if earlier.head is None:
                earlier.head = 'miss'
                emit(earlier, 'head' if isinstance(earlier.note, HoldNote) else 'tap',
                     'miss', now, 'ordered-force-miss', frame)
            if isinstance(earlier.note, HoldNote):
                earlier.holding = False
                holding[col].pop(earlier.index, None)
                if earlier.tail is None:
                    earlier.tail = 'miss'
                    emit(earlier, 'tail', 'miss', now, 'ordered-force-miss', frame)
                if earlier.body is None:
                    earlier.body = 'combo-break'
                    emit(earlier, 'body', 'combo-break', now, 'ordered-force-miss', frame)
        force_indices[col] = index

    def head_result(column, obj, result, now, cause, frame):
        obj.head = result
        emit(obj, 'head' if isinstance(obj.note, HoldNote) else 'tap', result, now, cause, frame)
        if result != 'miss':
            force_earlier(column, obj.note.time_ms, now, frame)

    def tail_result(column, obj, result, now, cause, frame):
        # DrawableHoldNoteTail.GetCappedResult caps a broken/missed hold to Meh.
        if result != 'miss' and (obj.head in (None, 'miss') or obj.body == 'combo-break'):
            result = '50'
        obj.tail = result
        emit(obj, 'tail', result, now, cause, frame)
        if result != 'miss':
            force_earlier(column, obj.end, now, frame)

    def finish_body(obj, now, cause, frame):
        if obj.body is None:
            obj.body = 'ignore-hit' if obj.tail not in (None, 'miss') else 'combo-break'
            if obj.body == 'combo-break':
                emit(obj, 'body', obj.body, now, cause, frame)
        obj.holding = False
        holding[obj.note.column].pop(obj.index, None)

    due = defaultdict(list)
    for col, column in enumerate(columns):
        for obj in column:
            due[math.floor(obj.note.time_ms + windows.meh) + 1].append((col, obj.index))
            if isinstance(obj.note, HoldNote):
                due[math.floor(obj.end + windows.meh * 1.5) + 1].append((col, obj.index))
    deadlines = sorted(due)
    expiry_index = 0

    def expire(now):
        nonlocal expiry_index
        ready = set()
        while expiry_index < len(deadlines) and deadlines[expiry_index] <= now:
            ready.update(due[deadlines[expiry_index]])
            expiry_index += 1
        # The old scan's column/object order remains authoritative on ties.
        for col, index in sorted(ready):
            column = columns[col]
            obj = column[index]
            if obj.head is None and now - obj.note.time_ms > windows.meh:
                head_result(column, obj, 'miss', now, 'timeout', None)
            if isinstance(obj.note, HoldNote) and obj.tail is None and (now - obj.end) / 1.5 > windows.meh:
                tail_result(column, obj, 'miss', now, 'timeout', None)
                finish_body(obj, now, 'tail-finalisation', None)

    held = 0
    ordered_inputs = sorted(enumerate(inputs), key=lambda pair: pair[1].time_ms)
    # Deadline work is indexed once; completed prefixes are never rescanned.
    deadline_index = 0
    for frame, event in ordered_inputs:
        now = event.time_ms
        while deadline_index < len(deadlines) and deadlines[deadline_index] < now:
            expire(deadlines[deadline_index])
            deadline_index += 1
        released, pressed = held & ~event.keys_held, event.keys_held & ~held
        for col, column in enumerate(columns):
            if released & (1 << col):
                for index in sorted(holding[col]):
                    obj = holding[col].get(index)
                    if obj is not None and not obj.finished and obj.holding:
                        result = windows.result_for((now - obj.end) / 1.5)
                        if obj.tail is None and result is not None:
                            tail_result(column, obj, result, now, 'release', frame)
                        finish_body(obj, now, 'release', frame)
        for col, column in enumerate(columns):
            if not pressed & (1 << col):
                continue
            # Every older parent is blocked by the next parent's start.
            # Future objects outside the Miss window cannot consume/resume.
            first = max(0, bisect_right(starts[col], now) - 1)
            stop = bisect_right(starts[col], now + windows.miss)
            for index in range(first, stop):
                obj = column[index]
                if obj.finished:
                    continue
                # CheckHittable applies to the top-level note/hold, not its
                # nested head/tail. Previous windows end at the NEXT start.
                if index + 1 < len(column) and now >= column[index + 1].note.time_ms:
                    continue
                if isinstance(obj.note, HoldNote):
                    if now - obj.end > windows.meh:
                        continue  # cannot resume during late tail lenience
                    if now - obj.note.time_ms >= -windows.miss:
                        obj.holding = True
                        holding[col][index] = obj
                if obj.head is not None:
                    continue  # hold resume does not consume this press
                result = windows.result_for(now - obj.note.time_ms)
                if result is not None:
                    head_result(column, obj, result, now, 'press', frame)
                    break
        held = event.keys_held
        # DrawableHitObject.UpdateAfterChildren runs after input dispatch.
        # Preserve this order for EVERY original same-time replay frame too.
        expire(now)
        if deadline_index < len(deadlines) and deadlines[deadline_index] == now:
            deadline_index += 1
    # Complete the same schedule after an early-ending (or empty) input stream.
    for now in deadlines[deadline_index:]:
        expire(now)
    return LazerComboTimeline.build(tuple(facts))
