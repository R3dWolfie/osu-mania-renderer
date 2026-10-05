"""Immutable stable HM.ReCalculate / addTransformations presentation data.

Source: HitObjectManagerMania.cs:344-644; SpeedMania.DistanceAt/TimeAt.
R3D supplies its scroll-speed setting, using stable's fixed-speed convention
(RelativeSpeed = Speed * 100 / primaryBPM). It does not recover local osu!
preferences from replay data. All offsets use the renderer's video clock.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from math import floor, isfinite


@dataclass(frozen=True)
class Movement:
    start_ms: float
    end_ms: float
    start_y: float
    end_y: float

    def at(self, time_ms: float) -> float:
        if self.end_ms == self.start_ms:
            return self.end_y
        progress = max(0.0, min(1.0, (time_ms - self.start_ms) /
                                (self.end_ms - self.start_ms)))
        return self.start_y + (self.end_y - self.start_y) * progress


@dataclass(frozen=True, eq=False)
class LegacyManiaTiming:
    offsets: tuple[float, ...]
    velocities: tuple[float, ...]  # 480-reference units / video ms
    distances: tuple[float, ...]
    measures: tuple[float, ...]
    warning_times: tuple[float, ...]
    rate: float = 1.0

    @classmethod
    def build(cls, points, *, first_note_ms: int, last_note_ms: int,
              scroll_speed: float = 20, rate: float = 1):
        """Collapse controls, extend opening SV backwards, then place measures.

        Converted maps already contain only red controls. Initial inherited
        SV before the first note applies from the start of time. Both BPM and
        inherited SV change speed, including stable's 10..1000 raw SV clamp.
        """
        if not points or not any(p.uninherited for p in points):
            return None  # no authoritative timing input: do not invent measures
        def source_time(point):
            return point.time_ms if point.raw_time_ms is None else point.raw_time_ms
        points = sorted(points, key=source_time)
        red = [p for p in points if p.uninherited and p.beat_length_ms > 0]
        if not red:
            return None
        primary = {}
        last_time = last_note_ms
        for p in reversed(red):
            offset = source_time(p) / rate
            if offset > last_time:
                continue
            start = 0 if p is points[0] else offset
            primary[p.beat_length_ms] = primary.get(p.beat_length_ms, 0) + int(last_time - start)
            last_time = offset
        if not primary:
            # No tempo intersects the supplied chart span (also possible in
            # a compositor preview with no note times). Use the caller's
            # untimed fallback, as with absent timing, instead of max({}).
            return None
        primary_length = max(primary, key=primary.get)
        primary_bpm = max(1, round(60000 / primary_length))
        # In video time, rate cancels between RelativeSpeed's EffectiveBPM
        # and the Audio clock. The fixed speed stays fixed under DT/HT.
        speed_factor = 21 * max(1, min(40, round(scroll_speed))) * 100 / primary_bpm
        opening = red[0]
        measure_length = opening.beat_length_ms * opening.time_signature / rate
        pre_start = source_time(points[0]) / rate
        while pre_start >= 0:
            pre_start -= measure_length
        pre_start -= measure_length
        changes, timings = [], []
        last_length = opening.beat_length_ms
        for p in points:
            raw = p.raw_beat_length_ms
            if raw is None:
                raw = p.beat_length_ms if p.uninherited else -100 / p.sv_multiplier
            if not isfinite(raw) or raw == 0:
                continue
            length = raw if raw > 0 else last_length * max(10, min(1000, -raw)) / 100
            offset = source_time(p) / rate
            if len(changes) == 1 and raw < 0 and offset < first_note_ms:
                offset = changes[0][0]
            if changes and offset <= changes[-1][0]:
                changes.pop()
            if not changes or length != changes[-1][1]:
                changes.append((offset, length))
            if raw > 0:
                timings.append((source_time(p) / rate, raw / rate, p.time_signature))
                last_length = raw
        if not changes or not timings:
            return None
        changes[0] = (pre_start, changes[0][1])
        timings[0] = (pre_start, timings[0][1], timings[0][2])
        measures = []
        for i, (offset, length, meter) in enumerate(timings):
            end = timings[i + 1][0] - 1 if i + 1 < len(timings) else last_note_ms + 1
            beat_time = offset
            while beat_time < end:
                measures.append(beat_time)
                beat_time += length * meter
        offsets = tuple(c[0] for c in changes)
        velocities = tuple(speed_factor / c[1] for c in changes)
        distances = [0.0]
        for i in range(1, len(changes)):
            distances.append(distances[-1] + (offsets[i] - offsets[i - 1]) * velocities[i - 1])
        warnings = tuple(first_note_ms - 1000 * i / rate for i in (1, 2, 3)
                         if first_note_ms - 1000 * i / rate > pre_start)
        return cls(offsets, velocities, tuple(distances), tuple(measures), warnings, rate)

    def distance_at(self, time_ms: float) -> float:
        index = max(0, bisect_right(self.offsets, time_ms) - 1)
        return self.distances[index] + (time_ms - self.offsets[index]) * self.velocities[index]

    def time_at_distance(self, distance: float) -> float:
        index = max(0, bisect_right(self.distances, distance) - 1)
        return self.offsets[index] + (distance - self.distances[index]) / self.velocities[index]

    @lru_cache(maxsize=32768)
    def movements(self, note_time: float, hit_position: float,
                  extra_distance: float = 0, extra_time: float = 0) -> tuple[Movement, ...]:
        """Transcribe HM backward/forward walks, including integer endpoints.

        A control at the exact object time uses the PREVIOUS speed first;
        zero-length intervals are skipped. Source animation phase is the
        first sorted transformation's Time1, never scheduled time/approach.
        """
        index = max(0, bisect_left(self.offsets, note_time) - 1)
        result = []
        end_time, end_pos = note_time, hit_position
        final_pos = min(extra_distance, 0)
        for i in range(index, -1, -1):
            start_time = self.offsets[i]
            if start_time == end_time:
                continue
            start_pos = end_pos - self.velocities[i] * (end_time - start_time)
            if start_pos < final_pos:
                start_time = end_time - (end_pos - final_pos) / self.velocities[i]
                start_pos = final_pos
            result.append(Movement(int(start_time * self.rate) / self.rate,
                                   int(end_time * self.rate) / self.rate, start_pos, end_pos))
            if start_pos == final_pos:
                break
            end_time, end_pos = start_time, start_pos
        start_time, start_pos = note_time, hit_position
        final_time = note_time + extra_time
        final_pos = float('inf')
        for i in range(index, len(self.offsets)):
            control_end = self.offsets[i + 1] if i + 1 < len(self.offsets) else float('inf')
            while True:
                end_time = min(control_end, final_time)
                end_pos = start_pos + self.velocities[i] * (end_time - start_time)
                if end_pos > final_pos:
                    end_time = start_time + (final_pos - start_pos) / self.velocities[i]
                    end_pos = final_pos
                if start_time != end_time:
                    result.append(Movement(int(start_time * self.rate) / self.rate,
                                           int(end_time * self.rate) / self.rate, start_pos, end_pos))
                if end_pos == final_pos:
                    return tuple(sorted(result, key=lambda m: (m.start_ms, m.end_ms)))
                start_time, start_pos = end_time, end_pos
                if end_time == final_time:
                    final_pos = end_pos - hit_position + 480 + max(extra_distance, 0)
                    final_time = float('inf')
                    continue  # source REDO, still in this control point
                break
        return tuple(sorted(result, key=lambda m: (m.start_ms, m.end_ms)))

    def first_movement_start(self, note_time: float, hit_position: float,
                             extra_distance: float = 0) -> float:
        return self.movements(note_time, hit_position, extra_distance)[0].start_ms

    def position(self, note_time: float, time_ms: float, hit_position: float,
                 extra_distance: float = 0, extra_time: float = 0) -> float | None:
        for movement in self.movements(note_time, hit_position, extra_distance, extra_time):
            if movement.start_ms <= time_ms <= movement.end_ms:
                return movement.at(time_ms)
        return None


def animation_frame(time_ms: float, start_ms: float, frame_count: int, rate: float = 1) -> int:
    return max(0, floor((time_ms - start_ms) * rate / (1000 / 60))) % max(1, frame_count)
