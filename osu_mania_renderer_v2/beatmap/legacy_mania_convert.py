"""Faithful port of osu!lazer's *legacy* osu!standard -> osu!mania beatmap
conversion (the `ManiaBeatmapConverter` + `Legacy` pattern generators).

This replaces the old heuristic slider-stream generator. It reproduces the
exact note COUNT, TIMES and hold-vs-tap decision that lazer makes for every
standard object, by porting:

  * ``LegacyRandom`` (osu-stable ``FastRandom`` LCG) -- bit-exact.
  * ``ManiaBeatmapConverter`` seed + per-object dispatch + density/lastPattern
    /lastStair bookkeeping.
  * ``LegacyPatternGenerator`` (conversionDifficulty, GetRandomNoteCount,
    FindAvailableColumn, GetRandomColumn).
  * ``SliderPatternGenerator``, ``HitCirclePatternGenerator``,
    ``SpinnerPatternGenerator`` (the actual pattern decisions).

CRITICAL: the RNG is a single shared sequential stream consumed by every
generator across every object, in beatmap order. Even though the renderer
ultimately discards lazer's column CHOICE (it re-derives columns from the
replay's key presses), the column-selection RNG draws MUST still be performed
here in the right order, because they advance the stream and therefore change
the note COUNT of every later object. So this is a full port including column
placement; the caller drops lazer's columns afterwards.

Sources (fetched on the build machine under /tmp/lazer/):
  LegacyRandom.cs, ManiaBeatmapConverter.cs, LegacyPatternGenerator.cs,
  SliderPatternGenerator.cs, HitCirclePatternGenerator.cs,
  SpinnerPatternGenerator.cs, PassThroughPatternGenerator.cs, Pattern.cs,
  PatternType.cs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ─────────────────────────── PatternType flags ───────────────────────────
# Mirrors PatternType.cs ([Flags] enum).
PT_NONE = 0
PT_FORCE_STACK = 1 << 0
PT_FORCE_NOT_STACK = 1 << 1
PT_KEEP_SINGLE = 1 << 2
PT_LOW_PROBABILITY = 1 << 3
PT_ALTERNATE = 1 << 4
PT_FORCE_SIG_SLIDER = 1 << 5
PT_FORCE_NOT_SLIDER = 1 << 6
PT_GATHERED = 1 << 7
PT_MIRROR = 1 << 8
PT_REVERSE = 1 << 9
PT_CYCLE = 1 << 10
PT_STAIR = 1 << 11
PT_REVERSE_STAIR = 1 << 12


# ─────────────────────────────── HitSound bits ───────────────────────────
HIT_NORMAL = 0
HIT_WHISTLE = 1 << 1   # 2
HIT_FINISH = 1 << 2    # 4
HIT_CLAP = 1 << 3      # 8


# ─────────────────────────────── LegacyRandom ────────────────────────────
class LegacyRandom:
    """Bit-exact port of osu.Game.Utils.LegacyRandom (osu-stable FastRandom).

    A PRNG specified at http://heliosphan.org/fastrandom.html. All state is
    32-bit unsigned; arithmetic wraps mod 2**32. Shifts are the C# unsigned
    shifts (logical), which match Python's ``>>``/masked ``<<`` on
    non-negative ints.
    """

    _Y = 842502087
    _Z = 3579807591
    _W = 273326509
    _MASK32 = 0xFFFFFFFF
    _INT_MASK = 0x7FFFFFFF
    # int_to_real = 1.0 / (int.MaxValue + 1.0) = 1 / 2147483648
    _INT_TO_REAL = 1.0 / 2147483648.0

    def __init__(self, seed: int):
        self.x = seed & self._MASK32
        self.y = self._Y
        self.z = self._Z
        self.w = self._W
        self._bit_buffer = 0
        self._bit_index = 32

    def next_uint(self) -> int:
        t = (self.x ^ ((self.x << 11) & self._MASK32)) & self._MASK32
        self.x = self.y
        self.y = self.z
        self.z = self.w
        self.w = (self.w ^ (self.w >> 19) ^ t ^ (t >> 8)) & self._MASK32
        return self.w

    def next(self) -> int:
        """Random int in [0, int.MaxValue)."""
        return self._INT_MASK & self.next_uint()

    def next_upper(self, upper_bound: float) -> int:
        """C# ``Next(int upperBound)`` -> (int)(NextDouble() * upperBound)."""
        return int(self.next_double() * upper_bound)

    def next_range(self, lower_bound: float, upper_bound: float) -> int:
        """C# ``Next(lower, upper)`` -> (int)(lower + NextDouble()*(upper-lower)).

        Both the int- and double-overload truncate toward zero via the C#
        ``(int)`` cast; Python ``int()`` truncates toward zero too.
        """
        return int(lower_bound + self.next_double() * (upper_bound - lower_bound))

    def next_double(self) -> float:
        return self._INT_TO_REAL * self.next()

    def next_bool(self) -> bool:
        if self._bit_index == 32:
            self._bit_buffer = self.next_uint()
            self._bit_index = 1
            return (self._bit_buffer & 1) == 1
        self._bit_index += 1
        self._bit_buffer >>= 1
        return (self._bit_buffer & 1) == 1


# ─────────────────────────────── Hit objects ─────────────────────────────
@dataclass(frozen=True)
class StdObject:
    """A parsed standard hit object the converter needs."""
    start_time: float
    x: float
    y: float
    kind: str                 # "circle" | "slider" | "spinner"
    end_time: float           # for spinner/hold; == start_time for circle
    hit_sound: int            # base hitSound bitfield
    # slider-specific
    span_count: int = 1
    pixel_length: float = 0.0
    slider_velocity: float = 1.0  # per-object SV multiplier (green TP), 1.0 default
    has_kiai: bool = False        # EffectPoint.KiaiMode at start
    # node (edge) hitsounds: list of bitfields, one per slider node
    node_hit_sounds: tuple[int, ...] = ()


@dataclass
class ManiaObj:
    """Produced mania note. ``end_time`` > ``start_time`` ⇒ hold note."""
    column: int
    start_time: int
    end_time: int            # == start_time for a tap
    hit_sound: int = 0

    @property
    def is_hold(self) -> bool:
        return self.end_time != self.start_time


# ─────────────────────────────── Pattern ─────────────────────────────────
class Pattern:
    """Port of Pattern.cs."""

    def __init__(self) -> None:
        self.hit_objects: list[ManiaObj] = []
        self._columns: set[int] = set()

    def column_has_object(self, column: int) -> bool:
        return column in self._columns

    @property
    def column_with_objects(self) -> int:
        return len(self._columns)

    def add_object(self, obj: ManiaObj) -> None:
        self.hit_objects.append(obj)
        self._columns.add(obj.column)

    def add_pattern(self, other: "Pattern") -> None:
        for h in other.hit_objects:
            self.hit_objects.append(h)
            self._columns.add(h.column)

    def clear(self) -> None:
        self.hit_objects.clear()
        self._columns.clear()


class NotEnoughColumnsException(Exception):
    pass


# ───────────────────────── Base legacy generator ─────────────────────────
class LegacyPatternGenerator:
    def __init__(
        self,
        random: LegacyRandom,
        hit_object: StdObject,
        total_columns: int,
        previous_pattern: Pattern,
        conversion_difficulty: float,
    ) -> None:
        self.random = random
        self.hit_object = hit_object
        self.total_columns = total_columns
        self.previous_pattern = previous_pattern
        self.random_start = 1 if total_columns == 8 else 0
        self._conversion_difficulty = conversion_difficulty

    # ConversionDifficulty is computed once for the whole beatmap and passed
    # in (lazer lazily memoises it the same way; the value is identical for
    # every generator).
    @property
    def conversion_difficulty(self) -> float:
        return self._conversion_difficulty

    def get_column(self, position: float, allow_special: bool = False) -> int:
        if allow_special and self.total_columns == 8:
            local_x_divisor = 512.0 / 7
            return max(0, min(6, int(math.floor(position / local_x_divisor)))) + 1
        local_x_divisor = 512.0 / self.total_columns
        return max(0, min(self.total_columns - 1,
                          int(math.floor(position / local_x_divisor))))

    def get_random_note_count(
        self, p2: float, p3: float, p4: float = 0.0, p5: float = 0.0, p6: float = 0.0,
    ) -> int:
        val = self.random.next_double()
        if val >= 1 - p6:
            return 6
        if val >= 1 - p5:
            return 5
        if val >= 1 - p4:
            return 4
        if val >= 1 - p3:
            return 3
        return 2 if val >= 1 - p2 else 1

    def get_random_column(self, lower_bound=None, upper_bound=None) -> int:
        lb = self.random_start if lower_bound is None else lower_bound
        ub = self.total_columns if upper_bound is None else upper_bound
        return self.random.next_range(lb, ub)

    def find_available_column(
        self,
        initial_column: int,
        *patterns: Pattern,
        lower_bound: int | None = None,
        upper_bound: int | None = None,
        next_column=None,
        validation=None,
    ) -> int:
        lb = self.random_start if lower_bound is None else lower_bound
        ub = self.total_columns if upper_bound is None else upper_bound
        if next_column is None:
            def next_column(_):
                return self.get_random_column(lb, ub)

        def is_valid(column: int) -> bool:
            if validation is not None and validation(column) is False:
                return False
            for p in patterns:
                if p.column_has_object(column):
                    return False
            return True

        if is_valid(initial_column):
            return initial_column

        has_valid_columns = False
        i = lb
        while i < ub:
            has_valid_columns = is_valid(i)
            if has_valid_columns:
                break
            i += 1
        if not has_valid_columns:
            raise NotEnoughColumnsException()

        col = initial_column
        while True:
            col = next_column(col)
            if is_valid(col):
                break
        return col


# ──────────────────── PassThrough (mania-source only) ────────────────────
# Not reachable for std->mania (IsForCurrentRuleset is False) but kept for
# completeness / spinners parsed as holds in some odd maps.


# ─────────────────────── HitCircle pattern generator ─────────────────────
class HitCirclePatternGenerator(LegacyPatternGenerator):
    def __init__(
        self, random, hit_object, total_columns, previous_pattern,
        conversion_difficulty, previous_time, previous_position, density,
        last_stair, beat_length_at_start,
    ):
        super().__init__(random, hit_object, total_columns, previous_pattern,
                         conversion_difficulty)
        self.stair_type = last_stair
        self.convert_type = PT_NONE

        px, py = hit_object.x, hit_object.y
        ppx, ppy = previous_position
        position_separation = math.hypot(px - ppx, py - ppy)
        time_separation = hit_object.start_time - previous_time
        beat_length = beat_length_at_start

        if time_separation <= 80:
            self.convert_type |= PT_FORCE_NOT_STACK | PT_KEEP_SINGLE
        elif time_separation <= 95:
            self.convert_type |= PT_FORCE_NOT_STACK | PT_KEEP_SINGLE | last_stair
        elif time_separation <= 105:
            self.convert_type |= PT_FORCE_NOT_STACK | PT_LOW_PROBABILITY
        elif time_separation <= 125:
            self.convert_type |= PT_FORCE_NOT_STACK
        elif time_separation <= 135 and position_separation < 20:
            self.convert_type |= PT_CYCLE | PT_KEEP_SINGLE
        elif time_separation <= 150 and position_separation < 20:
            self.convert_type |= PT_FORCE_STACK | PT_LOW_PROBABILITY
        elif position_separation < 20 and density >= beat_length / 2.5:
            self.convert_type |= PT_REVERSE | PT_LOW_PROBABILITY
        elif density < beat_length / 2.5 or hit_object.has_kiai:
            pass  # high density
        else:
            self.convert_type |= PT_LOW_PROBABILITY

        if not (self.convert_type & PT_KEEP_SINGLE):
            if (hit_object.hit_sound & HIT_FINISH) and total_columns != 8:
                self.convert_type |= PT_MIRROR
            elif hit_object.hit_sound & HIT_CLAP:
                self.convert_type |= PT_GATHERED

    @property
    def _has_special_column(self) -> bool:
        return bool(self.hit_object.hit_sound & HIT_CLAP) and \
            bool(self.hit_object.hit_sound & HIT_FINISH)

    def generate(self) -> Pattern:
        p = self._generate_core()
        for obj in p.hit_objects:
            if (self.convert_type & PT_STAIR) and obj.column == self.total_columns - 1:
                self.stair_type = PT_REVERSE_STAIR
            if (self.convert_type & PT_REVERSE_STAIR) and obj.column == self.random_start:
                self.stair_type = PT_STAIR
        return p

    def _generate_core(self) -> Pattern:
        pattern = Pattern()
        tc = self.total_columns
        rs = self.random_start

        if tc == 1:
            self._add(pattern, 0)
            return pattern

        last_column = (self.previous_pattern.hit_objects[0].column
                       if self.previous_pattern.hit_objects else 0)

        if (self.convert_type & PT_REVERSE) and self.previous_pattern.hit_objects:
            for i in range(rs, tc):
                if self.previous_pattern.column_has_object(i):
                    self._add(pattern, rs + tc - i - 1)
            return pattern

        if ((self.convert_type & PT_CYCLE)
                and len(self.previous_pattern.hit_objects) == 1
                and (tc != 8 or last_column != 0)
                and (tc % 2 == 0 or last_column != tc // 2)):
            column = rs + tc - last_column - 1
            self._add(pattern, column)
            return pattern

        if (self.convert_type & PT_FORCE_STACK) and self.previous_pattern.hit_objects:
            for i in range(rs, tc):
                if self.previous_pattern.column_has_object(i):
                    self._add(pattern, i)
            return pattern

        if len(self.previous_pattern.hit_objects) == 1:
            if self.convert_type & PT_STAIR:
                target_column = last_column + 1
                if target_column == tc:
                    target_column = rs
                self._add(pattern, target_column)
                return pattern
            if self.convert_type & PT_REVERSE_STAIR:
                target_column = last_column - 1
                if target_column == rs - 1:
                    target_column = tc - 1
                self._add(pattern, target_column)
                return pattern

        if self.convert_type & PT_KEEP_SINGLE:
            return self._generate_random_notes(1)

        if self.convert_type & PT_MIRROR:
            if self.conversion_difficulty > 6.5:
                return self._generate_random_pattern_with_mirrored(0.12, 0.38, 0.12)
            if self.conversion_difficulty > 4:
                return self._generate_random_pattern_with_mirrored(0.12, 0.17, 0)
            return self._generate_random_pattern_with_mirrored(0.12, 0, 0)

        if self.conversion_difficulty > 6.5:
            if self.convert_type & PT_LOW_PROBABILITY:
                return self._generate_random_pattern(0.78, 0.42, 0, 0)
            return self._generate_random_pattern(1, 0.62, 0, 0)

        if self.conversion_difficulty > 4:
            if self.convert_type & PT_LOW_PROBABILITY:
                return self._generate_random_pattern(0.35, 0.08, 0, 0)
            return self._generate_random_pattern(0.52, 0.15, 0, 0)

        if self.conversion_difficulty > 2:
            if self.convert_type & PT_LOW_PROBABILITY:
                return self._generate_random_pattern(0.18, 0, 0, 0)
            return self._generate_random_pattern(0.45, 0, 0, 0)

        return self._generate_random_pattern(0, 0, 0, 0)

    def _generate_random_notes(self, note_count: int) -> Pattern:
        pattern = Pattern()
        tc = self.total_columns
        rs = self.random_start
        allow_stacking = not (self.convert_type & PT_FORCE_NOT_STACK)

        if not allow_stacking:
            note_count = min(note_count,
                             tc - rs - self.previous_pattern.column_with_objects)

        next_column = self.get_column(self.hit_object.x, True)

        def get_next_column(last):
            if self.convert_type & PT_GATHERED:
                last += 1
                if last == tc:
                    last = rs
            else:
                last = self.get_random_column()
            return last

        for _ in range(note_count):
            if allow_stacking:
                next_column = self.find_available_column(
                    next_column, pattern, next_column=get_next_column)
            else:
                next_column = self.find_available_column(
                    next_column, pattern, self.previous_pattern,
                    next_column=get_next_column)
            self._add(pattern, next_column)

        return pattern

    def _generate_random_pattern(self, p2, p3, p4, p5) -> Pattern:
        pattern = Pattern()
        pattern.add_pattern(
            self._generate_random_notes(self._get_random_note_count(p2, p3, p4, p5)))
        if self.random_start > 0 and self._has_special_column:
            self._add(pattern, 0)
        return pattern

    def _generate_random_pattern_with_mirrored(self, centre_probability, p2, p3) -> Pattern:
        if self.convert_type & PT_FORCE_NOT_STACK:
            return self._generate_random_pattern(
                1 / 2.0 + p2 / 2, p2, (p2 + p3) / 2, p3)

        pattern = Pattern()
        tc = self.total_columns
        rs = self.random_start
        note_count, add_to_centre = self._get_random_note_count_mirrored(
            centre_probability, p2, p3)

        column_limit = (tc if tc % 2 == 0 else tc - 1) // 2
        next_column = self.get_random_column(upper_bound=column_limit)

        for _ in range(note_count):
            next_column = self.find_available_column(
                next_column, pattern, upper_bound=column_limit)
            self._add(pattern, next_column)
            self._add(pattern, rs + tc - next_column - 1)

        if add_to_centre:
            self._add(pattern, tc // 2)

        if rs > 0 and self._has_special_column:
            self._add(pattern, 0)

        return pattern

    def _get_random_note_count(self, p2, p3, p4, p5) -> int:
        tc = self.total_columns
        if tc == 2:
            p2 = p3 = p4 = p5 = 0
        elif tc == 3:
            p2 = min(p2, 0.1); p3 = p4 = p5 = 0
        elif tc == 4:
            p2 = min(p2, 0.23); p3 = min(p3, 0.04); p4 = p5 = 0
        elif tc == 5:
            p3 = min(p3, 0.15); p4 = min(p4, 0.03); p5 = 0

        if self.hit_object.hit_sound & HIT_CLAP:
            p2 = 1
        return self.get_random_note_count(p2, p3, p4, p5)

    def _get_random_note_count_mirrored(self, centre_probability, p2, p3):
        tc = self.total_columns
        if tc == 2:
            centre_probability = 0; p2 = 0; p3 = 0
        elif tc == 3:
            centre_probability = min(centre_probability, 0.03); p2 = 0; p3 = 0
        elif tc == 4:
            centre_probability = 0
            p2 = 1 - max((1 - p2) * 2, 0.8); p3 = 0
        elif tc == 5:
            centre_probability = min(centre_probability, 0.03); p3 = 0
        elif tc == 6:
            centre_probability = 0
            p2 = 1 - max((1 - p2) * 2, 0.5)
            p3 = 1 - max((1 - p3) * 2, 0.85)

        p2 = max(0.0, min(1.0, p2))
        p3 = max(0.0, min(1.0, p3))

        centre_val = self.random.next_double()
        note_count = self.get_random_note_count(p2, p3)
        add_to_centre = (tc % 2 != 0 and note_count != 3
                         and centre_val > 1 - centre_probability)
        return note_count, add_to_centre

    def _add(self, pattern: Pattern, column: int) -> None:
        t = int(round_half_even(self.hit_object.start_time))
        pattern.add_object(ManiaObj(column=column, start_time=t, end_time=t,
                                    hit_sound=self.hit_object.hit_sound))


# ───────────────────────── Slider pattern generator ──────────────────────
class SliderPatternGenerator(LegacyPatternGenerator):
    def __init__(self, random, hit_object, total_columns, previous_pattern,
                 conversion_difficulty, slider_multiplier, timing_beat_length):
        super().__init__(random, hit_object, total_columns, previous_pattern,
                         conversion_difficulty)
        self.convert_type = PT_NONE
        if not hit_object.has_kiai:
            self.convert_type = PT_LOW_PROBABILITY

        self.span_count = max(1, hit_object.span_count)
        self.start_time = int(round_half_even(hit_object.start_time))

        # beatLength adjusted for slider velocity (mania ruleset).
        if hit_object.slider_velocity != 1.0:
            beat_length = _precision_adjusted_beat_length(
                hit_object.slider_velocity, timing_beat_length)
        else:
            beat_length = timing_beat_length

        distance = hit_object.pixel_length
        self.end_time = int(math.floor(
            self.start_time
            + distance * beat_length * self.span_count * 0.01 / slider_multiplier))
        self.segment_duration = (self.end_time - self.start_time) // self.span_count

    # ConvertHitObject reads these to drive recordNote/computeDensity.
    def generate(self) -> list[Pattern]:
        original = self._generate()
        if len(original.hit_objects) == 1:
            return [original]
        intermediate = Pattern()
        end_time_pattern = Pattern()
        for obj in original.hit_objects:
            if self.end_time != int(round_half_even(obj.end_time)):
                intermediate.add_object(obj)
            else:
                end_time_pattern.add_object(obj)
        return [intermediate, end_time_pattern]

    def _generate(self) -> Pattern:
        tc = self.total_columns
        rs = self.random_start

        if tc == 1:
            pattern = Pattern()
            self._add(pattern, 0, self.start_time, self.end_time)
            return pattern

        if self.span_count > 1:
            if self.segment_duration <= 90:
                return self._generate_random_hold_notes(self.start_time, 1)
            if self.segment_duration <= 120:
                self.convert_type |= PT_FORCE_NOT_STACK
                return self._generate_random_notes(self.start_time, self.span_count + 1)
            if self.segment_duration <= 160:
                return self._generate_stair(self.start_time)
            if self.segment_duration <= 200 and self.conversion_difficulty > 3:
                return self._generate_random_multiple_notes(self.start_time)
            duration = self.end_time - self.start_time
            if duration >= 4000:
                return self._generate_n_random_notes(self.start_time, 0.23, 0, 0)
            if self.segment_duration > 400 and self.span_count < tc - 1 - rs:
                return self._generate_tiled_hold_notes(self.start_time)
            return self._generate_hold_and_normal_notes(self.start_time)

        if self.segment_duration <= 110:
            if self.previous_pattern.column_with_objects < tc:
                self.convert_type |= PT_FORCE_NOT_STACK
            else:
                self.convert_type &= ~PT_FORCE_NOT_STACK
            return self._generate_random_notes(
                self.start_time, 1 if self.segment_duration < 80 else 2)

        if self.conversion_difficulty > 6.5:
            if self.convert_type & PT_LOW_PROBABILITY:
                return self._generate_n_random_notes(self.start_time, 0.78, 0.3, 0)
            return self._generate_n_random_notes(self.start_time, 0.85, 0.36, 0.03)

        if self.conversion_difficulty > 4:
            if self.convert_type & PT_LOW_PROBABILITY:
                return self._generate_n_random_notes(self.start_time, 0.43, 0.08, 0)
            return self._generate_n_random_notes(self.start_time, 0.56, 0.18, 0)

        if self.conversion_difficulty > 2.5:
            if self.convert_type & PT_LOW_PROBABILITY:
                return self._generate_n_random_notes(self.start_time, 0.3, 0, 0)
            return self._generate_n_random_notes(self.start_time, 0.37, 0.08, 0)

        if self.convert_type & PT_LOW_PROBABILITY:
            return self._generate_n_random_notes(self.start_time, 0.17, 0, 0)
        return self._generate_n_random_notes(self.start_time, 0.27, 0, 0)

    def _generate_random_hold_notes(self, start_time, note_count) -> Pattern:
        pattern = Pattern()
        tc = self.total_columns
        usable_columns = tc - self.random_start - self.previous_pattern.column_with_objects
        next_column = self.get_random_column()

        for _ in range(min(usable_columns, note_count)):
            next_column = self.find_available_column(
                next_column, pattern, self.previous_pattern)
            self._add(pattern, next_column, start_time, self.end_time)

        for _ in range(note_count - usable_columns):
            next_column = self.find_available_column(next_column, pattern)
            self._add(pattern, next_column, start_time, self.end_time)

        return pattern

    def _generate_random_notes(self, start_time, note_count) -> Pattern:
        pattern = Pattern()
        tc = self.total_columns
        next_column = self.get_column(self.hit_object.x, True)
        if (self.convert_type & PT_FORCE_NOT_STACK) and \
                self.previous_pattern.column_with_objects < tc:
            next_column = self.find_available_column(next_column, self.previous_pattern)

        last_column = next_column
        for _ in range(note_count):
            self._add(pattern, next_column, start_time, start_time)
            next_column = self.find_available_column(
                next_column, validation=lambda c, lc=last_column: c != lc)
            last_column = next_column
            start_time += self.segment_duration
        return pattern

    def _generate_stair(self, start_time) -> Pattern:
        pattern = Pattern()
        tc = self.total_columns
        rs = self.random_start
        column = self.get_column(self.hit_object.x, True)
        increasing = self.random.next_double() > 0.5

        for _ in range(self.span_count + 1):
            self._add(pattern, column, start_time, start_time)
            start_time += self.segment_duration
            if increasing:
                if column >= tc - 1:
                    increasing = False
                    column -= 1
                else:
                    column += 1
            else:
                if column <= rs:
                    increasing = True
                    column += 1
                else:
                    column -= 1
        return pattern

    def _generate_random_multiple_notes(self, start_time) -> Pattern:
        pattern = Pattern()
        tc = self.total_columns
        rs = self.random_start
        legacy = 4 <= tc <= 8
        interval = self.random.next_range(1, tc - (1 if legacy else 0))
        next_column = self.get_column(self.hit_object.x, True)

        for _ in range(self.span_count + 1):
            self._add(pattern, next_column, start_time, start_time)
            next_column += interval
            if next_column >= tc - rs:
                next_column = next_column - tc - rs + (1 if legacy else 0)
            next_column += rs
            if tc > 2:
                self._add(pattern, next_column, start_time, start_time)
            next_column = self.get_random_column()
            start_time += self.segment_duration
        return pattern

    def _generate_n_random_notes(self, start_time, p2, p3, p4) -> Pattern:
        tc = self.total_columns
        if tc == 2:
            p2 = p3 = p4 = 0
        elif tc == 3:
            p2 = min(p2, 0.1); p3 = 0; p4 = 0
        elif tc == 4:
            p2 = min(p2, 0.3); p3 = min(p3, 0.04); p4 = 0
        elif tc == 5:
            p2 = min(p2, 0.34); p3 = min(p3, 0.1); p4 = min(p4, 0.03)

        can_generate_two = not (self.convert_type & PT_LOW_PROBABILITY)
        can_generate_two = can_generate_two and (
            self._has_double_sample(self.hit_object.hit_sound)
            or self._has_double_sample(self._sample_sound_at(self.start_time)))
        if can_generate_two:
            p2 = 1

        return self._generate_random_hold_notes(
            start_time, self.get_random_note_count(p2, p3, p4))

    def _generate_tiled_hold_notes(self, start_time) -> Pattern:
        pattern = Pattern()
        tc = self.total_columns
        column_repeat = min(self.span_count, tc)
        end_time = start_time + self.segment_duration * self.span_count

        next_column = self.get_column(self.hit_object.x, True)
        if (self.convert_type & PT_FORCE_NOT_STACK) and \
                self.previous_pattern.column_with_objects < tc:
            next_column = self.find_available_column(next_column, self.previous_pattern)

        for _ in range(column_repeat):
            next_column = self.find_available_column(next_column, pattern)
            self._add(pattern, next_column, start_time, end_time)
            start_time += self.segment_duration
        return pattern

    def _generate_hold_and_normal_notes(self, start_time) -> Pattern:
        pattern = Pattern()
        tc = self.total_columns
        hold_column = self.get_column(self.hit_object.x, True)
        if (self.convert_type & PT_FORCE_NOT_STACK) and \
                self.previous_pattern.column_with_objects < tc:
            hold_column = self.find_available_column(hold_column, self.previous_pattern)

        self._add(pattern, hold_column, start_time, self.end_time)

        next_column = self.get_random_column()
        if self.conversion_difficulty > 6.5:
            note_count = self.get_random_note_count(0.63, 0)
        elif self.conversion_difficulty > 4:
            note_count = self.get_random_note_count(0.12 if tc < 6 else 0.45, 0)
        elif self.conversion_difficulty > 2.5:
            note_count = self.get_random_note_count(0 if tc < 6 else 0.24, 0)
        else:
            note_count = 0
        note_count = min(tc - 1, note_count)

        ss = self._sample_sound_at(start_time)
        ignore_head = not (ss & (HIT_WHISTLE | HIT_FINISH | HIT_CLAP))

        row_pattern = Pattern()
        for _ in range(self.span_count + 1):
            if not (ignore_head and start_time == self.start_time):
                for _ in range(note_count):
                    next_column = self.find_available_column(
                        next_column, row_pattern,
                        validation=lambda c, hc=hold_column: c != hc)
                    self._add(row_pattern, next_column, start_time, start_time)
            pattern.add_pattern(row_pattern)
            row_pattern.clear()
            start_time += self.segment_duration
        return pattern

    # ── helpers ──
    def _sample_sound_at(self, time: int) -> int:
        """Equivalent of sampleInfoListAt(time): the node-sample hitsound at
        the slider node index for ``time`` (or the base hitSound). We only
        need the bitfield to test for clap/finish/whistle."""
        nodes = self.hit_object.node_hit_sounds
        if not nodes:
            return self.hit_object.hit_sound
        if self.segment_duration == 0:
            index = 0
        else:
            index = (time - self.start_time) // self.segment_duration
        if 0 <= index < len(nodes):
            return nodes[index]
        return self.hit_object.hit_sound

    @staticmethod
    def _has_double_sample(hit_sound: int) -> bool:
        return bool(hit_sound & HIT_CLAP) or bool(hit_sound & HIT_FINISH)

    @property
    def _conv_dummy(self):  # pragma: no cover - placeholder
        return None

    def _add(self, pattern: Pattern, column: int, start_time: int, end_time: int) -> None:
        pattern.add_object(ManiaObj(column=column, start_time=int(start_time),
                                    end_time=int(end_time),
                                    hit_sound=self.hit_object.hit_sound))


# ───────────────────────── Spinner pattern generator ─────────────────────
class SpinnerPatternGenerator(LegacyPatternGenerator):
    def __init__(self, random, hit_object, total_columns, previous_pattern,
                 conversion_difficulty):
        super().__init__(random, hit_object, total_columns, previous_pattern,
                         conversion_difficulty)
        self.end_time = int(hit_object.end_time)
        self.convert_type = (PT_NONE
                             if previous_pattern.column_with_objects == total_columns
                             else PT_FORCE_NOT_STACK)

    def generate(self) -> list[Pattern]:
        return [self._generate()]

    def _generate(self) -> Pattern:
        pattern = Pattern()
        tc = self.total_columns
        generate_hold = (self.end_time - self.hit_object.start_time) >= 100

        if tc == 8 and (self.hit_object.hit_sound & HIT_FINISH) \
                and (self.end_time - self.hit_object.start_time) < 1000:
            self._add(pattern, 0, generate_hold)
        elif tc == 8:
            self._add(pattern, self._get_random_column(), generate_hold)
        else:
            self._add(pattern, self._get_random_column(0), generate_hold)
        return pattern

    def _get_random_column(self, lower_bound=None) -> int:
        if self.convert_type & PT_FORCE_NOT_STACK:
            return self.find_available_column(
                self.get_random_column(lower_bound), self.previous_pattern,
                lower_bound=lower_bound)
        return self.find_available_column(
            self.get_random_column(lower_bound), lower_bound=lower_bound)

    def _add(self, pattern: Pattern, column: int, hold_note: bool) -> None:
        st = int(self.hit_object.start_time)
        if hold_note:
            pattern.add_object(ManiaObj(column=column, start_time=st,
                                        end_time=self.end_time,
                                        hit_sound=self.hit_object.hit_sound))
        else:
            pattern.add_object(ManiaObj(column=column, start_time=st, end_time=st,
                                        hit_sound=self.hit_object.hit_sound))


# ─────────────────────────────── helpers ─────────────────────────────────
def round_half_even(value: float) -> int:
    """C# ``Math.Round`` default (MidpointRounding.ToEven / banker's)."""
    f = math.floor(value)
    diff = value - f
    if diff < 0.5:
        return int(f)
    if diff > 0.5:
        return int(f) + 1
    # exactly .5 → round to even
    return int(f) if (int(f) % 2 == 0) else int(f) + 1


def _precision_adjusted_beat_length(slider_velocity: float, timing_beat_length: float) -> float:
    """Port of LegacyRulesetExtensions.GetPrecisionAdjustedBeatLength for the
    mania ruleset.

        sliderVelocityAsBeatLength = -100 / sliderVelocity
        bpmMultiplier = clamp(-sliderVelocityAsBeatLength, 10, 10000) / 100   (mania)
        beatLength = timingPoint.BeatLength * bpmMultiplier

    Note -sliderVelocityAsBeatLength = 100 / sliderVelocity, so for SV=1 the
    multiplier is 1.0. The clamp uses a (float) cast in C#; the resulting
    value is well within float precision for normal SVs.
    """
    slider_velocity_as_beat_length = -100.0 / slider_velocity
    bpm_multiplier = max(10.0, min(10000.0, -slider_velocity_as_beat_length)) / 100.0
    return timing_beat_length * bpm_multiplier


# ─────────────────────────── Top-level convert ───────────────────────────
@dataclass
class ConvertResult:
    objects: list[ManiaObj]
    seed: int
    conversion_difficulty: float
    note_count: int


def round_half_even_seed(value: float) -> int:
    """MathF.Round → round-half-to-even on a float, then (int) cast."""
    return round_half_even(value)


def compute_seed(drain_rate: float, circle_size: float,
                 overall_difficulty: float, approach_rate: float) -> int:
    """ManiaBeatmapConverter seed:
        (int)Round(DrainRate + CircleSize) * 20
        + (int)(OverallDifficulty * 41.2)
        + (int)Round(ApproachRate)
    """
    a = round_half_even_seed(drain_rate + circle_size) * 20
    b = int(overall_difficulty * 41.2)
    c = round_half_even_seed(approach_rate)
    return a + b + c


def compute_conversion_difficulty(
    drain_rate: float, approach_rate: float, hit_object_count: int,
    first_start_time: float, last_start_time: float, total_break_time: float,
) -> float:
    """LegacyPatternGenerator.ConversionDifficulty."""
    drain_time = int((last_start_time - first_start_time - total_break_time) / 1000)
    if drain_time == 0:
        drain_time = 10000
    clamped_ar = max(4.0, min(7.0, approach_rate))
    cd = ((drain_rate + clamped_ar) / 1.5 + hit_object_count / drain_time * 9.0) / 38.0 * 5.0 / 1.15
    return min(cd, 12.0)


def convert_legacy(
    objects: list[StdObject],
    *,
    total_columns: int,
    drain_rate: float,
    circle_size: float,
    overall_difficulty: float,
    approach_rate: float,
    slider_multiplier: float,
    total_break_time: float,
    beat_length_at,
) -> ConvertResult:
    """Run the full faithful lazer std→mania legacy conversion.

    ``beat_length_at(time)`` must return the *uninherited* timing point
    BeatLength (ms/beat) active at ``time`` (used both for slider timing and
    HitCircle density thresholds).

    Returns every produced mania note (with lazer's column choices, which the
    caller is free to overwrite from a replay), plus the seed and
    conversion difficulty for diagnostics.
    """
    seed = compute_seed(drain_rate, circle_size, overall_difficulty, approach_rate)
    random = LegacyRandom(seed)

    sorted_objects = sorted(objects, key=lambda o: o.start_time)
    first_time = sorted_objects[0].start_time if sorted_objects else 0.0
    last_time = sorted_objects[-1].start_time if sorted_objects else 0.0
    conversion_difficulty = compute_conversion_difficulty(
        drain_rate, approach_rate, len(sorted_objects),
        first_time, last_time, total_break_time)

    last_pattern = Pattern()
    last_stair = PT_STAIR
    last_time_recorded = 0.0
    last_position = (0.0, 0.0)
    # density bookkeeping (LimitedCapacityQueue of size 7)
    prev_note_times: list[float] = []
    density = float("inf")
    out: list[ManiaObj] = []

    def compute_density(new_note_time: float):
        nonlocal density
        prev_note_times.append(new_note_time)
        if len(prev_note_times) > 7:
            prev_note_times.pop(0)
        if len(prev_note_times) >= 2:
            density = (prev_note_times[-1] - prev_note_times[0]) / len(prev_note_times)

    for obj in sorted_objects:
        if obj.kind == "circle":
            compute_density(obj.start_time)
            gen = HitCirclePatternGenerator(
                random, obj, total_columns, last_pattern, conversion_difficulty,
                last_time_recorded, last_position, density, last_stair,
                beat_length_at(obj.start_time))
            last_time_recorded = obj.start_time
            last_position = (obj.x, obj.y)
            new_pattern = gen.generate()
            last_stair = gen.stair_type
            last_pattern = new_pattern
            out.extend(new_pattern.hit_objects)

        elif obj.kind == "slider":
            gen = SliderPatternGenerator(
                random, obj, total_columns, last_pattern, conversion_difficulty,
                slider_multiplier, beat_length_at(obj.start_time))
            # ConvertHitObject: recordNote/computeDensity for each span node.
            for i in range(gen.span_count + 1):
                t = obj.start_time + gen.segment_duration * i
                last_time_recorded = t
                last_position = (obj.x, obj.y)
                compute_density(t)
            patterns = gen.generate()
            # lastPattern is the LAST yielded pattern (endTimePattern when split).
            for p in patterns:
                out.extend(p.hit_objects)
            last_pattern = patterns[-1]

        elif obj.kind == "spinner":
            gen = SpinnerPatternGenerator(
                random, obj, total_columns, last_pattern, conversion_difficulty)
            last_time_recorded = obj.end_time
            last_position = (256.0, 192.0)
            compute_density(obj.end_time)
            patterns = gen.generate()
            for p in patterns:
                out.extend(p.hit_objects)
            # Spinner does NOT update lastPattern (only HitCircle/Slider do).

    return ConvertResult(objects=out, seed=seed,
                         conversion_difficulty=conversion_difficulty,
                         note_count=len(out))
