"""Parse osu!mania .osu files. Mania-only — std/taiko/ctb raise NotAManiaError."""
from __future__ import annotations

from bisect import bisect_right as _bisect_right
from pathlib import Path

from osu_mania_renderer_v2.errors import BeatmapParseError, NotAManiaError
from osu_mania_renderer_v2.models import (
    BeatmapInfo, HitSample, HoldNote, Note, TimingPoint,
)

# Single-slot identity cache for sv_distance_at's TP-times list. The old
# code rebuilt `[tp.time_ms for tp in timing_points]` on EVERY call — and
# sv_distance_at runs once per visible note per frame. The same
# timing_points tuple is passed for a whole render, so cache the derived
# list keyed on tuple identity. Bit-identical results; perf only.
_SV_TIMES_CACHE: tuple | None = None

# Mania hit-object type bit 7 (128) = hold note.
_HOLD_TYPE_BIT = 1 << 7


def parse_beatmap(
    path: Path,
    *,
    allow_converted: bool = False,
    convert_to_keys: int = 4,
    replay_key_events: tuple | None = None,
) -> BeatmapInfo:
    """Parse a .osu file.

    When the file declares Mode != 3 (i.e. it's a standard/taiko/ctb
    beatmap) and `allow_converted=True`, route through the mania
    converter — that's what produces the chart the player actually saw
    when they hit the in-game "convert to mania" toggle.

    When `replay_key_events` is also provided, the converter uses the
    player's actual key presses to recover osu!stable's exact column
    choices for every note the player hit — dramatically improving
    fidelity for replay rendering."""
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    sections = _split_sections(text)
    general = _kv(sections.get("General", ""))
    metadata = _kv(sections.get("Metadata", ""))
    difficulty = _kv(sections.get("Difficulty", ""))
    events = sections.get("Events", "")
    hit_objects_raw = sections.get("HitObjects", "")

    mode_str = general.get("Mode", "0")
    try:
        mode = int(mode_str)
    except ValueError as e:
        raise BeatmapParseError(f"Invalid Mode={mode_str!r}") from e
    if mode != 3:
        if not allow_converted:
            raise NotAManiaError(mode)
        # Converted path: turn the standard chart into a synthetic mania
        # chart and short-circuit the rest of this function. The renderer
        # never has to know the source mode wasn't mania.
        from osu_mania_renderer_v2.converter import convert_standard_to_mania
        try:
            audio_lead_in_ms = int(float(general.get("AudioLeadIn", "0")))
        except ValueError:
            audio_lead_in_ms = 0
        seed = (metadata.get("BeatmapID") or metadata.get("Version") or "")
        try:
            slider_multiplier = float(difficulty.get("SliderMultiplier", "1.4"))
        except (TypeError, ValueError):
            slider_multiplier = 1.4
        try:
            overall_difficulty = float(difficulty.get("OverallDifficulty", "5"))
        except (TypeError, ValueError):
            overall_difficulty = 5.0
        # The faithful-lazer note generator needs the FULL difficulty set:
        # the RNG seed is Round(HP+CS)*20 + (int)(OD*41.2) + Round(AR), and
        # conversionDifficulty depends on HP, AR, the object count and the
        # drain time (last-first start time minus total break time).
        hp_drain_rate = _float_or_none(difficulty.get("HPDrainRate"))
        circle_size = _float_or_none(difficulty.get("CircleSize"))
        approach_rate = _float_or_none(difficulty.get("ApproachRate"))
        return convert_standard_to_mania(
            hit_objects_block=hit_objects_raw,
            timing_points=_parse_timing_points(sections.get("TimingPoints", "")),
            audio_filename=general.get("AudioFilename"),
            background_filename=_parse_background(events),
            audio_lead_in_ms=audio_lead_in_ms,
            artist=metadata.get("Artist", ""),
            title=metadata.get("Title", ""),
            difficulty=metadata.get("Version", ""),
            creator=metadata.get("Creator", ""),
            beatmap_id=_int_or_none(metadata.get("BeatmapID")),
            beatmapset_id=_int_or_none(metadata.get("BeatmapSetID")),
            default_sample_set=general.get("SampleSet", "Soft"),
            slider_multiplier=slider_multiplier,
            overall_difficulty=overall_difficulty,
            key_count=convert_to_keys,
            seed_source=str(seed),
            replay_key_events=replay_key_events,
            hp_drain_rate=hp_drain_rate,
            circle_size=circle_size,
            approach_rate=approach_rate,
            total_break_time_ms=_parse_total_break_time(events),
            kiai_points=_parse_kiai_points(sections.get("TimingPoints", "")),
        )

    try:
        key_count = int(float(difficulty["CircleSize"]))
    except (KeyError, ValueError) as e:
        raise BeatmapParseError("Missing or invalid CircleSize (key count)") from e

    audio_filename = general.get("AudioFilename")
    try:
        audio_lead_in_ms = int(float(general.get("AudioLeadIn", "0")))
    except ValueError:
        audio_lead_in_ms = 0

    background = _parse_background(events)

    notes, max_time = _parse_hit_objects(hit_objects_raw, key_count)
    notes_sorted = tuple(sorted(notes, key=lambda n: n.time_ms))
    timing_points = _parse_timing_points(sections.get("TimingPoints", ""))
    default_sample_set = general.get("SampleSet", "Soft")

    try:
        od = float(difficulty.get("OverallDifficulty", "5"))
    except (TypeError, ValueError):
        od = 5.0

    return BeatmapInfo(
        key_count=key_count,
        notes=notes_sorted,
        audio_filename=audio_filename,
        background_filename=background,
        total_duration_ms=max_time,
        audio_lead_in_ms=audio_lead_in_ms,
        artist=metadata.get("Artist", ""),
        title=metadata.get("Title", ""),
        difficulty=metadata.get("Version", ""),
        creator=metadata.get("Creator", ""),
        beatmap_id=_int_or_none(metadata.get("BeatmapID")),
        beatmapset_id=_int_or_none(metadata.get("BeatmapSetID")),
        default_sample_set=default_sample_set,
        timing_points=timing_points,
        overall_difficulty=od,
    )


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.startswith("[") and line.endswith("]"):
            if current is not None:
                sections[current] = "\n".join(buf)
            current = line[1:-1]
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf)
    if not sections:
        raise BeatmapParseError("No section headers found")
    return sections


def _kv(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.splitlines():
        if not line or line.startswith("//"):
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def _parse_background(events: str) -> str | None:
    for line in events.splitlines():
        # Background events look like: 0,0,"filename.jpg",0,0
        parts = line.split(",")
        if len(parts) >= 3 and parts[0].strip() in ("0", "Background"):
            return parts[2].strip().strip('"')
    return None


def _parse_hit_objects(block: str, key_count: int) -> tuple[list, int]:
    notes: list = []
    max_time = 0
    for line in block.splitlines():
        if not line or line.startswith("//"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            x = int(parts[0])
            time_ms = int(parts[2])
            type_flags = int(parts[3])
            hit_sound = int(parts[4])
        except ValueError:
            continue
        column = int(x * key_count / 512)
        column = min(max(column, 0), key_count - 1)
        if type_flags & _HOLD_TYPE_BIT:
            # Hold: parts[5] is "endTime:normalSet:additionSet:index:volume:filename".
            if len(parts) < 6:
                continue
            tail = parts[5]
            tail_bits = tail.split(":", 5)
            try:
                end_ms = int(tail_bits[0])
            except (ValueError, IndexError):
                continue
            sample = _parse_hit_sample_fields(tail_bits[1:])
            notes.append(HoldNote(
                column=column, time_ms=time_ms, end_time_ms=end_ms,
                hit_sound=hit_sound, hit_sample=sample,
            ))
            max_time = max(max_time, end_ms)
        else:
            # Tap: parts[5] (if present) is the hitSample string itself.
            sample = HitSample()
            if len(parts) >= 6:
                sample = _parse_hit_sample_fields(parts[5].split(":", 4))
            notes.append(Note(
                column=column, time_ms=time_ms,
                hit_sound=hit_sound, hit_sample=sample,
            ))
            max_time = max(max_time, time_ms)
    return notes, max_time


def _parse_hit_sample_fields(fields: list[str]) -> HitSample:
    """Build a HitSample from the 5 colon-separated trailing fields
    (normalSet, additionSet, index, volume, filename). Missing trailing
    fields are tolerated and treated as the default 0/empty."""
    def _i(i: int) -> int:
        try:
            return int(fields[i]) if i < len(fields) and fields[i] else 0
        except ValueError:
            return 0
    filename = fields[4].strip() if len(fields) > 4 else ""
    return HitSample(
        normal_set=_i(0), addition_set=_i(1), index=_i(2),
        volume=_i(3), filename=filename,
    )


def _parse_timing_points(block: str) -> tuple:
    """Each timing-point row is:
        time, beatLength, meter, sampleSet, sampleIndex, volume,
        uninherited, effects

    `beatLength` carries BOTH the BPM (positive value, on uninherited red
    points) AND the scroll-velocity multiplier (negative value on inherited
    green points, where `-100/beatLength == sv_multiplier`). Mania uses the
    SV multiplier to speed up / slow down the playfield mid-song, which is
    what the user wants the renderer to honour.
    """
    out: list[TimingPoint] = []
    for line in block.splitlines():
        if not line or line.startswith("//"):
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            time_ms = int(float(parts[0]))
            beat_length = float(parts[1])
            sample_set = int(parts[3])
            custom_index = int(parts[4])
            volume = int(parts[5])
            uninherited = parts[6].strip() == "1"
        except ValueError:
            continue
        beat_length_ms = 500.0
        if uninherited:
            sv_multiplier = 1.0
            if beat_length > 0:
                beat_length_ms = beat_length
        else:
            # Inherited (green) TP: beat_length is negative. SV = -100 / bl.
            # Guard against malformed maps where it isn't negative.
            sv_multiplier = (-100.0 / beat_length) if beat_length < 0 else 1.0
            # Wider clamp than before — gimmick maps legitimately reach
            # 20-30×. The OLD clamp at 8× hid the SV personality on
            # charts like yambabom's perthed remix. Floor stays at 0.05
            # to prevent divide-by-zero in cumulative-distance lookup.
            sv_multiplier = max(0.05, min(50.0, sv_multiplier))
        out.append(TimingPoint(
            time_ms=time_ms, sample_set=sample_set,
            custom_index=custom_index, volume=volume,
            sv_multiplier=sv_multiplier, uninherited=uninherited,
            beat_length_ms=beat_length_ms,
        ))
    out.sort(key=lambda tp: tp.time_ms)
    return tuple(out)


def _int_or_none(s: str | None) -> int | None:
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _float_or_none(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_total_break_time(events: str) -> float:
    """Sum of all break-period durations (ms). Break events are
    ``2,start,end`` (or ``Break,start,end``). Used by lazer's
    conversionDifficulty drain-time term."""
    total = 0.0
    for line in events.splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and parts[0].strip() in ("2", "Break"):
            try:
                start = float(parts[1])
                end = float(parts[2])
            except ValueError:
                continue
            if end > start:
                total += end - start
    return total


def _parse_kiai_points(block: str) -> tuple[tuple[int, bool], ...]:
    """Extract (time, kiai) pairs from every timing point's effects field
    (bit 0 = kiai), sorted by time. Mirrors lazer's EffectControlPoints,
    whose KiaiMode persists until the next effect point overrides it."""
    out: list[tuple[int, bool]] = []
    for line in block.splitlines():
        if not line.strip() or line.startswith("//"):
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            time_ms = int(float(parts[0]))
            effects = int(parts[7]) if parts[7].strip() else 0
        except ValueError:
            continue
        out.append((time_ms, bool(effects & 1)))
    out.sort(key=lambda kp: kp[0])
    return tuple(out)


# ─── Cumulative SV distance ───
#
# Lazer-mania positions notes by INTEGRATED scroll velocity, not by
# the SV at the note's own moment in time. Two notes scheduled 100 ms
# apart but split across a 0.5×→2× SV boundary should appear visually
# at distances 50 + 50 = 100 "SV-ms" apart in the slow direction,
# 50 + 200 = 250 apart in the fast direction. The old point-sample
# code applied 0.5× or 2× to BOTH notes wholesale, which made notes
# warp/jump at section boundaries.
#
# `build_sv_distance_table(tps)` returns a parallel list of cumulative
# distance values at each TP boundary. `sv_distance_at(t_ms, tps,
# table)` then evaluates the cumulative integral at any time via
# binary search — O(log N) per note per frame, cheap on hot path.


def build_sv_distance_table(
    timing_points: tuple["TimingPoint", ...],
) -> tuple[float, ...]:
    """Pre-compute cumulative SV distance at each TP boundary.
    `table[i]` is the integral of SV from time 0 to
    `timing_points[i].time_ms`, measured in milliseconds-equivalent
    (i.e. distance at SV=1 equals elapsed time in ms).

    Pre-first-TP time is treated as SV=1 — this matches lazer's
    behaviour where notes scheduled before any timing point use the
    default scroll. So `table[0]` already includes `time_ms * 1.0`
    rather than starting from 0. Cheap: O(N) once at render setup."""
    if not timing_points:
        return ()
    # table[0] = SV=1 distance accumulated from t=0 to first_tp.time_ms.
    cum = [float(timing_points[0].time_ms)]
    for i in range(1, len(timing_points)):
        prev = timing_points[i - 1]
        dt = max(0, timing_points[i].time_ms - prev.time_ms)
        cum.append(cum[-1] + dt * prev.sv_multiplier)
    return tuple(cum)


def sv_distance_at(
    t_ms: int,
    timing_points: tuple["TimingPoint", ...],
    table: tuple[float, ...],
) -> float:
    """Cumulative SV-integrated distance at time `t_ms`. Same units as
    `build_sv_distance_table` (milliseconds-equivalent). For times
    before the first TP the integral is taken at SV=1 from time 0."""
    if not timing_points:
        return float(t_ms)
    first = timing_points[0]
    if t_ms <= first.time_ms:
        # Pre-first-TP: assume SV=1. Integral is just elapsed time.
        # Subtle: most maps start their first TP at the audio offset
        # not at t=0, so notes scheduled before that fall here.
        return float(t_ms)
    # Binary search for the TP that owns `t_ms`. `bisect_right` gives
    # the first index strictly past t_ms; subtract 1 for the owning TP.
    global _SV_TIMES_CACHE
    _stc = _SV_TIMES_CACHE
    if _stc is None or _stc[0] is not timing_points:
        _stc = (timing_points, [tp.time_ms for tp in timing_points])
        _SV_TIMES_CACHE = _stc
    times = _stc[1]
    idx = _bisect_right(times, t_ms) - 1
    if idx < 0:
        return float(t_ms)
    tp = timing_points[idx]
    return table[idx] + (t_ms - tp.time_ms) * tp.sv_multiplier
