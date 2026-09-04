"""Regression tests for osu_mania_renderer_v2.converter.

Why this file exists: the std→mania converter must press-match against
the **fully-expanded mania-note list** (slider streams included), not
the top-level std-object list. An earlier version of the converter only
recovered the FIRST stream note's column from the replay and random-
walked the rest via `_pick_col`. That made slider-heavy converted maps
render with stream notes in wrong columns — score reads correct from
the .osr header (96%+) but the HUD/visual shows the player missing
nearly everything they actually hit. See
project_std_to_mania_slider_columns.md in memory for the full story.
"""
from __future__ import annotations

from osu_mania_renderer_v2.converter import (
    DEFAULT_CONVERTED_KEY_COUNT,
    convert_standard_to_mania,
)
from osu_mania_renderer_v2.models import HoldNote, KeyEvent, TimingPoint


def _kev(t: int, mask: int) -> KeyEvent:
    return KeyEvent(time_ms=t, keys_held=mask)


def _press_then_release(t: int, col: int, gap_ms: int = 15) -> list[KeyEvent]:
    """One press at `t` on `col`, released `gap_ms` later. Returns the
    two KeyEvents the press_events extractor needs to spot the
    0→1 transition."""
    return [_kev(t, 1 << col), _kev(t + gap_ms, 0)]


def test_slider_stream_notes_inherit_player_press_columns() -> None:
    """A std slider expands into a mania stream of N notes (every ~beat/4
    ms). When the player tapped distinct columns at each ¼-beat, the
    converter must place each in-stream mania note in the column the
    player actually pressed — not let `_expand_slider_to_stream`'s
    random-walk decide.
    """
    # BPM=120 → beat=500 ms → stream step=125 ms.
    # 300-pixel length-1-repeat slider at SliderMultiplier=1.4, SV=1:
    # per-repeat = (300 / (100*1.4*1.0)) * 500 = 1071 ms.
    # That's ~8 stream ¼-beat slots starting at t=1000.
    hit_obj_block = "100,192,1000,2,0,L|100:50,1,300\n"
    timing_points = (
        TimingPoint(
            time_ms=0, beat_length_ms=500.0, sv_multiplier=1.0,
            uninherited=True, sample_set="", custom_index=0, volume=100,
        ),
    )
    # Player tapped 4 distinct columns at the first 4 ¼-beat positions.
    key_events: list[KeyEvent] = [_kev(0, 0)]
    expected_first4 = [(1000, 2), (1125, 3), (1250, 0), (1375, 1)]
    for t, col in expected_first4:
        key_events.extend(_press_then_release(t, col))

    bm = convert_standard_to_mania(
        hit_objects_block=hit_obj_block,
        timing_points=timing_points,
        audio_filename="a.mp3", background_filename=None,
        audio_lead_in_ms=0,
        artist="", title="", difficulty="", creator="",
        beatmap_id=None, beatmapset_id=None,
        default_sample_set="Soft",
        slider_multiplier=1.4, overall_difficulty=5.0,
        key_count=DEFAULT_CONVERTED_KEY_COUNT,
        seed_source="regression-test",
        replay_key_events=tuple(key_events),
    )

    # Find the 4 stream notes that align with the player's presses.
    notes_by_time = {n.time_ms: n for n in bm.notes}
    for t, expected_col in expected_first4:
        assert t in notes_by_time, (
            f"converter dropped the stream note at t={t}; got times "
            f"{sorted(notes_by_time)}"
        )
        actual_col = notes_by_time[t].column
        assert actual_col == expected_col, (
            f"slider-stream note at t={t} drifted off the player's "
            f"hand: expected column {expected_col} (matching press), "
            f"got {actual_col}"
        )


def test_circle_press_match_unchanged() -> None:
    """Press-matching on plain circles (no slider expansion) still
    works — the new mania-note-level matcher is strictly stronger than
    the old std-object-level matcher, so single-circle cases stay
    pinned to the player's press."""
    hit_obj_block = "256,192,500,5,0,0:0:0:0:\n"   # type=5: circle+new_combo
    timing_points = (
        TimingPoint(
            time_ms=0, beat_length_ms=500.0, sv_multiplier=1.0,
            uninherited=True, sample_set="", custom_index=0, volume=100,
        ),
    )
    key_events = tuple([_kev(0, 0)] + _press_then_release(500, 3))
    bm = convert_standard_to_mania(
        hit_objects_block=hit_obj_block,
        timing_points=timing_points,
        audio_filename="a.mp3", background_filename=None,
        audio_lead_in_ms=0,
        artist="", title="", difficulty="", creator="",
        beatmap_id=None, beatmapset_id=None,
        default_sample_set="Soft",
        slider_multiplier=1.4, overall_difficulty=5.0,
        key_count=DEFAULT_CONVERTED_KEY_COUNT,
        seed_source="circle-regression",
        replay_key_events=key_events,
    )
    assert len(bm.notes) == 1, f"expected 1 note from 1 circle, got {len(bm.notes)}"
    assert bm.notes[0].column == 3, (
        f"circle press recovery broken: expected col 3, got {bm.notes[0].column}"
    )


def test_unmatched_stream_notes_fall_back_to_heuristic() -> None:
    """If the player only pressed during part of a slider (e.g. let go
    early), the trailing stream notes have no matching press and must
    fall back to the position-based heuristic rather than crashing or
    leaving a placeholder column. Heuristic columns are RNG-influenced
    so we don't assert specific values — just that every note has a
    valid column in [0, key_count)."""
    hit_obj_block = "100,192,1000,2,0,L|100:50,1,300\n"
    timing_points = (
        TimingPoint(
            time_ms=0, beat_length_ms=500.0, sv_multiplier=1.0,
            uninherited=True, sample_set="", custom_index=0, volume=100,
        ),
    )
    # Player only tapped the first stream slot, then let go.
    key_events = tuple([_kev(0, 0)] + _press_then_release(1000, 2))
    bm = convert_standard_to_mania(
        hit_objects_block=hit_obj_block,
        timing_points=timing_points,
        audio_filename="a.mp3", background_filename=None,
        audio_lead_in_ms=0,
        artist="", title="", difficulty="", creator="",
        beatmap_id=None, beatmapset_id=None,
        default_sample_set="Soft",
        slider_multiplier=1.4, overall_difficulty=5.0,
        key_count=DEFAULT_CONVERTED_KEY_COUNT,
        seed_source="missed-stream",
        replay_key_events=key_events,
    )
    key_count = DEFAULT_CONVERTED_KEY_COUNT
    for n in bm.notes:
        assert 0 <= n.column < key_count, (
            f"note at t={n.time_ms} has out-of-range column {n.column}"
        )
    # The first stream note must still match the one press the player made.
    notes_by_time = {n.time_ms: n for n in bm.notes}
    assert notes_by_time[1000].column == 2

    # And the slider must still end in a hold so the tail still releases.
    tail = bm.notes[-1]
    assert isinstance(tail, HoldNote), (
        "slider expansion must end in a HoldNote so the tail registers"
    )
