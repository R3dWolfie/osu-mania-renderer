import pytest

from osu_mania_renderer_v2.beatmap.beatmap import build_sv_distance_table
from osu_mania_renderer_v2.beatmap.judgments import (
    JudgmentEvent, compute_consumed_times, compute_judgments, reconcile_to_counts,
)
from osu_mania_renderer_v2.beatmap.models import HoldNote, KeyEvent, Note, TimingPoint, VisualMods
from osu_mania_renderer_v2.render.render import _key_edges_per_col
from osu_mania_renderer_v2.render.scene import build_hold_visual_states, snapshot

# Playfield occupies the bottom 1000ms of approach time at the receptors.
APPROACH_MS = 600


def test_notes_before_approach_are_above_playfield():
    s = snapshot(
        notes=(Note(0, 1000),),
        key_events=(),
        t_ms=0,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    # The SV-safe queue can contain offscreen notes. At t=0 this one is
    # above the field; it reaches the top at 1000 - 600 = 400ms.
    assert len(s.visible_notes) == 1
    assert s.visible_notes[0].y_fraction < 0
    entered = snapshot((Note(0, 1000),), (), 400, 4, APPROACH_MS, VisualMods())
    assert entered.visible_notes[0].y_fraction == 0


def test_note_becomes_visible_within_approach_window():
    s = snapshot(
        notes=(Note(0, 1000),),
        key_events=(),
        t_ms=500,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    assert len(s.visible_notes) == 1


def test_note_hits_receptor_at_exact_time():
    s = snapshot(
        notes=(Note(0, 1000),),
        key_events=(),
        t_ms=1000,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    n = s.visible_notes[0]
    # Y-position 1.0 means at the receptor.
    assert abs(n.y_fraction - 1.0) < 1e-3


def test_held_keys_reflected():
    s = snapshot(
        notes=(),
        key_events=(KeyEvent(time_ms=500, keys_held=0b0011),),
        t_ms=600,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    assert s.keys_held == (True, True, False, False)


def test_hold_note_renders_as_segment_when_active():
    s = snapshot(
        notes=(HoldNote(0, 1000, 2000),),
        key_events=(),
        t_ms=1500,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    visible_holds = [n for n in s.visible_notes if n.is_hold]
    assert len(visible_holds) == 1
    # Head has passed the receptor (y > 1), tail is still in the playfield.
    h = visible_holds[0]
    assert h.head_y_fraction > 1.0
    assert h.tail_y_fraction < 1.0


def _hold_snapshot(t_ms, events, *, timing_points=()):
    notes = (HoldNote(0, 1000, 2000),)
    judgments = compute_judgments(notes, events, 1)
    presses, releases = _key_edges_per_col(events, 1)
    states = build_hold_visual_states(notes, judgments.events, presses, releases, 127)
    return snapshot(
        notes, events, t_ms, 1, APPROACH_MS, VisualMods(),
        consumed_times=compute_consumed_times(notes, events, 1),
        hold_visual_states=states,
        timing_points=timing_points,
        sv_table=build_sv_distance_table(timing_points),
    )


def test_successful_hold_freezes_head_but_keeps_scrolling_body_position():
    events = (KeyEvent(1000, 1), KeyEvent(2000, 0))
    before = _hold_snapshot(999, events).visible_notes[0]
    assert not before.hold_head_hit
    assert before.hold_clip_head_y_fraction is None
    held = _hold_snapshot(1500, events).visible_notes[0]
    assert held.hold_head_hit and held.hold_active
    assert held.head_y_fraction == held.hold_clip_head_y_fraction == 1
    assert held.body_head_y_fraction == pytest.approx(1 + 500 / APPROACH_MS)


def test_missed_head_and_wide_attempt_do_not_freeze_or_mask():
    events = (KeyEvent(800, 1), KeyEvent(1800, 0))
    # The attempt map includes this press, but the scored head was missed.
    assert compute_consumed_times((HoldNote(0, 1000, 2000),), events, 1) == {(0, 1000): 800}
    for t_ms in (1500, 2050):
        hold = _hold_snapshot(t_ms, events).visible_notes[0]
        assert not hold.hold_head_hit and not hold.hold_active
        assert hold.hold_clip_head_y_fraction is None
        assert hold.head_y_fraction == hold.body_head_y_fraction > 1
    assert _hold_snapshot(2250, events).visible_notes == ()


def test_early_successful_press_does_not_consume_before_scheduled_start():
    events = (KeyEvent(950, 1), KeyEvent(2000, 0))
    early = _hold_snapshot(975, events).visible_notes[0]
    assert early.hold_head_hit and not early.hold_active
    assert early.head_y_fraction < 1
    assert early.hold_clip_head_y_fraction is None
    started = _hold_snapshot(1000, events).visible_notes[0]
    assert started.hold_active and started.hold_clip_head_y_fraction == 1


def test_late_head_is_not_frozen_until_the_actual_press():
    events = (KeyEvent(1050, 1), KeyEvent(2000, 0))
    before = _hold_snapshot(1049, events).visible_notes[0]
    assert before.head_y_fraction > 1
    assert not before.hold_head_hit and before.hold_clip_head_y_fraction is None
    hit = _hold_snapshot(1050, events).visible_notes[0]
    assert hit.hold_active and hit.head_y_fraction == 1


def test_drop_moves_head_and_mask_from_release_and_repress_never_refreezes():
    events = (KeyEvent(1000, 1), KeyEvent(1300, 0), KeyEvent(1400, 1), KeyEvent(2000, 0))
    at_drop = _hold_snapshot(1300, events).visible_notes[0]
    assert not at_drop.hold_active
    assert at_drop.head_y_fraction == at_drop.hold_clip_head_y_fraction == 1
    scene = _hold_snapshot(1500, events)
    hold = scene.visible_notes[0]
    assert scene.keys_held == (True,)
    assert hold.hold_head_hit and not hold.hold_active
    assert hold.head_y_fraction == hold.hold_clip_head_y_fraction == pytest.approx(1 + 200 / 600)
    assert hold.body_head_y_fraction == pytest.approx(1 + 500 / 600)
    assert _hold_snapshot(2050, events).visible_notes  # dropped tail can pass through


def test_release_before_start_unfreezes_from_scheduled_start():
    events = (KeyEvent(950, 1), KeyEvent(975, 0))
    assert _hold_snapshot(990, events).visible_notes[0].hold_clip_head_y_fraction is None
    hold = _hold_snapshot(1060, events).visible_notes[0]
    assert not hold.hold_active
    assert hold.head_y_fraction == hold.hold_clip_head_y_fraction == pytest.approx(1.1)


def test_drop_uses_integrated_sv_distance_for_travelling_mask():
    events = (KeyEvent(1000, 1), KeyEvent(1300, 0))
    points = (TimingPoint(0, 1, 0, 100), TimingPoint(1400, 1, 0, 100, sv_multiplier=2))
    hold = _hold_snapshot(1600, events, timing_points=points).visible_notes[0]
    assert hold.head_y_fraction == pytest.approx(1 + (100 + 400) / 600)
    assert hold.body_head_y_fraction == pytest.approx(1 + (400 + 400) / 600)


def test_tail_finish_and_held_late_window_follow_stable_lifetime():
    released = (KeyEvent(1000, 1), KeyEvent(1990, 0))
    assert _hold_snapshot(1989, released).visible_notes[0].hold_active
    assert _hold_snapshot(1990, released).visible_notes == ()
    still_held = (KeyEvent(1000, 1),)
    assert _hold_snapshot(2126, still_held).visible_notes[0].head_y_fraction == 1
    assert _hold_snapshot(2127, still_held).visible_notes == ()


def test_visual_state_requires_a_head_match_and_real_rising_edge():
    notes = (HoldNote(0, 1000, 2000),)
    tail_only = (JudgmentEvent(1000, 0, "geki", 0, is_tail=True),)
    assert build_hold_visual_states(notes, tail_only, [[1000]], [[2000]], 127) == {}
    phantom = (JudgmentEvent(1000, 0, "geki", 0),)
    assert build_hold_visual_states(notes, phantom, [[]], [[]], 127) == {}


def test_aggregate_reconciliation_does_not_rewrite_hold_lifecycle_or_scores():
    notes = (HoldNote(0, 1000, 2000),)
    events = (KeyEvent(1000, 1), KeyEvent(1300, 0))
    original = compute_judgments(notes, events, 1)
    reconciled = reconcile_to_counts(original, 0, 0, 0, 0, 0, 1)
    assert reconciled.events[0].judgment == "miss"
    presses, releases = _key_edges_per_col(events, 1)
    states = build_hold_visual_states(notes, original.events, presses, releases, 127)
    assert states[(0, 1000)].head_hit_time_ms == 1000
    assert states[(0, 1000)].drop_time_ms == 1300
    assert original.events[0].judgment == "geki"
    assert reconciled.events[0].judgment == "miss"
