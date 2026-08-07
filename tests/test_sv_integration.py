"""Unit tests for the lazer-faithful piecewise-constant SV integration
introduced 2026-05. Validates the cumulative-distance helpers in
`osu_mania_renderer_v2.beatmap` and the `_y_integrated` formula in
`osu_mania_renderer_v2.scene`.

The old point-sample approach was broken on charts with mid-song SV
changes (notes warped at section boundaries). These tests pin down
the new behaviour so regressions show up immediately."""
from osu_mania_renderer_v2.beatmap.beatmap import (
    build_sv_distance_table,
    sv_distance_at,
)
from osu_mania_renderer_v2.beatmap.models import TimingPoint
from osu_mania_renderer_v2.render.scene import _y_integrated


def _tp(time_ms: int, sv: float) -> TimingPoint:
    """Shorthand: inherited TP with the given SV multiplier."""
    return TimingPoint(
        time_ms=time_ms,
        sample_set=0, custom_index=0, volume=100,
        sv_multiplier=sv, uninherited=False, beat_length_ms=500.0,
    )


def test_empty_timing_points_table():
    """No TPs → empty table; sv_distance_at returns t_ms unchanged."""
    assert build_sv_distance_table(()) == ()
    assert sv_distance_at(1000, (), ()) == 1000.0


def test_single_tp_sv_one_is_identity():
    """SV=1 throughout → cumulative distance equals elapsed time."""
    tps = (_tp(0, 1.0),)
    table = build_sv_distance_table(tps)
    assert table == (0.0,)
    assert sv_distance_at(0, tps, table) == 0.0
    assert sv_distance_at(500, tps, table) == 500.0
    assert sv_distance_at(2000, tps, table) == 2000.0


def test_single_tp_sv_two_doubles_distance():
    """SV=2 throughout → distance accumulates at 2× elapsed time."""
    tps = (_tp(0, 2.0),)
    table = build_sv_distance_table(tps)
    assert sv_distance_at(0, tps, table) == 0.0
    assert sv_distance_at(500, tps, table) == 1000.0
    assert sv_distance_at(1000, tps, table) == 2000.0


def test_two_section_boundary_no_warp():
    """0..1000 ms at SV=1, then 1000..2000 ms at SV=2.
    At the boundary, the cumulative distance must be continuous —
    no jump. This is the property the old code violated."""
    tps = (_tp(0, 1.0), _tp(1000, 2.0))
    table = build_sv_distance_table(tps)
    # At time=999 ms (last instant of first section): distance ≈ 999.
    assert abs(sv_distance_at(999, tps, table) - 999.0) < 1e-9
    # At time=1000 ms (boundary): exactly 1000 (sum of first section).
    assert sv_distance_at(1000, tps, table) == 1000.0
    # At time=1001 ms (1 ms into second section at SV=2):
    # 1000 + 1*2 = 1002. Continuity preserved.
    assert sv_distance_at(1001, tps, table) == 1002.0
    # At time=2000 ms (1000 ms into second section at SV=2):
    # 1000 + 1000*2 = 3000.
    assert sv_distance_at(2000, tps, table) == 3000.0


def test_three_sections_compound_integration():
    """Three sections: 1×, 2×, 0.5×. Distance accumulates per-section."""
    tps = (_tp(0, 1.0), _tp(100, 2.0), _tp(200, 0.5))
    table = build_sv_distance_table(tps)
    # End of section 1: 100ms * 1.0 = 100.
    assert sv_distance_at(100, tps, table) == 100.0
    # End of section 2: 100 + 100ms * 2.0 = 300.
    assert sv_distance_at(200, tps, table) == 300.0
    # 100ms into section 3: 300 + 100ms * 0.5 = 350.
    assert sv_distance_at(300, tps, table) == 350.0


def test_pre_first_tp_is_treated_as_sv1():
    """A time before the first TP integrates at SV=1 from t=0."""
    tps = (_tp(500, 2.0),)
    table = build_sv_distance_table(tps)
    # Times before the first TP: distance = elapsed time (SV=1).
    assert sv_distance_at(0, tps, table) == 0.0
    assert sv_distance_at(250, tps, table) == 250.0
    # At the TP: cumulative = 500 (still pre-TP convention).
    assert sv_distance_at(500, tps, table) == 500.0
    # After the TP, SV=2 applies: 500 + 100*2 = 700.
    assert sv_distance_at(600, tps, table) == 700.0


def test_y_integrated_matches_legacy_at_sv1():
    """When SV is constant 1 across the map, the integrated _y must
    equal what the legacy point-sample _y produced. This is the
    "no regression on simple maps" guarantee."""
    tps = (_tp(0, 1.0),)
    table = build_sv_distance_table(tps)
    approach_ms = 600
    t_now = 0
    note_time = 400
    current_cum = sv_distance_at(t_now, tps, table)
    y = _y_integrated(note_time, current_cum, approach_ms, tps, table)
    # Legacy: 1 - (400 - 0) * 1 / 600 = 0.3333
    assert abs(y - (1.0 - 400 / 600)) < 1e-9


def test_y_integrated_uses_integrated_distance_not_local_sv():
    """The key bug we're fixing: a note positioned where SV varies
    between t_now and note_time. Legacy code used SV at note's time;
    integration uses ∫SV dt."""
    # SV=1 from 0..500ms, SV=2 from 500..1000ms.
    # Note at time=1000 (deep in SV=2 section).
    # Receptor at time=0.
    tps = (_tp(0, 1.0), _tp(500, 2.0))
    table = build_sv_distance_table(tps)
    approach_ms = 1000
    t_now = 0
    note_time = 1000
    current_cum = sv_distance_at(t_now, tps, table)
    # Note's cumulative distance: 500*1 + 500*2 = 1500.
    # Receptor's cumulative distance: 0.
    # y = 1 - (1500 - 0) / 1000 = -0.5 (above the top — not yet visible).
    y = _y_integrated(note_time, current_cum, approach_ms, tps, table)
    assert abs(y - (-0.5)) < 1e-9
    # Legacy formula with SV=2 (note's own SV) would have given:
    # 1 - (1000 - 0) * 2 / 1000 = -1.0 — overshooting.
    # Integration is more accurate (note actually traversed 500ms of
    # SV=1 first, so its cumulative is 1500, not 2000).


def test_y_integrated_note_arriving_at_receptor():
    """As t_now approaches note_time, y → 1 (receptor row)."""
    tps = (_tp(0, 1.5),)
    table = build_sv_distance_table(tps)
    approach_ms = 600
    note_time = 1000
    current_cum = sv_distance_at(note_time, tps, table)
    y = _y_integrated(note_time, current_cum, approach_ms, tps, table)
    assert abs(y - 1.0) < 1e-9


def test_clamp_widened_to_50():
    """The 2026-05 widening of the SV clamp from 8× to 50× must
    pass through extreme SVs on gimmick maps without crushing them."""
    # build_sv_distance_table just consumes whatever TPs we pass; the
    # clamp is in beatmap._parse_timing_block (not exercised here),
    # but we can still verify the integration handles 50× correctly.
    tps = (_tp(0, 50.0),)
    table = build_sv_distance_table(tps)
    # 100ms at SV=50 = 5000 distance.
    assert sv_distance_at(100, tps, table) == 5000.0
