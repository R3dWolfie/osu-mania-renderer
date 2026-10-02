"""TheAussie Mania HEM port (historical filename kept for focused suites)."""

from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.beatmap.judgments import JudgmentEvent, windows_for_od
from osu_mania_renderer_v2.beatmap.models import RenderOptions, VisualMods
from osu_mania_renderer_v2.gpu.hit_error_meter import (
    BLUE,
    BOTTOM_MARGIN,
    CENTRE_DARK_BLUE,
    GREEN,
    HIT_RESULT_COLOURS,
    NATURAL_HEIGHT,
    REFERENCE_HEIGHT,
    YELLOW,
    HitErrorChevronTransition,
    hit_error_chevron_position,
    hit_error_edge_alpha,
    hit_error_meter_geometry,
    hit_error_offset_position,
    hit_error_tick_state,
    hit_window_bands,
    next_hit_error_chevron_transition,
    next_hit_error_ema,
)
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer
from osu_mania_renderer_v2.render.render import build_frame_state
from osu_mania_renderer_v2.render.scene import HitErrorEvent
from osu_mania_renderer_v2.wiki_elements import hud


def test_bottom_anchor_preserves_x_origin_and_horizontal_geometry():
    g = hit_error_meter_geometry(1280, 720)
    assert g.scale == 1
    assert (REFERENCE_HEIGHT, NATURAL_HEIGHT, BOTTOM_MARGIN) == (720, 22, 8)
    assert (g.component_x, g.component_top) == (540, 690)
    assert (g.natural_width, g.natural_height) == (244, 22)
    assert 720 - (g.component_top + g.natural_height) == 8 * g.scale
    assert (g.axis_start, g.axis_centre, g.cross_centre) == (22, 122, 7)
    assert g.early_icon_centre == (8, 7)
    assert g.late_icon_centre == (236, 7)
    assert [g.axis_point(p) for p in (0, 0.5, 1)] == [22, 122, 222]
    assert g.marker_segment(0.5, 1) == ((122, 0), (122, 14))
    assert g.chevron_points(0.5) == ((118, 21), (122, 15), (126, 21))
    assert 720 - g.component_top - g.cross_centre == 23


def test_1080_geometry_scales_every_local_coordinate_by_1_5():
    g = hit_error_meter_geometry(1920, 1080)
    assert g.scale == 1.5
    assert (g.component_x, g.component_top) == (810, 1035)
    assert (g.natural_width, g.natural_height) == (366, 33)
    assert 1080 - (g.component_top + g.natural_height) == 12 == 8 * g.scale
    assert (g.axis_start, g.axis_centre, g.cross_centre) == (33, 183, 10.5)
    assert g.early_icon_centre == (12, 10.5)
    assert g.late_icon_centre == (354, 10.5)
    assert g.chevron_points(0.5) == ((177, 31.5), (183, 22.5), (189, 31.5))


@pytest.mark.parametrize("width,height", [(1024, 768), (2560, 1080), (900, 720)])
def test_non_16_9_keeps_height_scale_and_centres_reference_surface(width, height):
    g = hit_error_meter_geometry(width, height)
    scale = height / 720
    assert g.component_x == pytest.approx((width - 1280 * scale) / 2 + 540 * scale)
    assert g.component_top == 690 * scale
    assert height - (g.component_top + g.natural_height) == pytest.approx(8 * scale)


@pytest.mark.parametrize("offset,expected", [(-200, 0), (-127, 0), (0, 0.5), (127, 1), (200, 1)])
def test_hit_offset_mapping_is_clamped(offset, expected):
    assert hit_error_offset_position(offset, 127) == expected


@pytest.mark.parametrize("offset,window", [(float("nan"), 100), (10, 0), (10, float("inf"))])
def test_invalid_offset_mapping_is_neutral(offset, window):
    assert hit_error_offset_position(offset, window) == 0.5


def test_three_bands_use_perfect_good_meh_and_hard_inner_rects():
    bands = hit_window_bands((20, 40, 60, 80, 100))
    assert [b.window_ms for b in bands] == [20, 60, 100]
    assert [b.relative_length for b in bands] == [0.2, 0.6, 1]
    assert [b.colour for b in bands] == [BLUE, GREEN, YELLOW]
    g = hit_error_meter_geometry(1280, 720)
    assert [g.band_rect(b.relative_length) for b in bands] == [
        (102, 6, 40, 2),
        (62, 6, 120, 2),
        (22, 6, 200, 2),
    ]
    assert HIT_RESULT_COLOURS["geki"] == HIT_RESULT_COLOURS["300"] == BLUE
    assert HIT_RESULT_COLOURS["katu"] == HIT_RESULT_COLOURS["100"] == GREEN
    assert HIT_RESULT_COLOURS["50"] == YELLOW
    assert CENTRE_DARK_BLUE == (0x47 / 255, 0x8F / 255, 0xB3 / 255)


@pytest.mark.parametrize("scale", [1, 1.5, 2])
def test_outer_tip_fade_is_linear_six_scaled_pixels_only(scale):
    span, fade = 200 * scale, 6 * scale
    for axis, expected in [(0, 0), (1, 1 / 6), (3, 0.5), (6, 1), (20, 1), (100, 1)]:
        assert hit_error_edge_alpha(axis * scale, span, fade) == pytest.approx(expected)
        assert hit_error_edge_alpha(span - axis * scale, span, fade) == pytest.approx(expected)


def test_tick_uses_exact_100ms_outquint_then_5000ms_exit():
    assert hit_error_tick_state(-1).alpha == hit_error_tick_state(0).alpha == 0
    assert hit_error_tick_state(50).alpha == pytest.approx(0.6 * (1 - 0.5**5))
    assert hit_error_tick_state(50).width_fraction == 1 - 0.5**5
    assert hit_error_tick_state(100).alpha == 0.6
    assert hit_error_tick_state(100).width_fraction == 1
    assert hit_error_tick_state(2600).alpha == 0.3
    assert hit_error_tick_state(2600).width_fraction == 1 - 0.5**5
    assert hit_error_tick_state(5099).alpha > 0
    assert hit_error_tick_state(5100).alpha == 0


def _plan(entries):
    """Minimal production frame-state fixture, not painter-only fake markers."""
    replay = SimpleNamespace(
        key_events=(),
        max_combo=100,
        accuracy=100.0,
        count_geki=0,
        count_300=len(entries),
        count_katu=0,
        count_100=0,
        count_50=0,
        count_miss=0,
        mania_acc_weight=300,
    )
    return SimpleNamespace(
        replay=replay,
        modded=SimpleNamespace(notes=()),
        key_count=4,
        gameplay_end_ms=10000,
        results_start_ms=11000,
        effective_approach_ms=600,
        visual_mods=VisualMods(),
        judged_hits={},
        sv_for_note={},
        timing_points=(),
        sv_table=(),
        note_times=(),
        max_hold_dur_ms=0,
        judgment_events=tuple(j for _, j in entries),
        judgment_timeline=list(entries),
        hit_error_windows=windows_for_od(8.0),
        total_quality=300 * len(entries),
        kiai_ranges=[],
        per_column_ur=(0, 0, 0, 0),
        miss_break_times=[],
        press_iters=[[], [], [], []],
        release_iters=[[], [], [], []],
        acronyms=(),
        player_pp=0,
        max_pp=0,
        stars=0,
        mania_mw=300,
        n_scoring=len(entries),
        max_combo_portion=300,
        score_scale=1,
        score_final=None,
    )


def _state(plan, time):
    return build_frame_state(plan, time, 0, 100)[0]


def test_frame_state_keeps_real_scored_sample_ages_ema_and_ur_semantics():
    plan = _plan(
        [
            (1020, JudgmentEvent(1000, 1, "300", 20)),
            (1100, JudgmentEvent(1100, 1, "miss", None)),
            (1200, JudgmentEvent(1180, 1, "300", 20, is_tail=True, scoring=False)),
        ]
    )
    assert _state(plan, 999).hit_error_chevron_position is None
    scene = _state(plan, 1250)
    assert scene.hit_error_events == (HitErrorEvent(20.0, "300", 230),)
    assert scene.hit_error_ema_ms == 2
    assert scene.avg_hit_offset_ms == 20
    assert scene.unstable_rate == 0
    assert len(scene.recent_offsets) == 2  # UR's existing tail inclusion is preserved
    expired = _state(plan, 6120)
    assert expired.hit_error_events == ()
    assert expired.hit_error_ema_ms == 2
    assert expired.hit_error_chevron_position == hit_error_offset_position(
        2, plan.hit_error_windows[-1]
    )


def test_frame_state_caps_at_newest_fifty_scored_markers():
    plan = _plan([(1000 + i, JudgmentEvent(1000 + i, i % 4, "geki", float(i))) for i in range(60)])
    scene = _state(plan, 1100)
    assert len(scene.hit_error_events) == 50
    assert [e.offset_ms for e in scene.hit_error_events] == list(range(10, 60))


def test_moving_average_exact_fold_starts_from_zero():
    assert next_hit_error_ema(0, 10) == 1
    average = 0
    for offset in (10, -10, 20):
        average = next_hit_error_ema(average, offset)
    assert average == (1 * 0.9 - 10 * 0.1) * 0.9 + 20 * 0.1


def test_800ms_outquint_start_midpoint_and_completed_target():
    transition = next_hit_error_chevron_transition(None, 1000, 20, 100)
    assert transition == HitErrorChevronTransition(1000, 0.5, 0.6)
    assert hit_error_chevron_position(transition, 1000) == 0.5
    assert hit_error_chevron_position(transition, 1400) == 0.5 + (0.6 - 0.5) * (1 - 0.5**5)
    assert hit_error_chevron_position(transition, 1800) == 0.6
    assert hit_error_chevron_position(transition, 2500) == 0.6


def test_interrupted_transform_starts_at_actual_displayed_position():
    first = next_hit_error_chevron_transition(None, 1000, 20, 100)
    second = next_hit_error_chevron_transition(first, 1400, -20, 100)
    reached = 0.5 + (0.6 - 0.5) * (1 - 0.5**5)
    assert second.start_position == reached
    assert hit_error_chevron_position(second, 1400) == reached
    assert hit_error_chevron_position(second, 2200) == 0.4


def test_frame_state_direct_incremental_and_backward_seek_are_identical():
    entries = [
        (1000, JudgmentEvent(990, 1, "geki", 10)),
        (1300, JudgmentEvent(1315, 2, "300", -15)),
        (2300, JudgmentEvent(2275, 3, "katu", 25)),
        (2300, JudgmentEvent(2320, 0, "300", -20)),
    ]
    times = [900, 1000, 1150, 1300, 1600, 2100, 2300, 2700, 3100]
    direct = [_state(_plan(entries), t) for t in times]
    forward_plan = _plan(entries)
    forward = [_state(forward_plan, t) for t in times]
    backward = [_state(forward_plan, t) for t in reversed(times)][::-1]
    for states in (forward, backward):
        for got, expected in zip(states, direct, strict=True):
            assert got.hit_error_events == expected.hit_error_events
            assert got.hit_error_ema_ms == expected.hit_error_ema_ms
            assert got.hit_error_chevron_position == expected.hit_error_chevron_position
            assert got.avg_hit_offset_ms == expected.avg_hit_offset_ms
            assert got.unstable_rate == expected.unstable_rate
    transition, average = None, 0
    for time, judgment in entries:
        average = next_hit_error_ema(average, judgment.hit_offset_ms)
        transition = next_hit_error_chevron_transition(
            transition, time, average, windows_for_od(8)[-1]
        )
    assert direct[-1].hit_error_chevron_position == hit_error_chevron_position(transition, 3100)


@pytest.mark.parametrize("is_argon", [True, False])
def test_both_skins_and_both_paths_call_same_horizontal_painter(is_argon):
    scene, calls = SimpleNamespace(), []
    renderer = object.__new__(FrameRenderer)
    renderer._is_argon_default = lambda: is_argon
    renderer._hit_error_painter = SimpleNamespace(draw=lambda s: calls.append(s))
    FrameRenderer._draw_hit_error_meter(renderer, scene)
    ctx = SimpleNamespace(
        fr=renderer, scene=scene, options=RenderOptions(resolution=(1280, 720), fps=60)
    )
    hud.hit_strip(element=None, skin=None, assets=None, variables=None, ctx=ctx)
    assert calls == [scene, scene]
    assert not hasattr(hud, "_argon_hit_error")
