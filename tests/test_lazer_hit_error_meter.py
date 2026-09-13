"""Focused regression coverage for the custom-skin lazer hit-error meter."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.beatmap.judgments import JudgmentEvent, windows_for_od
from osu_mania_renderer_v2.beatmap.models import VisualMods
from osu_mania_renderer_v2.gpu.renderer import (
    LAZER_HIT_RESULT_COLOURS,
    FrameRenderer,
    hit_error_offset_position,
    lazer_hit_error_meter_geometry,
    lazer_hit_error_tick_state,
    lazer_hit_window_bands,
    next_hit_error_ema,
)
from osu_mania_renderer_v2.render.render import build_frame_state
from osu_mania_renderer_v2.render.scene import HitErrorEvent


def test_lazer_meter_uses_768_reference_geometry():
    geometry = lazer_hit_error_meter_geometry(1280, 768)

    assert geometry.ui_scale == 1
    assert geometry.length == 200
    assert geometry.left == 540
    assert geometry.bar_thickness == 2
    assert geometry.judgment_width == 14
    assert geometry.judgment_thickness == 4
    assert geometry.chevron_size == 8
    assert geometry.centre_outer_size == 8
    assert geometry.centre_inner_size == 4


def test_lazer_meter_scales_to_720p_and_stays_bottom_centre():
    geometry = lazer_hit_error_meter_geometry(1280, 720)
    scale = 720 / 768

    assert geometry.ui_scale == pytest.approx(scale)
    assert geometry.length == pytest.approx(200 * scale)
    assert geometry.left == pytest.approx((1280 - 200 * scale) / 2)
    assert geometry.axis_y == pytest.approx(18 * scale)
    assert geometry.judgment_width == pytest.approx(14 * scale)
    assert geometry.judgment_thickness == pytest.approx(4 * scale)
    assert geometry.centre_outer_size == pytest.approx(8 * scale)
    assert geometry.centre_inner_size == pytest.approx(4 * scale)
    # The custom meter is global HUD chrome, not the 420px playfield strip.
    assert geometry.length != 420


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (-200, 0),
        (-127, 0),
        (0, 0.5),
        (127, 1),
        (200, 1),
    ],
)
def test_hit_offset_mapping_is_clamped(offset, expected):
    assert hit_error_offset_position(offset, 127) == expected


def test_hit_window_bands_use_actual_od_windows_and_lazer_colours():
    windows = windows_for_od(8.0)
    bands = lazer_hit_window_bands(windows)

    assert [band.judgment for band in bands] == [
        "geki", "300", "katu", "100", "50",
    ]
    assert [band.window_ms for band in bands] == pytest.approx(windows)
    assert [band.relative_length for band in bands] == pytest.approx(
        [window / windows[-1] for window in windows],
    )
    assert [band.colour for band in bands] == [
        LAZER_HIT_RESULT_COLOURS[judgment]
        for judgment in ("geki", "300", "katu", "100", "50")
    ]
    assert bands[-1].relative_length == 1


def test_timed_event_carries_real_age_judgment_and_offset():
    event = HitErrorEvent(offset_ms=-12.5, judgment="300", age_ms=2400)

    assert event == HitErrorEvent(-12.5, "300", 2400)


def test_frame_state_derives_real_tick_age_and_ema_from_judgment_timeline():
    judgment = JudgmentEvent(
        time_ms=1000, column=1, judgment="300", hit_offset_ms=20,
    )
    windows = windows_for_od(8.0)
    replay = SimpleNamespace(
        key_events=(), max_combo=1, accuracy=100.0,
        count_geki=0, count_300=1, count_katu=0,
        count_100=0, count_50=0, count_miss=0,
        mania_acc_weight=300,
    )
    plan = SimpleNamespace(
        replay=replay,
        modded=SimpleNamespace(notes=()),
        key_count=4,
        gameplay_end_ms=2000,
        results_start_ms=3000,
        effective_approach_ms=600,
        visual_mods=VisualMods(),
        judged_hits={},
        sv_for_note={},
        timing_points=(),
        sv_table=(),
        note_times=(),
        max_hold_dur_ms=0,
        judgment_events=(judgment,),
        judgment_timeline=[(1020, judgment)],
        hit_error_windows=windows,
        total_quality=300,
        kiai_ranges=[],
        per_column_ur=(0, 0, 0, 0),
        miss_break_times=[],
        press_iters=[[], [], [], []],
        acronyms=(),
        player_pp=0,
        max_pp=0,
        stars=0,
        mania_mw=300,
        n_scoring=1,
        max_combo_portion=300,
        score_scale=1,
        score_final=None,
    )

    scene, score_smoothed, accuracy_smoothed = build_frame_state(
        plan, 1050, 0, 100,
    )

    assert scene.hit_error_events == (HitErrorEvent(20.0, "300", 30),)
    assert scene.hit_error_windows == windows
    assert scene.hit_error_ema_ms == pytest.approx(2)

    expired, _, _ = build_frame_state(
        plan, 6120, score_smoothed, accuracy_smoothed,
    )
    assert expired.hit_error_events == ()
    assert expired.hit_error_ema_ms == pytest.approx(2)


def test_tick_uses_100ms_entrance_then_5000ms_exit():
    start = lazer_hit_error_tick_state(0)
    entering = lazer_hit_error_tick_state(50)
    full = lazer_hit_error_tick_state(100)
    fading = lazer_hit_error_tick_state(2600)
    expired = lazer_hit_error_tick_state(5100)

    assert start.alpha == start.width_fraction == 0
    assert 0 < entering.alpha < 0.6
    assert 0 < entering.width_fraction < 1
    assert full.alpha == pytest.approx(0.6)
    assert full.width_fraction == pytest.approx(1)
    assert fading.alpha == pytest.approx(0.3)
    assert fading.width_fraction == pytest.approx(1 - 0.5**5)
    assert expired.alpha == expired.width_fraction == 0


def test_moving_average_matches_lazer_fold():
    average = next_hit_error_ema(0, 20)
    assert average == pytest.approx(2)
    assert next_hit_error_ema(average, -10) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("is_argon", "expected"),
    [(True, "argon"), (False, "legacy")],
)
def test_argon_and_custom_legacy_paths_remain_separate(
    is_argon, expected, monkeypatch,
):
    from osu_mania_renderer_v2.wiki_elements import hud

    renderer = object.__new__(FrameRenderer)
    calls = []
    renderer._is_argon_default = lambda: is_argon
    renderer._draw_hit_strip = lambda scene: pytest.fail(
        "Argon must not use the old horizontal R3D strip",
    )
    renderer._shared_frame_context = lambda scene: ("ctx", scene)
    monkeypatch.setattr(
        hud, "_argon_hit_error",
        lambda ctx: calls.append(("argon", ctx[1])),
    )
    renderer._draw_lazer_hit_error_meter = (
        lambda scene: calls.append(("legacy", scene))
    )
    scene = SimpleNamespace()

    FrameRenderer._draw_hit_error_meter(renderer, scene)

    assert calls == [(expected, scene)]
