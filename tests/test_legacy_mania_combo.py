"""Focused custom legacy Mania combo presentation tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.beatmap.judgments import JudgmentEvent, windows_for_od
from osu_mania_renderer_v2.beatmap.models import VisualMods
from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection
from osu_mania_renderer_v2.gpu.renderer import (
    FrameRenderer,
    legacy_combo_break_animation,
    legacy_combo_y_scale,
    legacy_text_layout,
)
from osu_mania_renderer_v2.render.render import build_frame_state


def test_combo_increment_is_vertical_only_300ms_ease_out():
    assert legacy_combo_y_scale(0) == pytest.approx(1.4)
    assert legacy_combo_y_scale(150) == pytest.approx(1.1)
    assert legacy_combo_y_scale(300) == 1


def test_combo_break_burst_is_200ms_scale_and_fade():
    assert legacy_combo_break_animation(0) == pytest.approx((1.0, 0.8))
    assert legacy_combo_break_animation(100) == pytest.approx((2.5, 0.4))
    assert legacy_combo_break_animation(200) == (1.0, 0.0)


def test_combo_layout_uses_combo_slots_fixed_digits_and_overlap():
    layout = legacy_text_layout(
        "15", {"1": (10, 40), "5": (20, 40)},
        font="combo", scale=1, overlap=4,
    )

    assert layout.width == 36
    assert [glyph.slot for glyph in layout.glyphs] == ["combo_1", "combo_5"]
    assert layout.glyphs[0].x == 5
    assert layout.glyphs[1].x == 16


class _ComboAtlas:
    def global_source(self, slot):
        return "user" if slot.startswith("combo_") else "missing"

    def global_native_size(self, _slot):
        return 20.0, 40.0

    def global_has_visible_pixels(self, _slot):
        return True

    def index_of(self, slot):
        return int(slot[-1])


def _renderer():
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=768)
    renderer.options = SimpleNamespace(show_combo=True)
    renderer.pf_x = 440
    renderer.pf_w = 400
    renderer.combo_baseline_y_gl = 300
    renderer.skin_ini = SimpleNamespace(combo_overlap=4)
    renderer.mania_section = ManiaSection(keys=4)
    renderer.atlas = _ComboAtlas()
    renderer.normal = []
    renderer.additive = []
    renderer._draw_sprite_idx = lambda *args: renderer.normal.append(args)
    renderer._draw_additive_sprite_idx = lambda *args: renderer.additive.append(args)
    renderer._cached_text = lambda *_args: pytest.fail("PIL combo fallback used")
    return renderer


def test_custom_combo_draws_skin_glyphs_white_at_combo_position():
    renderer = _renderer()
    scene = SimpleNamespace(
        combo=15,
        combo_age_ms=0,
        combo_break_age_ms=9999,
        combo_break_previous_value=0,
    )

    FrameRenderer._draw_custom_legacy_combo(renderer, scene, draw_combo=True)

    assert len(renderer.normal) == 2
    assert {draw[-1] for draw in renderer.normal} == {(1.0, 1.0, 1.0, 1.0)}
    assert all(draw[4] == 56 for draw in renderer.normal)
    # The two fixed-width cells remain centred on the playfield (x=640).
    assert min(draw[1] for draw in renderer.normal) == 622
    assert max(draw[1] + draw[3] for draw in renderer.normal) == 658


def test_combo_break_draws_previous_value_red_and_additive():
    renderer = _renderer()
    scene = SimpleNamespace(
        combo=0,
        combo_age_ms=100,
        combo_break_age_ms=100,
        combo_break_previous_value=25,
    )

    FrameRenderer._draw_custom_legacy_combo(renderer, scene, draw_combo=True)

    assert not renderer.normal
    assert len(renderer.additive) == 2
    assert all(draw[-1] == (1.0, 0.0, 0.0, 0.4) for draw in renderer.additive)
    assert all(draw[3:5] == (50, 100) for draw in renderer.additive)


def test_frame_state_carries_previous_combo_value_for_break_burst():
    hit = JudgmentEvent(
        time_ms=100, column=0, judgment="300", hit_offset_ms=0,
    )
    miss = JudgmentEvent(time_ms=200, column=1, judgment="miss")
    replay = SimpleNamespace(
        key_events=(), mods=0, max_combo=1, accuracy=50.0,
        count_geki=0, count_300=1, count_katu=0,
        count_100=0, count_50=0, count_miss=1,
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
        judgment_events=(hit, miss),
        judgment_timeline=[(100, hit), (200, miss)],
        hit_error_windows=windows_for_od(8.0),
        total_quality=300,
        kiai_ranges=[],
        per_column_ur=(0, 0, 0, 0),
        miss_break_times=[],
        press_iters=[[], [], [], []],
        acronyms=("4K",),
        player_pp=0,
        max_pp=0,
        stars=0,
        mania_mw=300,
        n_scoring=2,
        max_combo_portion=300,
        score_scale=1,
        score_final=None,
    )

    scene, _score, _accuracy = build_frame_state(plan, 250, 0, 100)

    assert scene.combo == 0
    assert scene.combo_break_previous_value == 1
    assert scene.combo_break_age_ms == 50
    assert scene.replay_mods == 0
