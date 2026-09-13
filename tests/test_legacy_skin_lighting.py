"""Legacy custom-skin receptor and lighting regression coverage."""
from __future__ import annotations

from types import SimpleNamespace

import moderngl
import pytest

from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection
from osu_mania_renderer_v2.gpu.renderer import (
    FrameRenderer,
    legacy_hit_explosion_alpha,
    legacy_hit_explosion_frame,
    legacy_lighting_scale,
)


class _Atlas:
    def __init__(self, *, lighting_n: str = "missing", lighting_l: str = "missing"):
        self.sources = {"lighting_n": lighting_n, "lighting_l": lighting_l}

    def column_slot_index(self, kind, _column):
        return {"receptor_off": 10, "receptor_on": 11}[kind]

    def column_native_size(self, _kind, _column):
        # Real QA receptor @2x dimensions after ScaleAdjust.
        return 75.0, 187.5

    def column_aspect(self, _kind, _column):
        return 75.0 / 187.5

    def global_source(self, slot):
        return self.sources[slot]

    def global_native_size(self, slot):
        return {"lighting_n": (60.0, 30.0), "lighting_l": (40.0, 80.0)}[slot]

    def index_of(self, slot):
        return {"lighting_n": 20, "lighting_l": 30}[slot]

    def frame_count(self, _slot):
        return 1


def _renderer(*, lighting_n="missing", lighting_l="missing", argon=False, upside=False):
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=720, key_count=1)
    renderer.col_x = (100,)
    renderer.col_w = (105,)
    renderer.receptor_centre_y_gl = 72
    renderer.upside_down = upside
    renderer.mania_section = ManiaSection(keys=1, column_width=(70,))
    renderer.skin_ini = None
    renderer.atlas = _Atlas(lighting_n=lighting_n, lighting_l=lighting_l)
    renderer._is_argon_default = lambda: argon
    renderer._stage_light_fps = lambda _frames: 60.0

    renderer.normal_draws = []
    renderer.named_draws = []
    renderer.additive_draws = []
    renderer._draw_sprite_idx = lambda *args: renderer.normal_draws.append(args)
    renderer._draw_sprite = lambda *args: renderer.named_draws.append(args)
    renderer._draw_additive_sprite_idx = lambda *args: renderer.additive_draws.append(args)
    return renderer


def _scene(*, held=False, press_age=120, hit_age=9999, judgment=""):
    return SimpleNamespace(
        keys_held=(held,),
        key_press_age_ms=(press_age,),
        hit_light_age_ms=(hit_age,),
        hit_light_judgment=(judgment,),
    )


def test_latest_lighting_scale_prefers_explicit_width_and_falls_back_to_column_width():
    section = ManiaSection(
        keys=2,
        column_width=(70, 60),
        lighting_n_width=(45.0, 0.0),
        lighting_l_width=(0.0, 36.0),
    )

    assert legacy_lighting_scale(section, 0, "n") == pytest.approx(45.0 / 30.0)
    assert legacy_lighting_scale(section, 1, "n") == pytest.approx(60.0 / 30.0)
    assert legacy_lighting_scale(section, 0, "l") == pytest.approx(70.0 / 30.0)
    assert legacy_lighting_scale(section, 1, "l") == pytest.approx(36.0 / 30.0)
    assert legacy_lighting_scale(section, 0, "n", legacy_version=2.4) == 1.0


def test_hit_explosion_fade_and_frame_selection_are_deterministic():
    assert legacy_hit_explosion_alpha(0) == 0
    assert legacy_hit_explosion_alpha(40) == pytest.approx(0.5)
    assert legacy_hit_explosion_alpha(80) == 1
    assert legacy_hit_explosion_alpha(140) == pytest.approx(0.5)
    assert legacy_hit_explosion_alpha(200) == 0
    assert legacy_hit_explosion_frame(0, 12) == 0
    assert legacy_hit_explosion_frame(85, 12) == 5


def test_additive_atlas_draw_flushes_under_additive_then_restores_blend():
    renderer = object.__new__(FrameRenderer)
    gl = SimpleNamespace(
        blend_func=(moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA),
    )
    renderer.rc = SimpleNamespace(ctx=gl)
    events = []
    renderer._flush_sprite_batch = lambda: events.append(("flush", gl.blend_func))
    renderer._draw_sprite_idx = lambda *args: events.append(("draw", gl.blend_func, args))

    FrameRenderer._draw_additive_sprite_idx(
        renderer, 7, 10, 20, 30, 40, (1.0, 1.0, 1.0, 0.5),
    )

    assert events[0] == (
        "flush", (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA),
    )
    assert events[1][0:2] == ("draw", (moderngl.SRC_ALPHA, moderngl.ONE))
    assert events[2] == ("flush", (moderngl.SRC_ALPHA, moderngl.ONE))
    assert gl.blend_func == (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)


def test_custom_held_receptor_swaps_image_without_scale_bump():
    renderer = _renderer()

    FrameRenderer._draw_receptors(renderer, _scene(held=True))

    # Native design height 187.5 scaled by 720/768; width stays the column.
    assert renderer.normal_draws == [
        (11, 100, 0, 105, 176, (1, 1, 1, 1)),
    ]


def test_custom_upscroll_receptor_uses_native_height_at_stage_top():
    renderer = _renderer(upside=True)

    FrameRenderer._draw_receptors(renderer, _scene())

    assert renderer.normal_draws == [
        (10, 100, 720 - 176, 105, 176, (1, 1, 1, 1)),
    ]


def test_custom_hold_lighting_uses_width_native_aspect_and_fades_in():
    renderer = _renderer(lighting_l="user")
    renderer.mania_section = ManiaSection(
        keys=1,
        column_width=(70,),
        lighting_l_width=(30.0,),
    )

    FrameRenderer._draw_receptors(
        renderer, _scene(held=True, press_age=40),
    )

    assert renderer.additive_draws == [
        (30, 133, 35, 38, 75, (1.0, 1.0, 1.0, 0.5)),
    ]


def test_custom_hit_lighting_keeps_fixed_native_aspect_during_fade():
    renderer = _renderer(lighting_n="user")
    renderer.mania_section = ManiaSection(
        keys=1,
        column_width=(70,),
        lighting_n_width=(45.0,),
    )

    FrameRenderer._draw_receptors(
        renderer, _scene(hit_age=40, judgment="300"),
    )
    first = renderer.additive_draws[-1]
    renderer.additive_draws.clear()
    FrameRenderer._draw_receptors(
        renderer, _scene(hit_age=140, judgment="300"),
    )
    second = renderer.additive_draws[-1]

    assert first[1:5] == (110, 51, 84, 42)
    assert second[1:5] == first[1:5]
    assert first[-1] == (1.0, 1.0, 1.0, 0.5)
    assert second[-1] == (1.0, 1.0, 1.0, 0.5)


def test_custom_hit_position_is_not_shifted_by_argon_receptor_clamp():
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=720, key_count=4)
    renderer.mania_section = ManiaSection(
        keys=4,
        column_width=(70, 70, 70, 70),
        hit_position=432,
        upside_down=True,
    )
    renderer._is_argon_default = lambda: False

    FrameRenderer._compute_playfield_geometry(renderer)

    assert renderer.col_w == (105, 105, 105, 105)
    assert renderer.receptor_centre_y_gl == 648


@pytest.mark.parametrize("source", ["user", "bundle", "missing"])
def test_custom_lighting_never_falls_back_to_synthetic_circle(source):
    renderer = _renderer(lighting_n=source)

    FrameRenderer._draw_receptors(
        renderer, _scene(hit_age=100, judgment="300"),
    )

    assert all(draw[0] != "note_circle" for draw in renderer.named_draws)
    if source == "user":
        assert len(renderer.additive_draws) == 1
        # Authored colour is preserved: no judgement-result RGB tint.
        assert renderer.additive_draws[0][-1][:3] == (1.0, 1.0, 1.0)
    else:
        assert renderer.additive_draws == []


def test_argon_default_delegates_to_shared_argon_receptors():
    renderer = _renderer(argon=True)
    argon_scenes = []
    renderer._draw_argon_receptors = lambda scene: argon_scenes.append(scene)

    scene = _scene(held=True, hit_age=100, judgment="300")
    FrameRenderer._draw_receptors(renderer, scene)

    assert argon_scenes == [scene]
    assert renderer.normal_draws == []
    assert renderer.named_draws == []
