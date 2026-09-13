"""Legacy score/accuracy/mod HUD geometry regression coverage."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas
from osu_mania_renderer_v2.gpu.renderer import (
    FrameRenderer,
    legacy_hud_geometry,
)
from osu_mania_renderer_v2.wiki_elements.hud import hud as draw_wiki_hud


def test_legacy_hud_geometry_uses_768_reference_space():
    geometry = legacy_hud_geometry(768, score_native_height=100)

    assert geometry.ui_scale == 1
    assert geometry.score_height == 96
    assert geometry.accuracy_height == pytest.approx(57.6)
    assert geometry.score_right_margin == 10
    assert geometry.accuracy_right_margin == 17
    assert geometry.accuracy_top == 105
    assert geometry.mod_height == 48


def test_legacy_hud_geometry_scales_to_720p():
    geometry = legacy_hud_geometry(720, score_native_height=100)

    assert geometry.ui_scale == pytest.approx(0.9375)
    assert geometry.score_height == 90
    assert geometry.accuracy_height == pytest.approx(54)
    assert geometry.mod_height == 45


def test_equal_design_size_score_glyphs_match_at_1x_and_2x(tmp_path):
    one_x = tmp_path / "one-x"
    two_x = tmp_path / "two-x"
    one_x.mkdir()
    two_x.mkdir()
    Image.new("RGBA", (50, 100), (255, 255, 255, 255)).save(one_x / "score-0.png")
    Image.new("RGBA", (100, 200), (255, 255, 255, 255)).save(two_x / "score-0@2x.png")

    one_frames, one_source = SpriteAtlas._resolve_global(
        "score_0", skin_dir=one_x, beatmap_dir=None, section=None,
    )
    two_frames, two_source = SpriteAtlas._resolve_global(
        "score_0", skin_dir=two_x, beatmap_dir=None, section=None,
    )
    one_native_h = one_frames[0].height / one_frames[0].info["scale_adjust"]
    two_native_h = two_frames[0].height / two_frames[0].info["scale_adjust"]

    assert one_source == two_source == "user"
    assert one_native_h == two_native_h == 100
    assert legacy_hud_geometry(720, one_native_h) == legacy_hud_geometry(
        720, two_native_h,
    )


class _HudAtlas:
    def __init__(self, score_source="user", score_native_height=100):
        self.score_source = score_source
        self.score_native_height = score_native_height

    def global_source(self, name):
        return self.score_source if name == "score_0" else "missing"

    def global_native_size(self, name):
        return (50, self.score_native_height) if name == "score_0" else (0, 0)


def _old_gpu_hud_renderer(*, height=768, score_source="user"):
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=height)
    renderer.options = SimpleNamespace(show_score=True, show_pp_counter=False, show_mods=True)
    renderer.atlas = _HudAtlas(score_source=score_source)
    renderer._is_argon_default = lambda: score_source == "bundle"
    renderer.draws = []
    renderer._cached_text = lambda line, size, colour: (line, 200, 100)
    renderer._draw_external_texture = lambda texture, **kwargs: renderer.draws.append(
        (texture, kwargs)
    )
    renderer._draw_mode_pills = lambda scene, **kwargs: kwargs["anchor_y"]
    return renderer


def _scene():
    return SimpleNamespace(
        results_opacity=0,
        score=123456,
        score_smoothed=123456,
        accuracy=98.76,
        accuracy_smoothed=98.76,
        max_pp=0,
        mod_acronyms=(),
    )


def test_partial_custom_font_keeps_existing_raster_fallback_geometry():
    renderer = _old_gpu_hud_renderer()

    FrameRenderer._draw_hud(renderer, _scene())

    score = renderer.draws[0][1]
    accuracy = renderer.draws[1][1]
    assert (score["w"], score["h"]) == (192, 96)
    assert score["x"] + score["w"] == 1270
    assert score["y"] + score["h"] == 768
    assert (accuracy["w"], accuracy["h"]) == (116, 58)
    assert accuracy["x"] + accuracy["w"] == 1263
    assert accuracy["y"] + accuracy["h"] == 768 - 96 - 9


def test_old_gpu_bundle_hud_delegates_to_shared_argon_presentation():
    renderer = _old_gpu_hud_renderer(score_source="bundle")
    argon_scenes = []
    renderer._draw_argon_hud = lambda scene: argon_scenes.append(scene)

    scene = _scene()
    FrameRenderer._draw_hud(renderer, scene)

    assert argon_scenes == [scene]
    assert renderer.draws == []


@pytest.mark.parametrize(("height", "expected_height"), [(768, 48), (720, 45)])
def test_old_gpu_custom_mod_pill_uses_legacy_footprint(height, expected_height):
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=height)
    renderer._cached_text = lambda line, size, colour: (line, 20, 20)
    renderer.sprite_draws = []
    renderer.text_draws = []
    renderer._draw_sprite = lambda *args: renderer.sprite_draws.append(args)
    renderer._draw_external_texture = lambda texture, **kwargs: renderer.text_draws.append(
        (texture, kwargs)
    )

    FrameRenderer._draw_mode_pills(
        renderer,
        SimpleNamespace(mod_acronyms=("4K", "HD")),
        anchor_y=600,
        legacy_ui_scale=height / 768,
    )

    outer_pills = renderer.sprite_draws[::2]
    inner_pills = renderer.sprite_draws[1::2]
    assert [draw[4] for draw in outer_pills] == [expected_height, expected_height]
    assert inner_pills[0][1] - outer_pills[0][1] == round(expected_height * 0.06)
    if height == 720:
        assert [draw[1] for draw in outer_pills] == [1222, 1168]
        assert all(draw[1]["h"] == 22 for draw in renderer.text_draws)


def test_wiki_legacy_score_has_no_accuracy_top_margin():
    draws = []
    fr = SimpleNamespace(
        rc=SimpleNamespace(width=1280, height=768),
        atlas=_HudAtlas(),
        skin_ini=SimpleNamespace(score_overlap=0),
        mania_section=object(),
        options=SimpleNamespace(show_score=True, show_pp_counter=False, hud_opacity=1),
    )
    ctx = SimpleNamespace(
        fr=fr,
        scene=_scene(),
        options=fr.options,
        atlas=fr.atlas,
        skin_ini=fr.skin_ini,
        mania_section=fr.mania_section,
        key_count=4,
        persistent={},
        has_argon_font=lambda: False,
        has_score_font=lambda: True,
        draw_number=lambda text, **kwargs: draws.append((text, kwargs)),
    )

    draw_wiki_hud(element=None, skin=None, assets=None, variables=None, ctx=ctx)

    assert draws[0][1]["center_y"] == 720
    assert draws[1][1]["center_y"] == pytest.approx(634.2)
