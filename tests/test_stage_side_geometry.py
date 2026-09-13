"""Stable Mania stage-left/right placement regression coverage."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.gpu.renderer import (
    FrameRenderer,
    mania_health_geometry,
    mania_stage_side_geometry,
)


def test_stage_sides_use_distinct_stable_edge_anchors_at_720p():
    geometry = mania_stage_side_geometry(
        playfield_left=430,
        playfield_right=850,
        stage_height=720,
        left_native_width=800,
        right_native_width=800,
    )

    assert geometry.texture_scale == pytest.approx(0.9375)
    assert geometry.left_rect == pytest.approx((-320, 0, 750, 720))
    assert geometry.right_rect == pytest.approx((850, 0, 750, 720))
    assert geometry.left_rect[0] + geometry.left_rect[2] == pytest.approx(430)
    assert geometry.right_rect[0] == pytest.approx(850)
    # The real skin is 800x770 rather than the canonical 768-high side canvas;
    # stable's explicit full-height vector scale differs by only 0.26%.
    rendered_aspect = geometry.left_rect[2] / geometry.left_rect[3]
    assert rendered_aspect == pytest.approx(800 / 770, rel=0.003)


@pytest.mark.parametrize("stage_height", [480, 720, 1080])
def test_stage_side_scale_preserves_768_authored_aspect(stage_height):
    geometry = mania_stage_side_geometry(
        playfield_left=500,
        playfield_right=900,
        stage_height=stage_height,
        left_native_width=320,
        right_native_width=160,
    )

    left_width = geometry.left_rect[2]
    right_width = geometry.right_rect[2]
    assert left_width / stage_height == pytest.approx(320 / 768)
    assert right_width / stage_height == pytest.approx(160 / 768)
    assert geometry.left_rect[0] + left_width == pytest.approx(500)
    assert geometry.right_rect[0] == pytest.approx(900)


class _Atlas:
    def global_source(self, name):
        return {
            "stage_left": "user",
            "stage_right": "user",
            "playfield_frame": "missing",
        }.get(name, "missing")

    def global_native_size(self, name):
        return {
            "stage_left": (800, 770),
            "stage_right": (800, 770),
        }.get(name, (0, 0))

    def global_aspect(self, _name):
        return 1.0


def test_stage_decorations_draw_correct_full_resolution_asset_on_each_side():
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=720)
    renderer.pf_x = 430
    renderer.pf_w = 420
    renderer.col_w_uniform = 105
    renderer.receptor_centre_y_gl = 36
    renderer.upside_down = False
    renderer.atlas = _Atlas()
    renderer.direct_draws = []
    renderer.sprite_draws = []
    renderer._draw_direct = lambda *args, **kwargs: renderer.direct_draws.append(
        (args, kwargs),
    )
    renderer._draw_sprite = lambda *args, **kwargs: renderer.sprite_draws.append(
        (args, kwargs),
    )

    FrameRenderer._draw_stage_decorations(renderer)

    assert renderer.direct_draws == [
        (("stage_left", -320, 0, 750, 720), {"tint": (1, 1, 1, 1)}),
        (("stage_right", 850, 0, 750, 720), {"tint": (1, 1, 1, 1)}),
    ]
    assert all(
        args[0] not in ("stage_left", "stage_right")
        for args, _ in renderer.sprite_draws
    )


def test_right_stage_side_and_health_bar_keep_separate_stable_anchors():
    sides = mania_stage_side_geometry(430, 850, 720, 800, 800)
    health = mania_health_geometry(
        stage_right=850,
        stage_top=0,
        stage_height=720,
        background_native_size=(712, 100),
        fill_native_size=(695, 53),
        hp=0.5,
        new_default=False,
    )

    assert sides.right_rect[0] == 850
    assert health.background_anchor == pytest.approx((851.5, 720))
    assert health.background_anchor[0] > sides.right_rect[0]
