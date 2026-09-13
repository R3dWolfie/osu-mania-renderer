"""Stable-style legacy Mania health-gauge regression coverage."""
from __future__ import annotations

from types import SimpleNamespace

import moderngl
import pytest
from PIL import Image

from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas
from osu_mania_renderer_v2.gpu.renderer import (
    FrameRenderer,
    LegacyScorebarLayout,
    classify_legacy_scorebar,
    configure_direct_texture_sampling,
    mania_health_geometry,
    mania_health_new_default,
    standard_health_geometry,
    standard_health_marker_slot,
)


def _alpha_image(
    design_size: tuple[int, int],
    visible_rects: tuple[tuple[int, int, int, int], ...],
    *,
    scale_adjust: int = 1,
) -> Image.Image:
    """Create exact, ScaleAdjust-aware synthetic alpha geometry."""
    width, height = design_size
    image = Image.new(
        "RGBA",
        (width * scale_adjust, height * scale_adjust),
        (255, 255, 255, 0),
    )
    alpha = image.getchannel("A")
    for left, top, right, bottom in visible_rects:
        alpha.paste(
            255,
            (
                left * scale_adjust,
                top * scale_adjust,
                right * scale_adjust,
                bottom * scale_adjust,
            ),
        )
    image.putalpha(alpha)
    image.info["scale_adjust"] = scale_adjust
    return image


def _standard_composite(*, scale_adjust: int = 1):
    return (
        _alpha_image(
            (720, 420),
            ((0, 0, 660, 70), (30, 130, 600, 410)),
            scale_adjust=scale_adjust,
        ),
        _alpha_image(
            (650, 30), ((0, 0, 650, 30),),
            scale_adjust=scale_adjust,
        ),
    )


def test_obvious_standard_composite_requires_visible_art_outside_gauge():
    background, fill = _standard_composite()

    result = classify_legacy_scorebar(
        background, fill, new_default=False,
    )

    assert result.layout is LegacyScorebarLayout.STANDARD_HUD
    assert result.confidence == "high"
    assert result.outside_gauge_alpha_ratio > 0.55
    assert "dominant-outside-gauge-alpha" in result.evidence


def test_large_transparent_canvas_is_not_a_standard_hud_false_positive():
    background = _alpha_image((720, 420), ((0, 0, 660, 70),))
    fill = _alpha_image((650, 30), ((0, 0, 650, 30),))

    result = classify_legacy_scorebar(
        background, fill, new_default=False,
    )

    assert result.layout is not LegacyScorebarLayout.STANDARD_HUD
    assert result.background.alpha_bbox == (0.0, 0.0, 660.0, 70.0)


def test_conventional_compact_scorebar_selects_mania_side():
    background = _alpha_image((712, 100), ((20, 10, 702, 100),))
    fill = _alpha_image((695, 53), ((87, 0, 695, 32),))

    result = classify_legacy_scorebar(
        background, fill, new_default=False,
    )

    assert result.layout is LegacyScorebarLayout.MANIA_SIDE
    assert result.evidence == ("long-shallow-fill", "compact-scorebar-pair")


def test_unusual_inconclusive_scorebar_remains_uncertain():
    background = _alpha_image(
        (720, 220), ((0, 0, 660, 70), (100, 170, 220, 205)),
    )
    fill = _alpha_image((650, 30), ((0, 0, 650, 30),))

    result = classify_legacy_scorebar(
        background, fill, new_default=False,
    )

    assert result.layout is LegacyScorebarLayout.UNCERTAIN
    assert result.evidence == ("composite-signals-incomplete",)


def test_transparent_scorebar_placeholders_have_no_content_metrics():
    transparent = _alpha_image((1366, 768), ())
    fill = _alpha_image((650, 30), ((0, 0, 650, 30),))

    result = classify_legacy_scorebar(
        transparent, fill, new_default=False,
    )

    assert result.layout is LegacyScorebarLayout.UNCERTAIN
    assert result.background.alpha_bbox is None
    assert result.background.alpha_area == 0
    assert result.outside_gauge_alpha_ratio == 0


def test_scorebar_classifier_has_1x_2x_design_space_parity():
    one_x = classify_legacy_scorebar(
        *_standard_composite(scale_adjust=1), new_default=True,
    )
    two_x = classify_legacy_scorebar(
        *_standard_composite(scale_adjust=2), new_default=True,
    )

    assert one_x.layout is two_x.layout is LegacyScorebarLayout.STANDARD_HUD
    assert one_x.background == two_x.background
    assert one_x.fill == two_x.fill
    assert one_x.gauge_corridor == two_x.gauge_corridor
    assert one_x.outside_gauge_alpha_ratio == pytest.approx(
        two_x.outside_gauge_alpha_ratio,
    )
    assert standard_health_geometry(
        1080,
        one_x.background.design_size,
        one_x.fill.design_size,
        (40, 40),
        0.5,
        new_default=True,
    ) == standard_health_geometry(
        1080,
        two_x.background.design_size,
        two_x.fill.design_size,
        (40, 40),
        0.5,
        new_default=True,
    )


def test_mania_health_geometry_uses_stable_stage_relative_offsets():
    new = mania_health_geometry(
        stage_right=850,
        stage_top=25,
        stage_height=480,
        background_native_size=(712, 100),
        fill_native_size=(695, 53),
        hp=0.5,
        new_default=True,
    )
    old = mania_health_geometry(
        stage_right=850,
        stage_top=25,
        stage_height=480,
        background_native_size=(712, 100),
        fill_native_size=(695, 53),
        hp=0.5,
        new_default=False,
    )

    assert new.background_anchor == old.background_anchor == (851.0, 505.0)
    assert new.fill_anchor == pytest.approx((856.6, 499.8))
    assert old.fill_anchor == pytest.approx((858.0, 503.0))
    assert new.rotation_degrees == old.rotation_degrees == -90.0
    assert new.texture_scale == old.texture_scale == pytest.approx(0.4375)
    assert new.background_size == old.background_size == pytest.approx(
        (311.5, 43.75),
    )
    assert new.fill_size == old.fill_size == pytest.approx(
        (304.0625, 23.1875),
    )
    assert new.visible_fill_width == pytest.approx(152.03125)


@pytest.mark.parametrize("stage_height", [480, 720, 1080])
def test_mania_health_geometry_scales_with_rendered_stage_height(stage_height):
    stage_right = stage_height * 1.25
    geometry = mania_health_geometry(
        stage_right=stage_right,
        stage_top=0,
        stage_height=stage_height,
        background_native_size=(712, 100),
        fill_native_size=(695, 53),
        hp=1,
        new_default=False,
    )

    stage_scale = stage_height / 480
    texture_scale = 0.7 * stage_height / 768
    assert geometry.stage_scale == pytest.approx(stage_scale)
    assert geometry.texture_scale == pytest.approx(texture_scale)
    assert geometry.background_anchor == pytest.approx(
        (stage_right + stage_scale, stage_height),
    )
    assert geometry.fill_anchor == pytest.approx(
        (stage_right + 8 * stage_scale, 478 * stage_scale),
    )
    assert geometry.background_size == pytest.approx(
        (712 * texture_scale, 100 * texture_scale),
    )


@pytest.mark.parametrize("render_height", [480, 720, 1080])
@pytest.mark.parametrize(
    ("new_default", "fill_offset", "marker_y"),
    [
        (True, (7.5, 7.8), 10.625),
        (False, (3.0, 10.0), 10.0),
    ],
)
@pytest.mark.parametrize("hp", [0.0, 0.5, 1.0])
def test_standard_health_geometry_ports_stable_top_left_layout(
    render_height, new_default, fill_offset, marker_y, hp,
):
    geometry = standard_health_geometry(
        render_height,
        background_native_size=(1366, 768),
        fill_native_size=(650, 30),
        marker_native_size=(40, 40),
        hp=hp,
        new_default=new_default,
    )
    position_scale = render_height / 480
    texture_scale = render_height / 768

    assert geometry.position_scale == pytest.approx(position_scale)
    assert geometry.background_scale == pytest.approx(0.96 * texture_scale)
    assert geometry.fill_scale == pytest.approx(0.965 * texture_scale)
    assert geometry.marker_scale == pytest.approx(0.97 * texture_scale)
    assert geometry.background_anchor == (0, 0)
    assert geometry.background_size == pytest.approx(
        (1366 * 0.96 * texture_scale, 768 * 0.96 * texture_scale),
    )
    assert geometry.fill_anchor == pytest.approx(
        tuple(value * position_scale for value in fill_offset),
    )
    assert geometry.fill_size == pytest.approx(
        (650 * 0.965 * texture_scale, 30 * 0.965 * texture_scale),
    )
    assert geometry.visible_fill_width == pytest.approx(
        geometry.fill_size[0] * hp,
    )
    assert geometry.marker_center == pytest.approx(
        (
            fill_offset[0] * position_scale + 650 * hp * texture_scale,
            marker_y * position_scale,
        ),
    )
    assert geometry.marker_size == pytest.approx(
        (40 * 0.97 * texture_scale, 40 * 0.97 * texture_scale),
    )


@pytest.mark.parametrize(
    ("hp", "new_default", "expected"),
    [
        (0.1, True, "scorebar_marker"),
        (0.1, False, "scorebar_kidanger2"),
        (0.2, False, "scorebar_kidanger"),
        (0.5, False, "scorebar_ki"),
        (1.0, False, "scorebar_ki"),
    ],
)
def test_standard_marker_selection_matches_stable(hp, new_default, expected):
    assert standard_health_marker_slot(
        hp=hp, new_default=new_default,
    ) == expected


def test_equal_design_size_scorebars_match_at_1x_and_2x(tmp_path):
    one_x = tmp_path / "one-x"
    two_x = tmp_path / "two-x"
    one_x.mkdir()
    two_x.mkdir()
    Image.new("RGBA", (400, 40), (255, 255, 255, 255)).save(
        one_x / "scorebar-bg.png",
    )
    Image.new("RGBA", (800, 80), (255, 255, 255, 255)).save(
        two_x / "scorebar-bg@2x.png",
    )

    one_frames, one_source = SpriteAtlas._resolve_global(
        "scorebar_bg", skin_dir=one_x, beatmap_dir=None, section=None,
    )
    two_frames, two_source = SpriteAtlas._resolve_global(
        "scorebar_bg", skin_dir=two_x, beatmap_dir=None, section=None,
    )
    one = one_frames[0]
    two = two_frames[0]
    one_design = tuple(v / one.info["scale_adjust"] for v in one.size)
    two_design = tuple(v / two.info["scale_adjust"] for v in two.size)

    assert one_source == two_source == "user"
    assert one_design == two_design == (400, 40)
    assert mania_health_geometry(
        850, 0, 720, one_design, (300, 20), 1, False,
    ) == mania_health_geometry(
        850, 0, 720, two_design, (300, 20), 1, False,
    )


@pytest.mark.parametrize(
    ("fill_source", "marker_source", "expected"),
    [
        ("user", "user", True),
        ("user", "beatmap", True),
        ("user", "missing", False),
        ("user", "bundle", False),
        ("bundle", "missing", True),
    ],
)
def test_new_default_selection_only_controls_fill_offset(
    fill_source, marker_source, expected,
):
    assert mania_health_new_default(fill_source, marker_source) is expected


@pytest.mark.parametrize("hp", [0.0, 0.5, 1.0])
def test_direct_fill_is_cropped_in_uv_space_before_rotation(hp):
    renderer = object.__new__(FrameRenderer)
    draws = []
    renderer._draw_direct = lambda *args, **kwargs: draws.append((args, kwargs))

    FrameRenderer._draw_direct_clipped_x(
        renderer,
        "scorebar_colour",
        x=12,
        y=700,
        full_width=100,
        height=20,
        visible_fraction=hp,
        rotation_deg=90,
    )

    if hp == 0:
        assert draws == []
        return
    assert draws == [
        (
            ("scorebar_colour", 12, 700, 100 * hp, 20),
            {
                "tint": (1.0, 1.0, 1.0, 1.0),
                "source_u_end": hp,
                "rotation_deg": 90,
            },
        ),
    ]


class _TextureSamplingProbe:
    def __init__(self):
        self.filter = None
        self.repeat_x = True
        self.repeat_y = True
        self.mipmap_builds = 0

    def build_mipmaps(self):
        self.mipmap_builds += 1


@pytest.mark.parametrize("name", ["scorebar_bg", "scorebar_colour"])
def test_scorebar_direct_textures_use_linear_clamped_sampling(name):
    texture = _TextureSamplingProbe()

    configure_direct_texture_sampling(name, texture)

    assert texture.mipmap_builds == 0
    assert texture.filter == (moderngl.LINEAR, moderngl.LINEAR)
    assert texture.repeat_x is False
    assert texture.repeat_y is False


def test_other_direct_textures_keep_existing_mipmapped_sampling():
    texture = _TextureSamplingProbe()

    configure_direct_texture_sampling("stage_right", texture)

    assert texture.mipmap_builds == 1
    assert texture.filter == (
        moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR,
    )
    assert texture.repeat_x is True
    assert texture.repeat_y is True


@pytest.mark.parametrize("hp", [0.25, 0.5, 1.0])
def test_rotated_fill_grows_bottom_to_top_from_fixed_anchor(hp):
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(height=720)
    draws = []
    renderer._draw_direct_clipped_x = (
        lambda *args, **kwargs: draws.append((args, kwargs))
    )

    FrameRenderer._draw_mania_health_piece(
        renderer,
        "scorebar_colour",
        anchor=(850, 700),
        source_size=(400, 20),
        visible_fraction=hp,
    )

    _, draw = draws[0]
    visible_width = draw["full_width"] * draw["visible_fraction"]
    # A +90-degree GL rotation is stable's -90-degree screen-space rotation.
    # Its resulting top-left-space bounds stay fixed at the anchor's bottom
    # and extend upward by the cropped source width.
    assert draw["rotation_deg"] == 90
    assert draw["tint"] == (1.0, 1.0, 1.0, 1.0)
    assert draw["x"] == pytest.approx(850 + (20 - visible_width) / 2)
    assert draw["y"] == pytest.approx(20 + (visible_width - 20) / 2)
    centre_x = draw["x"] + visible_width / 2
    centre_y = draw["y"] + draw["height"] / 2
    rotated_bounds = (
        centre_x - draw["height"] / 2,
        720 - (centre_y + visible_width / 2),
        centre_x + draw["height"] / 2,
        720 - (centre_y - visible_width / 2),
    )
    assert rotated_bounds == pytest.approx((850, 700 - 400 * hp, 870, 700))


class _Atlas:
    def __init__(self, sources=None):
        self.sources = {
            "scorebar_bg": "user",
            "scorebar_colour": "user",
            # A marker is deliberately available: HpBarMania must still never
            # draw it, though it selects the new-default fill offset in stable.
            "scorebar_marker": "user",
            "scorebar_ki": "user",
            "scorebar_kidanger": "user",
            "scorebar_kidanger2": "user",
            **(sources or {}),
        }

    def global_source(self, name):
        return self.sources[name]

    def global_native_size(self, name):
        if self.sources[name] == "missing":
            return 0, 0
        return {
            "scorebar_bg": (712, 100),
            "scorebar_colour": (695, 53),
            "scorebar_marker": (20, 30),
            "scorebar_ki": (20, 30),
            "scorebar_kidanger": (22, 32),
            "scorebar_kidanger2": (24, 34),
        }[name]

    def global_has_visible_pixels(self, name):
        return self.sources[name] != "missing"


def _renderer(*, sources=None, show_hp_bar=True):
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(
        width=1280,
        height=720,
        ctx=SimpleNamespace(blend_func=None),
    )
    renderer.pf_x = 430
    renderer.pf_w = 420
    renderer.col_w_uniform = 105
    renderer.atlas = _Atlas(sources)
    renderer.options = SimpleNamespace(show_hp_bar=show_hp_bar)
    renderer._is_argon_default = lambda: False
    renderer.clipped_draws = []
    renderer.direct_draws = []
    renderer.sprite_draws = []
    renderer._draw_direct_clipped_x = (
        lambda *args, **kwargs: renderer.clipped_draws.append((args, kwargs))
    )
    renderer._draw_direct = (
        lambda *args, **kwargs: renderer.direct_draws.append((args, kwargs))
    )
    renderer._flush_sprite_batch = lambda: None
    renderer._draw_sprite = lambda *args: renderer.sprite_draws.append(args)
    return renderer


def test_mania_draw_uses_only_neutral_authored_background_and_fill():
    renderer = _renderer()

    FrameRenderer._draw_hp_bar(renderer, SimpleNamespace(hp=0.1))

    assert [args[0] for args, _ in renderer.clipped_draws] == [
        "scorebar_bg",
        "scorebar_colour",
    ]
    assert renderer.sprite_draws == []
    assert all(
        draw["tint"] == (1.0, 1.0, 1.0, 1.0)
        and draw["rotation_deg"] == 90
        for _, draw in renderer.clipped_draws
    )
    background = renderer.clipped_draws[0][1]
    fill = renderer.clipped_draws[1][1]
    assert background["visible_fraction"] == 1
    # Generic new-style LegacyHealthDisplay would tint this red. Mania keeps
    # the neutral multiplication asserted above and preserves the skin colour.
    assert fill["visible_fraction"] == 0.1


def test_standard_composite_draws_horizontal_top_left_assets_and_marker():
    renderer = _renderer()
    renderer._legacy_scorebar_classification = classify_legacy_scorebar(
        *_standard_composite(), new_default=True,
    )

    FrameRenderer._draw_hp_bar(renderer, SimpleNamespace(hp=0.5))

    assert [args[0] for args, _ in renderer.direct_draws] == [
        "scorebar_bg",
        "scorebar_marker",
    ]
    background = renderer.direct_draws[0]
    assert background[0][1] == 0
    assert background[1] == {"tint": (1.0, 1.0, 1.0, 1.0)}
    assert len(renderer.clipped_draws) == 1
    fill_args, fill_draw = renderer.clipped_draws[0]
    assert fill_args == ("scorebar_colour",)
    assert fill_draw["visible_fraction"] == 0.5
    assert fill_draw["rotation_deg"] == 0
    assert fill_draw["tint"] == (1.0, 1.0, 1.0, 1.0)
    assert renderer.sprite_draws == []
    assert renderer.rc.ctx.blend_func == (
        moderngl.SRC_ALPHA,
        moderngl.ONE_MINUS_SRC_ALPHA,
    )


def test_uncertain_classification_keeps_rotated_mania_side_policy():
    renderer = _renderer()
    background = _alpha_image(
        (720, 220), ((0, 0, 660, 70), (100, 170, 220, 205)),
    )
    fill = _alpha_image((650, 30), ((0, 0, 650, 30),))
    renderer._legacy_scorebar_classification = classify_legacy_scorebar(
        background, fill, new_default=False,
    )

    FrameRenderer._draw_hp_bar(renderer, SimpleNamespace(hp=0.5))

    assert renderer.direct_draws == []
    assert [args[0] for args, _ in renderer.clipped_draws] == [
        "scorebar_bg",
        "scorebar_colour",
    ]
    assert all(
        draw["rotation_deg"] == 90 for _, draw in renderer.clipped_draws
    )


def test_zero_hp_draws_background_but_no_fill():
    renderer = _renderer()

    FrameRenderer._draw_hp_bar(renderer, SimpleNamespace(hp=0))

    assert [args[0] for args, _ in renderer.clipped_draws] == ["scorebar_bg"]


@pytest.mark.parametrize(
    "sources",
    [
        {"scorebar_bg": "missing"},
        {"scorebar_colour": "missing"},
        {"scorebar_colour": "beatmap"},
    ],
)
def test_missing_or_mismatched_custom_scorebar_keeps_procedural_fallback(sources):
    renderer = _renderer(sources=sources)

    FrameRenderer._draw_hp_bar(renderer, SimpleNamespace(hp=0.5))

    assert len(renderer.sprite_draws) == 2
    assert renderer.clipped_draws == []


def test_hidden_hp_option_draws_nothing():
    renderer = _renderer(show_hp_bar=False)

    FrameRenderer._draw_hp_bar(renderer, SimpleNamespace(hp=0.5))

    assert renderer.clipped_draws == []
    assert renderer.sprite_draws == []
