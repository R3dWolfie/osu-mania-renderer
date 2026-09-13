"""Focused Argon shared-HUD geometry and website-settings regressions."""
from __future__ import annotations

import os
from types import SimpleNamespace

import moderngl
import numpy as np
import pytest

from osu_mania_renderer_v2.beatmap.judgments import windows_for_od
from osu_mania_renderer_v2.beatmap.models import RenderOptions, VisualMods
from osu_mania_renderer_v2.cli import _build_parser, _render_options_from_args
from osu_mania_renderer_v2.gpu.argon_health import (
    argon_health_path_geometry,
    argon_health_path_point,
    argon_health_path_renderer,
)
from osu_mania_renderer_v2.gpu.argon_wedge import (
    ARGON_WEDGE_ACCENT,
    ARGON_WEDGE_BOTTOM_ALPHA,
    ARGON_WEDGE_CORNER_RADIUS,
    ARGON_WEDGE_POSITIONS,
    ARGON_WEDGE_SHEAR_X,
    ARGON_WEDGE_SIZE,
    ARGON_WEDGE_TOP_ALPHA,
    argon_wedge_geometry,
    argon_wedge_renderer,
    argon_wedge_transform_point,
)
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.render.scene import HitErrorEvent, SceneState
from osu_mania_renderer_v2.wiki_elements import hud
from osu_mania_renderer_v2.wiki_elements.context import FrameContext
from osu_mania_renderer_v2.wiki_elements.hud import (
    ARGON_HEALTH_GLOW_PORTION,
    ARGON_PP_SCALE,
    argon_hit_error_axis_y,
    argon_hit_error_meter_geometry,
    argon_hud_geometry,
    hud_component_visibility,
)
from osu_mania_renderer_v2.wiki_renderer import (
    _build_wiki_parser,
    _wiki_visibility_options,
)


@pytest.mark.parametrize(
    ("width", "height", "scale"),
    [(1280, 720, 1.0), (1920, 1080, 1.5)],
)
def test_argon_hud_uses_720_reference_geometry(width, height, scale):
    geometry = argon_hud_geometry(width, height)

    assert geometry.scale == pytest.approx(scale)
    assert geometry.health_rect == pytest.approx(
        (50 * scale, 20 * scale, 300 * scale, 50 * scale),
    )
    assert geometry.health_bar_height == pytest.approx(30 * scale)
    assert geometry.health_main_radius == pytest.approx(10 * scale)
    assert geometry.health_glow_radius == pytest.approx(40 * scale)
    assert geometry.health_glow_rect == pytest.approx(
        (20 * scale, -10 * scale, 360 * scale, 110 * scale),
    )
    assert geometry.connector_rect == pytest.approx(
        (0, 30 * scale, 45 * scale, 3 * scale),
    )
    assert geometry.score_anchor == pytest.approx((250 * scale, 50 * scale))
    assert geometry.accuracy_anchor == pytest.approx(
        (width - 20 * scale, 20 * scale),
    )
    assert geometry.pp_anchor == pytest.approx(
        (width - 20 * scale, 72 * scale),
    )
    assert ARGON_PP_SCALE == pytest.approx(0.8)
    assert ARGON_HEALTH_GLOW_PORTION == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("height", "scale"),
    [(720, 1.0), (1080, 1.5)],
)
def test_argon_wedge_geometry_scales_from_source_height(height, scale):
    geometry = argon_wedge_geometry(height)

    assert geometry.scale == pytest.approx(scale)
    assert geometry.source_size == ARGON_WEDGE_SIZE
    assert geometry.output_size == pytest.approx(
        tuple(value * scale for value in ARGON_WEDGE_SIZE),
    )
    assert geometry.source_positions == ARGON_WEDGE_POSITIONS
    for output, (source_x, source_y) in zip(
        geometry.output_positions, ARGON_WEDGE_POSITIONS, strict=True,
    ):
        assert output == pytest.approx(
            (source_x * scale, source_y * scale),
        )
    assert geometry.corner_radius == ARGON_WEDGE_CORNER_RADIUS
    assert geometry.output_corner_radius == pytest.approx(
        ARGON_WEDGE_CORNER_RADIUS * scale,
    )
    assert geometry.shear_x == ARGON_WEDGE_SHEAR_X
    assert geometry.accent == ARGON_WEDGE_ACCENT
    assert geometry.top_alpha == ARGON_WEDGE_TOP_ALPHA
    assert geometry.bottom_alpha == ARGON_WEDGE_BOTTOM_ALPHA


@pytest.mark.parametrize(
    ("scale", "expected_top", "expected_bottom"),
    [
        (1.0, (-50.0, 15.0), (-107.6, 87.0)),
        (1.5, (-75.0, 22.5), (-161.4, 130.5)),
    ],
)
def test_argon_wedge_transform_uses_source_shear_without_scaling_factor(
    scale, expected_top, expected_bottom,
):
    position = ARGON_WEDGE_POSITIONS[0]

    assert argon_wedge_transform_point(
        position, (0.0, 0.0), scale,
    ) == pytest.approx(expected_top)
    assert argon_wedge_transform_point(
        position, (0.0, ARGON_WEDGE_SIZE[1]), scale,
    ) == pytest.approx(expected_bottom)


def test_argon_health_path_follows_top_diagonal_bottom_source_shape():
    points = argon_hud_geometry(1280, 720).health_path_points

    assert points[0] == pytest.approx((60, 30))
    assert points[1] == pytest.approx((260, 30))
    assert points[-2] == pytest.approx((310, 60))
    assert points[-1] == pytest.approx((340, 60))
    assert points[2][0] < points[3][0]
    assert points[2][1] < points[3][1]


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        (0.0, (10.0, 10.0)),
        (0.25, (82.587196, 10.0)),
        (0.5, (155.174392, 10.0)),
        (0.75, (226.202101, 16.244241)),
        (1.0, (290.0, 40.0)),
    ],
)
def test_argon_health_progress_uses_source_path_length(progress, expected):
    assert argon_health_path_point(progress) == pytest.approx(expected)


def test_argon_health_progress_is_continuous_through_rounded_transitions():
    path = argon_health_path_geometry()
    boundaries = (
        path.top_line_end,
        path.top_arc_end_progress,
        path.slash_end_progress,
        path.bottom_arc_end_progress,
    )

    for boundary in boundaries:
        before = argon_health_path_point(boundary - 1e-5, path)
        at = argon_health_path_point(boundary, path)
        after = argon_health_path_point(boundary + 1e-5, path)
        assert np.linalg.norm(np.subtract(before, at)) < 0.01
        assert np.linalg.norm(np.subtract(after, at)) < 0.01


def test_argon_hit_error_pair_uses_source_vertical_geometry():
    geometry = argon_hit_error_meter_geometry(1280, 720)
    left, right = geometry.instances

    assert geometry.width == 22
    assert geometry.height == 244
    assert geometry.axis_start == 22
    assert geometry.axis_centre == 122
    assert geometry.cross_centre == 15
    assert geometry.bar_length == 200
    assert geometry.column_size == 14
    assert geometry.band_size == 2
    assert geometry.tick_thickness == 4
    assert geometry.centre_marker_size == 8
    assert geometry.chevron_size == 8
    assert geometry.chevron_stroke == 2
    assert geometry.edge_fade_size == 6
    assert (left.left, left.top, left.mirrored) == (0, 238, False)
    assert (right.left, right.top, right.mirrored) == (1258, 238, True)
    assert argon_hit_error_axis_y(geometry, -127, 127) == 22
    assert argon_hit_error_axis_y(geometry, 0, 127) == 122
    assert argon_hit_error_axis_y(geometry, 127, 127) == 222


def test_argon_hit_error_pair_scales_from_720_at_1080p():
    geometry = argon_hit_error_meter_geometry(1920, 1080)

    assert geometry.scale == 1.5
    assert geometry.width == 33
    assert geometry.height == 366
    assert geometry.instances[0].top == 357
    assert geometry.instances[1].left == 1887


@pytest.mark.parametrize(
    ("option_name", "component_name"),
    [
        ("show_hp_bar", "health"),
        ("show_hit_error_meter", "hit_error_meter"),
        ("show_unstable_rate", "unstable_rate"),
        ("show_score", "score"),
        ("show_mods", "mods"),
        ("show_scoreboard", "scoreboard"),
        ("show_pp_counter", "performance_points"),
    ],
)
def test_website_hud_settings_gate_only_their_component(
    option_name, component_name,
):
    kwargs = {
        "resolution": (1280, 720),
        "fps": 60,
        "show_pp_counter": True,
    }
    options = RenderOptions(**kwargs)
    baseline = hud_component_visibility(options)

    kwargs[option_name] = False
    changed = hud_component_visibility(RenderOptions(**kwargs))

    for field in baseline.__dataclass_fields__:
        if field == component_name:
            assert getattr(changed, field) is False
        else:
            assert getattr(changed, field) == getattr(baseline, field)


def test_hud_opacity_zero_suppresses_all_touched_hud_components():
    visibility = hud_component_visibility(SimpleNamespace(hud_opacity=0.0))

    assert not any(
        getattr(visibility, field) for field in visibility.__dataclass_fields__
    )


def test_score_gate_does_not_hide_argon_health(monkeypatch):
    calls = []
    numbers = []
    monkeypatch.setattr(hud, "_draw_argon_wedges", lambda ctx: None)
    monkeypatch.setattr(
        hud, "_draw_argon_health_display",
        lambda ctx, geometry, hp: calls.append(("health", hp)),
    )
    monkeypatch.setattr(
        hud, "_argon_number",
        lambda ctx, text, **kwargs: numbers.append(text),
    )
    fr = SimpleNamespace(
        rc=SimpleNamespace(width=1280, height=720),
        _cached_text=lambda text, size, colour: (text, 20, 10),
        _draw_external_texture=lambda *args, **kwargs: None,
    )
    ctx = SimpleNamespace(
        fr=fr,
        height=720,
        options=RenderOptions(
            resolution=(1280, 720), fps=60,
            show_score=False, show_scoreboard=False,
        ),
        scene=SimpleNamespace(
            results_opacity=0,
            score=123,
            score_smoothed=123,
            accuracy=98.5,
            accuracy_smoothed=98.5,
            hp=0.75,
            max_pp=0,
            pp=0,
            replay_mods=0,
        ),
        atlas=SimpleNamespace(global_source=lambda slot: "missing"),
        draw_number=lambda *args, **kwargs: None,
    )

    hud._draw_argon_hud(ctx)

    assert calls == [("health", 0.75)]
    assert "123" not in numbers
    assert numbers == ["98.50%"]


def test_health_gate_keeps_decorative_argon_wedges(monkeypatch):
    wedges = []
    monkeypatch.setattr(
        hud, "_draw_argon_wedges", lambda ctx: wedges.append(ctx),
    )
    monkeypatch.setattr(hud, "_argon_number", lambda *args, **kwargs: None)
    fr = SimpleNamespace(
        rc=SimpleNamespace(width=1280, height=720),
        _cached_text=lambda text, size, colour: (text, 20, 10),
        _draw_external_texture=lambda *args, **kwargs: None,
    )
    ctx = SimpleNamespace(
        fr=fr,
        height=720,
        options=RenderOptions(
            resolution=(1280, 720), fps=60,
            show_hp_bar=False, show_score=False, show_scoreboard=False,
        ),
        scene=SimpleNamespace(
            results_opacity=0,
            score=0,
            score_smoothed=0,
            accuracy=100,
            accuracy_smoothed=100,
            max_pp=0,
            replay_mods=0,
        ),
        draw_number=lambda *args, **kwargs: None,
    )

    hud._draw_argon_hud(ctx)

    assert wedges == [ctx]


@pytest.mark.parametrize("show_scoreboard", [True, False])
def test_argon_gameplay_leaderboard_is_omitted_without_affecting_hud(
    monkeypatch, show_scoreboard,
):
    assert not hasattr(hud, "_draw_leaderboard")
    calls = []
    monkeypatch.setattr(
        hud, "_draw_argon_wedges", lambda ctx: calls.append("wedges"),
    )
    monkeypatch.setattr(
        hud,
        "_draw_argon_health_display",
        lambda ctx, geometry, hp: calls.append(("health", hp)),
    )
    monkeypatch.setattr(
        hud,
        "_argon_number",
        lambda ctx, text, **kwargs: calls.append(("number", text)),
    )
    monkeypatch.setattr(
        hud, "_draw_mod_icons", lambda *args, **kwargs: calls.append("mods"),
    )
    fr = SimpleNamespace(
        rc=SimpleNamespace(width=1280, height=720),
        _cached_text=lambda text, size, colour: (text, 20, 10),
        _draw_external_texture=lambda *args, **kwargs: None,
    )
    ctx = SimpleNamespace(
        fr=fr,
        height=720,
        options=RenderOptions(
            resolution=(1280, 720),
            fps=60,
            show_scoreboard=show_scoreboard,
        ),
        scene=SimpleNamespace(
            results_opacity=0,
            score=123,
            score_smoothed=123,
            accuracy=98.5,
            accuracy_smoothed=98.5,
            hp=0.75,
            max_pp=0,
            pp=0,
            replay_mods=0,
        ),
        draw_number=lambda *args, **kwargs: None,
        draw_direct=lambda name, *args: pytest.fail(
            f"Argon gameplay HUD must not draw leaderboard sprite {name!r}"
        ),
    )

    hud._draw_argon_hud(ctx)

    assert calls == [
        "wedges",
        ("number", "123"),
        ("health", 0.75),
        ("number", "98.50%"),
        "mods",
    ]


def test_show_mods_false_hides_actual_replay_mod_row():
    draws = []
    ctx = SimpleNamespace(
        height=720,
        options=RenderOptions(
            resolution=(1280, 720), fps=60, show_mods=False,
        ),
        scene=SimpleNamespace(replay_mods=1 << 15),
        draw_sprite=lambda *args: draws.append(args),
        text=lambda *args: pytest.fail("hidden mods must not rasterise text"),
        draw_external=lambda *args: draws.append(args),
    )

    hud._draw_mod_icons(ctx, 1260, 110)

    assert draws == []


def test_hit_error_setting_gates_both_argon_meters(monkeypatch):
    calls = []
    monkeypatch.setattr(hud, "is_argon_default", lambda ctx, col: True)
    monkeypatch.setattr(hud, "_argon_hit_error", lambda ctx: calls.append(ctx))
    ctx = SimpleNamespace(
        options=RenderOptions(resolution=(1280, 720), fps=60),
        fr=SimpleNamespace(),
        scene=SimpleNamespace(),
    )

    hud.hit_strip(element=None, skin=None, assets=None, variables=None, ctx=ctx)
    assert calls == [ctx]

    ctx.options = RenderOptions(
        resolution=(1280, 720), fps=60, show_hit_error_meter=False,
    )
    hud.hit_strip(element=None, skin=None, assets=None, variables=None, ctx=ctx)
    assert calls == [ctx]


def test_argon_meter_consumes_scene_mania_windows(monkeypatch):
    from osu_mania_renderer_v2.gpu import renderer as renderer_module

    supplied = (18.0, 40.0, 70.0, 100.0, 125.0)
    seen = []
    real_bands = renderer_module.lazer_hit_window_bands
    monkeypatch.setattr(
        renderer_module,
        "lazer_hit_window_bands",
        lambda windows: (seen.append(windows) or real_bands(windows)),
    )
    draws = []
    gl = SimpleNamespace(blend_func=None)
    fr = SimpleNamespace(
        rc=SimpleNamespace(width=1280, height=720, ctx=gl),
        _flush_sprite_batch=lambda: None,
        _draw_sprite=lambda *args: draws.append(args),
        _draw_direct=lambda *args, **kwargs: draws.append(args),
        _cached_text=lambda text, size, colour: (text, 8, 8),
        _draw_external_texture=lambda *args, **kwargs: None,
    )
    ctx = SimpleNamespace(
        fr=fr,
        height=720,
        scene=SimpleNamespace(
            hit_error_windows=supplied,
            hit_error_events=(),
            hit_error_ema_ms=None,
        ),
        draw_sprite=lambda *args: draws.append(args),
    )

    hud._argon_hit_error(ctx)

    assert seen == [supplied]
    # Static colour bars and centre markers are emitted for both edge meters.
    assert any(call[1] < 25 for call in draws if len(call) >= 2)
    assert any(call[1] > 1250 for call in draws if len(call) >= 2)


def test_standalone_ur_text_has_no_average_offset():
    text_calls = []
    draw_calls = []
    fr = SimpleNamespace(
        _cached_text=lambda text, size, colour: (
            text_calls.append((text, size, colour)) or (text, 100, 20)
        ),
        _draw_external_texture=lambda *args, **kwargs: draw_calls.append(kwargs),
    )
    ctx = SimpleNamespace(
        fr=fr,
        height=720,
        scene=SimpleNamespace(unstable_rate=137.6, avg_hit_offset_ms=8.0),
    )

    hud._argon_unstable_rate(ctx)

    assert text_calls[0][0] == "UR: 137.60"
    assert "Avg" not in text_calls[0][0]
    assert draw_calls


def test_argon_ur_wrapper_obeys_website_and_compatibility_gates(monkeypatch):
    calls = []
    monkeypatch.setattr(hud, "is_argon_default", lambda ctx, col: True)
    monkeypatch.setattr(hud, "_argon_unstable_rate", lambda ctx: calls.append(ctx))
    ctx = SimpleNamespace(
        options=RenderOptions(resolution=(1280, 720), fps=60),
    )

    hud.ur_summary(element=None, skin=None, assets=None, variables=None, ctx=ctx)
    assert calls == [ctx]

    ctx.options = RenderOptions(
        resolution=(1280, 720), fps=60, show_unstable_rate=False,
    )
    hud.ur_summary(element=None, skin=None, assets=None, variables=None, ctx=ctx)
    assert calls == [ctx]


def test_existing_cli_flags_map_to_explicit_hud_options():
    args = _build_parser().parse_args([
        "replay.osr", "beatmap", "--no-hp-bar", "--no-hit-error",
        "--no-ur", "--no-score", "--no-mods", "--no-scoreboard",
        "--show-pp",
    ])
    options = _render_options_from_args(args)

    assert options.show_hp_bar is False
    assert options.show_hit_error_meter is False
    assert options.show_hit_error_popup is False
    assert options.show_unstable_rate is False
    assert options.show_ur_bar is False
    assert options.show_score is False
    assert options.show_mods is False
    assert options.show_scoreboard is False
    assert options.show_pp_counter is True


def test_wiki_cli_accepts_dispatcher_flags_and_maps_them():
    parser = _build_wiki_parser()
    args, unknown = parser.parse_known_args([
        "replay.osr", "beatmap", "-o", "out.mp4", "--no-hp-bar",
        "--no-hit-error", "--no-ur", "--no-score", "--no-mods",
        "--no-scoreboard", "--show-pp",
    ])

    assert unknown == []
    mapped = _wiki_visibility_options(args)
    assert mapped["show_hp_bar"] is False
    assert mapped["show_hit_error_meter"] is False
    assert mapped["show_unstable_rate"] is False
    assert mapped["show_score"] is False
    assert mapped["show_mods"] is False
    assert mapped["show_scoreboard"] is False
    assert mapped["show_pp_counter"] is True


@pytest.mark.slow
def test_argon_wedge_shader_gradient_shear_cache_and_no_sprite_fallback():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")

    width, height = 1280, 720
    options = RenderOptions(resolution=(width, height), fps=60)
    with HeadlessGl(width=width, height=height) as gl:
        renderer = FrameRenderer(
            RenderContext(
                ctx=gl.ctx,
                fbo=gl.fbo,
                width=width,
                height=height,
                key_count=4,
            ),
            options,
        )
        ctx = FrameContext(
            fr=renderer,
            skin=None,
            gl=gl.ctx,
            fbo=gl.fbo,
            width=width,
            height=height,
            key_count=4,
        )
        renderer._draw_direct = lambda name, *args, **kwargs: pytest.fail(
            f"procedural wedge must not draw sprite {name!r}"
        )

        gl.fbo.use()
        gl.fbo.clear(0.0, 0.0, 0.0, 0.0)
        gl.ctx.enable(moderngl.BLEND)
        gl.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )
        hud._draw_argon_wedges(ctx)
        gl.ctx.finish()

        cached = argon_wedge_renderer(renderer)
        assert argon_wedge_renderer(renderer) is cached
        assert "argon_wedge" not in renderer._direct_arr_cache

        pixels = np.frombuffer(
            gl.fbo.read(components=4), dtype=np.uint8,
        ).reshape(height, width, 4)
        pixels = np.flipud(pixels)
        alpha = pixels[:, :, 3]

        # Source alpha rises from zero at the top to 0.25 at the bottom.
        assert int(alpha[15:20, :340].max()) <= 6
        assert int(alpha[76:86, :300].max()) >= 45
        assert int(alpha.max()) <= 112  # overlap only; never an opaque plate.

        # Equal source widths shift left down the shape by x' = x - 0.8y.
        top_x = np.flatnonzero(alpha[25] > 3)
        bottom_x = np.flatnonzero(alpha[80] > 3)
        assert top_x.size > 0
        assert bottom_x.size > 0
        assert int(top_x.max()) - int(bottom_x.max()) >= 40
        # The transformed lower-right corner is clipped by the round-rect SDF.
        assert int(alpha[82, 274]) >= 35
        assert int(alpha[90, 276]) <= 3

        # All material pixels retain the cyan accent ratio. In particular,
        # there is no opaque/dark navy sprite rectangle behind the gradient.
        material = pixels[alpha >= 8, :3].astype(np.int16)
        assert material.size > 0
        assert np.all(material[:, 1] + 2 >= material[:, 0] * 1.8)
        assert np.all(material[:, 2] + 2 >= material[:, 1] * 1.2)


@pytest.mark.slow
def test_argon_health_shader_bounds_progress_cache_and_blend_restore():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")

    width, height = 1280, 720
    options = RenderOptions(resolution=(width, height), fps=60)
    with HeadlessGl(width=width, height=height) as gl:
        renderer = FrameRenderer(
            RenderContext(
                ctx=gl.ctx,
                fbo=gl.fbo,
                width=width,
                height=height,
                key_count=4,
            ),
            options,
        )
        ctx = FrameContext(
            fr=renderer,
            skin=None,
            gl=gl.ctx,
            fbo=gl.fbo,
            width=width,
            height=height,
            key_count=4,
        )
        geometry = argon_hud_geometry(width, height)

        def render_health(progress):
            gl.fbo.use()
            gl.fbo.clear(0.0, 0.0, 0.0, 0.0)
            gl.ctx.enable(moderngl.BLEND)
            gl.ctx.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )
            hud._draw_argon_health_paths(ctx, geometry, progress)
            gl.ctx.finish()
            pixels = np.frombuffer(
                gl.fbo.read(components=4), dtype=np.uint8,
            ).reshape(height, width, 4)
            return np.flipud(pixels)

        full = render_health(1.0)
        cached = argon_health_path_renderer(renderer)
        assert argon_health_path_renderer(renderer) is cached

        visible = np.max(full[:, :, :3], axis=2) > 2
        ys, xs = np.nonzero(visible)
        glow_left, glow_top, glow_width, glow_height = geometry.health_glow_rect
        assert xs.min() >= int(glow_left)
        assert xs.max() < int(np.ceil(glow_left + glow_width))
        assert ys.min() >= max(0, int(glow_top))
        assert ys.max() < int(np.ceil(glow_top + glow_height))

        source_path = argon_health_path_geometry()

        def path_brightness(pixels, progress):
            local_x, local_y = argon_health_path_point(progress, source_path)
            x = int(round(geometry.health_rect[0] + local_x))
            y = int(round(geometry.health_rect[1] + local_y))
            sample = pixels[y - 2:y + 3, x - 2:x + 3, :3]
            return int(sample.max())

        assert min(
            path_brightness(full, progress)
            for progress in np.linspace(0.0, 1.0, 41)
        ) > 180

        half = render_health(0.5)
        assert path_brightness(half, 0.5) > 180
        assert path_brightness(half, 0.75) < 80

        empty = render_health(0.0)
        assert path_brightness(empty, 0.5) < 80

        # The shader's additive passes must not leak into ordinary HUD draws.
        renderer._draw_sprite(
            "column_bg", 500, 200, 20, 20, (1.0, 0.0, 0.0, 0.5),
        )
        renderer._flush_sprite_batch()
        gl.ctx.finish()
        post = np.frombuffer(
            gl.fbo.read(components=4), dtype=np.uint8,
        ).reshape(height, width, 4)
        red = post[210, 510, :3]
        assert 80 < red[0] < 200
        assert red[1] < 20
        assert red[2] < 20


@pytest.mark.slow
def test_argon_health_meter_and_ur_gpu_smoke():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")

    options = RenderOptions(resolution=(1280, 720), fps=60)
    with HeadlessGl(width=1280, height=720) as gl:
        renderer = FrameRenderer(
            RenderContext(
                ctx=gl.ctx,
                fbo=gl.fbo,
                width=1280,
                height=720,
                key_count=4,
            ),
            options,
        )
        scene = SceneState(
            t_ms=1000,
            visible_notes=(),
            keys_held=(False, False, False, False),
            visual_mods=VisualMods(),
            hp=0.65,
            unstable_rate=137.6,
            hit_error_windows=windows_for_od(8.0),
            hit_error_events=(HitErrorEvent(-20.0, "300", 100),),
            hit_error_ema_ms=-2.0,
        )
        ctx = FrameContext(
            fr=renderer,
            skin=None,
            gl=gl.ctx,
            fbo=gl.fbo,
            width=1280,
            height=720,
            key_count=4,
            scene=scene,
        )

        ctx.begin_frame()
        hud._draw_argon_hud(ctx)
        hud._argon_hit_error(ctx)
        hud._argon_unstable_rate(ctx)
        ctx.flush()

        assert max(gl.fbo.read(components=3)) > 50
