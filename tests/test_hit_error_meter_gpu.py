"""Opt-in real pixels for the shared horizontal Mania HEM."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import moderngl
import numpy as np
import pytest

from osu_mania_renderer_v2.beatmap.models import RenderOptions, VisualMods
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.hit_error_meter import (
    BLUE,
    CENTRE_DARK_BLUE,
    GREEN,
    YELLOW,
    hit_error_meter_geometry,
)
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.render.scene import HitErrorEvent, SceneState
from osu_mania_renderer_v2.wiki_elements import hud


def test_canonical_icon_bytes_are_unchanged():
    root = Path(__file__).resolve().parents[1] / "osu_mania_renderer_v2/assets/hud/hit-error-meter"
    expected = {
        "hare": "61713e94571614b5adbee4c149f4703bb3003294c6df356ab1465514607f2979",
        "tortoise": "f1b88e43076527be47c141d42d98576257b276c5cfaf8951ea457bb3c8f1c178",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / f"{name}.png").read_bytes()).hexdigest() == digest


@pytest.mark.slow
@pytest.mark.parametrize("width,height", [(1280, 720), (1920, 1080), (1024, 768)])
def test_horizontal_meter_pixels_depth_shapes_cache_and_path_parity(width, height, monkeypatch):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run real GL pixel checks")
    options = RenderOptions(resolution=(width, height), fps=60)
    with HeadlessGl(width=width, height=height) as gl:
        renderer = FrameRenderer(RenderContext(gl.ctx, gl.fbo, width, height, 4), options)
        scene = SceneState(
            t_ms=2000,
            visible_notes=(),
            keys_held=(False,) * 4,
            visual_mods=VisualMods(),
            hit_error_windows=(20, 40, 60, 80, 100),
            hit_error_events=(
                HitErrorEvent(-35, "300", 100),
                HitErrorEvent(0, "katu", 100),
                HitErrorEvent(80, "50", 100),
            ),
            hit_error_ema_ms=30,
            hit_error_chevron_position=0.65,
        )

        def draw(state, *, wiki=False):
            gl.fbo.use()
            gl.ctx.viewport = (0, 0, width, height)
            gl.ctx.enable(moderngl.BLEND)
            gl.fbo.clear(0, 0, 0, 0)
            if wiki:
                ctx = SimpleNamespace(fr=renderer, scene=state, options=options)
                hud.hit_strip(element=None, skin=None, assets=None, variables=None, ctx=ctx)
            else:
                renderer._draw_hit_error_meter(state)
            renderer._flush_sprite_batch()
            gl.ctx.finish()
            return (
                np.frombuffer(gl.fbo.read(components=4), dtype=np.uint8)
                .reshape(height, width, 4)[::-1]
                .copy()
            )

        base = draw(replace(scene, hit_error_events=()))
        pixels = draw(scene)
        painter = renderer._hit_error_painter
        textures = tuple(painter.icons.values())
        wraps = tuple(texture.extra for texture in textures)
        assert np.array_equal(pixels, draw(scene, wiki=True))
        assert renderer._hit_error_painter is painter
        assert tuple(painter.icons.values()) == textures
        assert tuple(texture.extra for texture in textures) == wraps
        # ModernGL's blend_func getter is unsupported; prove source-over
        # restoration with two translucent sprites after the meter instead.
        for _ in range(2):
            renderer._draw_sprite("column_bg", 10, 10, 10, 10, (1, 0, 0, 0.5))
        renderer._flush_sprite_batch()
        post = np.frombuffer(gl.fbo.read(components=4), dtype=np.uint8).reshape(height, width, 4)
        assert 188 <= post[15, 15, 0] <= 194  # ~191, not additive ~255
        # Skin choice cannot select a different Mania painter.
        renderer._is_argon_default = lambda: False
        assert np.array_equal(pixels, draw(scene))

        g = hit_error_meter_geometry(width, height)
        assert height - (g.component_top + g.natural_height) == pytest.approx(8 * g.scale)
        # Only the component's Y origin changed: byte-identical HEM RGB
        # after translating the previous accepted origin by 30 scaled pixels.
        from osu_mania_renderer_v2.gpu import hit_error_painter

        with monkeypatch.context() as patch:
            patch.setattr(
                hit_error_painter, "hit_error_meter_geometry",
                lambda w, h: replace(hit_error_meter_geometry(w, h), component_top=660 * g.scale),
            )
            previous_origin = draw(scene)
        shifted = np.roll(previous_origin, int(30 * g.scale), axis=0)
        delta = np.abs(pixels.astype(int) - shifted.astype(int))
        assert np.array_equal(pixels[:, :, :3], shifted[:, :, :3])
        # Normalized external-icon quad interpolation can round one alpha
        # channel by one LSB at 1080p; no visible RGB or geometry difference.
        assert delta[:, :, 3].max() <= 1

        def pixel(image, x, y):
            return image[
                int(g.component_top + y * g.scale), int(g.component_x + x * g.scale), :3
            ].astype(float)

        # Solid inner bands, without extra Great/Ok colour tiers or inner fades.
        for x, colour in [(50, YELLOW), (80, GREEN), (110, BLUE)]:
            assert pixel(base, x, 7) == pytest.approx(np.array(colour) * 255, abs=1)
        assert pixel(base, 61, 7) == pytest.approx(np.array(YELLOW) * 255, abs=1)
        assert pixel(base, 63, 7) == pytest.approx(np.array(GREEN) * 255, abs=1)
        # Continuous six-reference-pixel tips, not the old 20%-wide slices.
        for x in (22, 23, 24, 25, 26, 27):
            screen_x = int(g.component_x + x * g.scale)
            actual_local = (screen_x + 0.5 - g.component_x) / g.scale
            expected_alpha = max(0, min(1, (actual_local - 22) / 6))
            assert pixel(base, x, 7) == pytest.approx(
                np.array(YELLOW) * 255 * expected_alpha, abs=2
            )
        assert pixel(base, 28, 7) == pytest.approx(np.array(YELLOW) * 255, abs=1)

        # Foreground inner circle survives a centred additive green tick;
        # its surrounding outer blue circle sits behind that tick.
        assert pixel(pixels, 122, 7) == pytest.approx(np.array(CENTRE_DARK_BLUE) * 255, abs=1)
        assert pixel(pixels, 122, 4)[1] > pixel(base, 122, 4)[1]
        # Known signed tick positions, outside the two-pixel background band.
        for x, colour in [(87, BLUE), (202, YELLOW)]:
            assert pixel(pixels, x, 2) == pytest.approx(np.array(colour) * 255 * 0.6, abs=2)
            assert pixel(pixels, x + 4, 2).max() == 0
        # Capsule ends extend by half the stroke thickness, as Canvas round caps.
        assert pixel(pixels, 87, -1)[2] > 30

        # Both canonical contain-fit icons are present in their 16x14 slots.
        for cx in (8, 236):
            left = int(g.component_x + (cx - 8) * g.scale)
            top = int(g.component_top)
            box = pixels[top : top + int(14 * g.scale), left : left + int(16 * g.scale), :3]
            assert np.count_nonzero(box.max(axis=2)) > 10
            assert box.max() > 100

        # Outlined upward chevron at displayed .65, not filled/stepped or raw EMA.
        assert pixel(pixels, 152, 15).max() > 100
        assert pixel(pixels, 148, 21).max() > 100
        assert pixel(pixels, 152, 19).max() == 0  # hollow interior
        assert pixel(pixels, 142, 19).max() == 0

        # Misses never paint ticks, even on a hand-built diagnostic scene.
        assert np.array_equal(
            pixels,
            draw(
                replace(
                    scene,
                    hit_error_events=scene.hit_error_events + (HitErrorEvent(-90, "miss", 100),),
                )
            ),
        )
