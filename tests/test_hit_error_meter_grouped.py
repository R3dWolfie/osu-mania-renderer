"""The hit error meter drawn as grouped instanced runs (R3D_MANIA_GROUPED) must
be the stock meter to the byte.

What must hold:
  * the grouped fragment shader is the stock one with its uniforms fed per
    instance, and nothing else changed in it;
  * the switch: on by default on macOS, off elsewhere, forced either way;
  * real pixels (RUN_SLOW=1): the same scenes drawn by the stock painter and by
    the grouped one give byte-identical framebuffers, at three sizes, with
    many ticks, with none, with and without the chevron, and when one run
    holds more shapes than the buffer;
  * and the grouped painter really makes fewer draws.
"""
from __future__ import annotations

import os
from dataclasses import replace

import moderngl
import numpy as np
import pytest

from osu_mania_renderer_v2.beatmap.models import RenderOptions, VisualMods
from osu_mania_renderer_v2.gpu import hit_error_painter as H
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.render.scene import HitErrorEvent, SceneState


def test_the_grouped_fragment_is_the_stock_one_fed_per_instance():
    stock, grouped = H._FRAGMENT, H._grouped_fragment()
    assert "uniform" not in grouped
    for gl_type, name in H._GROUPED_NAMES:
        assert f"uniform {gl_type} {name};\n" in stock
        assert f"flat in {gl_type} v_{name};\n{gl_type} {name};\n" in grouped
        assert f"    {name} = v_{name};\n" in grouped
    # take the added lines out again and the stock source is back, exactly
    back = grouped
    for gl_type, name in H._GROUPED_NAMES:
        back = back.replace(f"flat in {gl_type} v_{name};\n{gl_type} {name};\n",
                            f"uniform {gl_type} {name};\n")
        back = back.replace(f"    {name} = v_{name};\n", "")
    assert back == stock
    # the position is computed by the stock expression
    for line in ("point = rect.xy + in_pos * rect.zw;",):
        assert line in H._VERTEX
        assert line.replace("rect", "i_rect") in H._VERTEX_GROUPED
    assert "gl_Position = vec4(point.x / screen.x * 2.0 - 1.0," in H._VERTEX_GROUPED


def test_the_switch(monkeypatch):
    monkeypatch.delenv("R3D_MANIA_GROUPED", raising=False)
    monkeypatch.setattr(H.sys, "platform", "darwin")
    assert H.grouped_draws() is True
    for platform in ("linux", "win32"):
        monkeypatch.setattr(H.sys, "platform", platform)
        assert H.grouped_draws() is False
    for value, want in (("1", True), ("0", False), ("off", False)):
        monkeypatch.setenv("R3D_MANIA_GROUPED", value)
        for platform in ("darwin", "linux"):
            monkeypatch.setattr(H.sys, "platform", platform)
            assert H.grouped_draws() is want


def _scenes():
    base = SceneState(
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
    many = tuple(HitErrorEvent((i * 37) % 180 - 90, ("300", "katu", "100", "50")[i % 4],
                               20 + (i * 53) % 700) for i in range(60))
    return {
        "three ticks": base,
        "no ticks": replace(base, hit_error_events=()),
        "no chevron": replace(base, hit_error_chevron_position=None),
        "sixty ticks": replace(base, hit_error_events=many),
        "one window": replace(base, hit_error_windows=(60,)),
    }


@pytest.mark.slow
@pytest.mark.parametrize("width,height", [(1280, 720), (1920, 1080), (1024, 768)])
def test_grouped_pixels_are_the_stock_pixels(width, height, monkeypatch):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run real GL pixel checks")
    options = RenderOptions(resolution=(width, height), fps=60)
    frames, draws = {}, {}
    for mode in ("0", "1"):
        monkeypatch.setenv("R3D_MANIA_GROUPED", mode)
        monkeypatch.setattr(H, "_SHAPE_CAP", 96 if mode == "0" else 7)   # force mid-run flushes
        with HeadlessGl(width=width, height=height) as gl:
            renderer = FrameRenderer(RenderContext(gl.ctx, gl.fbo, width, height, 4), options)
            for name, scene in _scenes().items():
                gl.fbo.use()
                gl.ctx.viewport = (0, 0, width, height)
                gl.ctx.enable(moderngl.BLEND)
                gl.fbo.clear(0.1, 0.2, 0.3, 1.0)
                renderer._draw_hit_error_meter(scene)
                renderer._flush_sprite_batch()
                gl.ctx.finish()
                frames[mode, name] = bytes(gl.fbo.read(components=4))
            painter = renderer._hit_error_painter
            assert painter._grouped is (mode == "1")
            assert painter._queued == 0                       # nothing left undrawn
    for name in _scenes():
        a, b = frames["0", name], frames["1", name]
        assert a == b, f"{name} at {width}x{height}: grouped differs from stock"
    # the scenes really differ from each other, so the comparison saw something
    assert len({frames["0", n] for n in _scenes()}) == len(_scenes())


@pytest.mark.slow
def test_grouped_makes_fewer_draws(monkeypatch):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run real GL pixel checks")
    width, height = 1280, 720
    options = RenderOptions(resolution=(width, height), fps=60)
    counts = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("R3D_MANIA_GROUPED", mode)
        with HeadlessGl(width=width, height=height) as gl:
            renderer = FrameRenderer(RenderContext(gl.ctx, gl.fbo, width, height, 4), options)
            gl.fbo.use()
            gl.ctx.enable(moderngl.BLEND)
            renderer._draw_hit_error_meter(_scenes()["three ticks"])       # builds the painter
            painter = renderer._hit_error_painter
            n = {"draws": 0}

            class _Counting:
                def __init__(self, vao):
                    self.vao = vao

                def render(self, *a, **k):
                    n["draws"] += 1
                    return self.vao.render(*a, **k)
            if mode == "1":
                painter._grouped_vao = _Counting(painter._grouped_vao)
            else:
                painter.vao = _Counting(painter.vao)
            renderer._draw_hit_error_meter(_scenes()["three ticks"])
            counts[mode] = n["draws"]
    assert counts["0"] == 9           # one per shape: bands, centre, 3 ticks, dark centre, chevron
    assert counts["1"] == 4           # bands + centre, ticks, dark centre, chevron
