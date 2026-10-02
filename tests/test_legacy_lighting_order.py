"""Layer regressions for additive legacy effects crossing neighbouring keys."""
from __future__ import annotations

import os
from types import SimpleNamespace

import moderngl
import numpy as np
from PIL import Image
import pytest

from osu_mania_renderer_v2.beatmap.models import RenderOptions, VisualMods
from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.render.scene import SceneState
from osu_mania_renderer_v2.wiki_elements import notes, stage
from osu_mania_renderer_v2.wiki_elements.context import FrameContext
from osu_mania_renderer_v2.wiki_renderer import ELEMENTS, RENDER_ORDER


_FIELD_ELEMENTS = {
    "stage_lights", "receptors_under", "notes", "combo_and_judgment",
    "receptors_over", "legacy_hit_lighting", "legacy_combo_and_judgment",
}


def _scene(*, held=(True, True), hit_ages=(80, 80)):
    return SceneState(
        t_ms=1000, visible_notes=(), keys_held=held, visual_mods=VisualMods(),
        key_press_age_ms=(120, 120), hit_light_age_ms=hit_ages,
        hit_light_judgment=("300", "300"),
    )


def _quiet_frame(fr, events):
    """Keep the actual frame orchestrator, keys and hit-light primitives."""
    fr.options = RenderOptions(
        resolution=(fr.rc.width, fr.rc.height), fps=60,
        show_hp_bar=False, show_progress_bar=False, hud_opacity=0,
        show_hit_error_meter=False,
    )
    for name in ("_draw_background", "_draw_stage_decorations", "_draw_columns",
                 "_draw_hud", "_draw_top_chrome", "draw_logo_splash"):
        setattr(fr, name, lambda *args: None)
    fr._draw_notes = lambda scene: events.append("notes")
    fr._draw_stage_lights = lambda scene: events.append("stage_light")
    fr._draw_combo_and_judgment = lambda scene: events.append("judgment")
    fr._draw_legacy_stage_foreground = lambda: events.append("foreground")
    fr._selected_legacy_scorebar_layout = lambda: None
    fr._legacy_timing_overlay_visibility = lambda: (False, False)
    fr._break_overlay = None


def _probe(*, keys_under=None, argon=False):
    events = []
    fr = object.__new__(FrameRenderer)
    fr.rc = SimpleNamespace(
        width=1280, height=720, key_count=2,
        ctx=SimpleNamespace(enable=lambda *args: None, blend_func=None),
        fbo=SimpleNamespace(use=lambda: None, clear=lambda *args: None),
    )
    fr.col_x, fr.col_w = (100, 205), (105, 105)
    fr.col_w_uniform = 105
    fr.receptor_centre_y_gl = 72
    fr.upside_down = False
    fr.mania_section = ManiaSection(
        keys=2, keys_under_notes=keys_under, column_width=(70, 70),
        lighting_n_width=(22, 22), lighting_l_width=(22, 22),
    ) if not argon else None
    fr.skin_ini = SimpleNamespace(legacy_version=2.7)
    fr._is_argon_default = lambda: argon
    fr._draw_argon_receptors = lambda scene: events.append("argon_keys")
    fr.atlas = SimpleNamespace(
        column_native_size=lambda kind, c: (75, 187.5),
        global_source=lambda slot: "user",
        global_native_size=lambda slot: (400, 80),
        index_of=lambda slot: {"lighting_n": 20, "lighting_l": 30}[slot],
        frame_count=lambda slot: 1,
    )
    fr._draw_legacy_column_direct = lambda name, *args, **kw: events.append(
        f"key{int(name.rsplit('/', 1)[1])}",
    )

    def additive(index, x, y, width, height, tint):
        c = round((x + width / 2 - 152.5) / 105)
        # Column zero's light crosses the opaque neighbouring key.
        if c == 0:
            assert x < 205 < x + width
        events.append(f"{'L' if index == 30 else 'N'}{c}")

    fr._draw_additive_sprite_idx = additive
    fr._flush_sprite_batch = lambda: None
    fr.apply_note_cover = lambda *args: None
    _quiet_frame(fr, events)
    return fr, events


def _wiki_field(fr, scene, monkeypatch, events, *, argon=False):
    ctx = FrameContext(
        fr=fr, skin=None, gl=fr.rc.ctx, fbo=fr.rc.fbo,
        width=fr.rc.width, height=fr.rc.height, key_count=2, scene=scene,
        persistent={"_skinmeta": {"provides": not argon}},
    )
    monkeypatch.setattr(notes, "_draw_notes_body", lambda ctx: events.append("notes"))
    monkeypatch.setattr(notes, "combo_and_judgment", lambda **kw: events.append("judgment"))
    monkeypatch.setattr(stage, "stage_foreground", lambda ctx: events.append("foreground"))
    if argon:
        monkeypatch.setattr(notes, "_receptors", lambda ctx: events.append("argon_keys"))
    for name in RENDER_ORDER:
        if name in _FIELD_ELEMENTS:
            ELEMENTS[name].render_fn(
                element=name, skin=None, assets=None, variables=None, ctx=ctx,
            )


@pytest.mark.parametrize("path", ["monolithic", "wiki"])
def test_receptors_never_emit_authored_hit_lights(path):
    fr, events = _probe()
    scene = _scene()
    if path == "monolithic":
        fr._draw_receptors(scene)
    else:
        ctx = FrameContext(
            fr=fr, skin=None, gl=fr.rc.ctx, fbo=fr.rc.fbo,
            width=1280, height=720, key_count=2, scene=scene,
        )
        notes._receptors(ctx)
    assert events == ["key0", "key1"]


@pytest.mark.parametrize("path", ["monolithic", "wiki"])
@pytest.mark.parametrize("keys_under", [None, False, True])
def test_frame_finishes_keys_and_foreground_before_cross_column_hit_lights(
    monkeypatch, path, keys_under,
):
    fr, events = _probe(keys_under=keys_under)
    scene = _scene()
    if path == "monolithic":
        fr.draw(scene)
    else:
        _wiki_field(fr, scene, monkeypatch, events)
    keys_and_notes = ["key0", "key1", "notes"] if keys_under else ["notes", "key0", "key1"]
    assert events == [
        "stage_light", *keys_and_notes, "foreground", "L0", "N0", "L1", "N1", "judgment",
    ]


@pytest.mark.parametrize("path", ["monolithic", "wiki"])
def test_argon_keeps_existing_judgment_before_key_area_order(monkeypatch, path):
    fr, events = _probe(argon=True)
    scene = _scene()
    if path == "monolithic":
        fr.draw(scene)
        assert events == ["stage_light", "notes", "judgment", "argon_keys"]
    else:
        _wiki_field(fr, scene, monkeypatch, events, argon=True)
        assert events == ["stage_light", "notes", "judgment", "argon_keys", "foreground"]
    assert not any(event.startswith(("L", "N")) for event in events)


@pytest.mark.slow
@pytest.mark.parametrize("path", ["monolithic", "wiki"])
@pytest.mark.parametrize("kind", ["n", "l"])
def test_gl_neighbouring_opaque_key_cannot_paint_over_additive_light(
    tmp_path, monkeypatch, path, kind,
):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    (tmp_path / "skin.ini").write_text(
        "[General]\nVersion: 2.7\n[Mania]\nKeys: 2\n"
        "KeyImage0: key0\nKeyImage0D: key0\nKeyImage1: key1\nKeyImage1D: key1\n"
        "LightingNWidth: 22,22\nLightingLWidth: 22,22\n",
    )
    Image.new("RGBA", (64, 768), (0, 40, 90, 255)).save(tmp_path / "key0.png")
    Image.new("RGBA", (64, 768), (0, 40, 90, 255)).save(tmp_path / "key1.png")
    for slot in ("n", "l"):
        colour = (120, 0, 0, 255) if slot == kind else (0, 0, 0, 0)
        Image.new("RGBA", (160, 64), colour).save(tmp_path / f"lighting{slot.upper()}.png")
    with HeadlessGl(width=192, height=768) as gl:
        fr = FrameRenderer(
            RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=192, height=768, key_count=2),
            skin_dir=tmp_path,
        )
        fr.col_x, fr.col_w = (16, 80), (64, 64)
        fr.col_w_uniform = 64
        fr.receptor_centre_y_gl = 40
        events = []
        _quiet_frame(fr, events)
        scene = _scene(
            held=(kind == "l", False),
            hit_ages=(80 if kind == "n" else 9999, 9999),
        )
        if path == "monolithic":
            fr.draw(scene)
        else:
            gl.fbo.use()
            gl.fbo.clear(0, 0, 0, 1)
            gl.ctx.enable(moderngl.BLEND)
            gl.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            _wiki_field(fr, scene, monkeypatch, events)
            fr._flush_sprite_batch()
        image = np.frombuffer(gl.fbo.read(components=3), dtype=np.uint8).reshape(768, 192, 3)
        assert image[40, 90].tolist() == pytest.approx([120, 40, 90], abs=1)

        # Reproduce the old interleaving with the same geometry/blending:
        # key0, light0, key1, light1. Key1 wipes the overlapping red light.
        gl.fbo.clear(0, 0, 0, 1)
        for c in range(2):
            fr._draw_legacy_key(c, held=scene.keys_held[c])
            fr._draw_custom_legacy_lighting(
                scene, c=c, x0=fr.col_x[c], cw=fr.col_w[c], centre_y=40,
                held=scene.keys_held[c],
            )
        fr._flush_sprite_batch()
        broken = np.frombuffer(gl.fbo.read(components=3), dtype=np.uint8).reshape(768, 192, 3)
        assert broken[40, 90].tolist() == pytest.approx([0, 40, 90], abs=1)
