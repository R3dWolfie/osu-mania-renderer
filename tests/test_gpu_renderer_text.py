import os
from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.beatmap.models import VisualMods
from osu_mania_renderer_v2.gpu import renderer as renderer_module
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.gpu.text import text_to_texture
from osu_mania_renderer_v2.render.scene import SceneState
from osu_mania_renderer_v2.wiki_elements.hud import banner as wiki_banner


def test_monolithic_gameplay_draw_does_not_invoke_banner() -> None:
    assert "_draw_banner" not in FrameRenderer.draw.__code__.co_names


def test_banner_metadata_remains_available_to_results_without_gameplay_draw(
    monkeypatch,
) -> None:
    renderer = object.__new__(FrameRenderer)
    renderer.rc = SimpleNamespace(width=1280, height=720, ctx=object())
    renderer._draw_external_texture = lambda *args, **kwargs: pytest.fail(
        "set_banner_text must not draw the gameplay banner",
    )
    texture = SimpleNamespace(extra=None, release=lambda: None)
    monkeypatch.setattr(
        renderer_module,
        "text_to_texture",
        lambda *_args, **_kwargs: (texture, 320, 24),
    )

    FrameRenderer.set_banner_text(
        renderer, "Nizikawa - F.K.S. [NOVICE Lv.8]   VIO",
    )
    assert renderer._banner_text.endswith("   VIO")

    captured_labels = []

    class _PlayerNameCaptured(Exception):
        pass

    def capture_first_label(text, *_args, **_kwargs):
        captured_labels.append(text)
        raise _PlayerNameCaptured

    renderer._draw_sprite = lambda *args, **kwargs: None
    renderer._draw_direct = lambda *args, **kwargs: None
    renderer._results_avatar_texture = lambda: None
    renderer._cached_text = capture_first_label
    result_ctx = SimpleNamespace(has_argon_font=lambda: False)

    with pytest.raises(_PlayerNameCaptured):
        FrameRenderer._draw_results_overlay(
            renderer,
            SimpleNamespace(results_opacity=1.0),
            ctx=result_ctx,
        )

    assert captured_labels == ["VIO"]


def test_wiki_banner_element_remains_no_op() -> None:
    calls = []
    ctx = SimpleNamespace(
        fr=SimpleNamespace(_draw_banner=lambda: calls.append("banner")),
    )

    wiki_banner(element=None, skin=None, assets=None, variables=None, ctx=ctx)

    assert calls == []


@pytest.mark.slow
def test_text_to_texture_returns_real_dimensions():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    with HeadlessGl(width=64, height=64) as gl:
        tex, w, h = text_to_texture(gl.ctx, "Hello", size=24)
        assert tex is not None
        assert w > 0 and h > 0


@pytest.mark.slow
def test_full_hud_renders():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    W, H = 480, 270
    with HeadlessGl(width=W, height=H) as gl:
        rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=W, height=H, key_count=4)
        fr = FrameRenderer(rc)
        fr.set_banner_text("Seiryu - AO-INFINITY [Hard]   R3D")
        fr._draw_banner = lambda: pytest.fail(
            "normal gameplay draw must not invoke the legacy banner",
        )
        scene = SceneState(
            t_ms=0, visible_notes=(), keys_held=(False,)*4,
            visual_mods=VisualMods(),
            score=865_612, combo=1305, max_combo=1305, accuracy=98.45,
        )
        fr.draw(scene)
        data = gl.fbo.read(components=3)
        assert max(data) > 50
