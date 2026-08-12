"""Canonical compositor registry and render smoke coverage."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Canonical painter order — ``render.pipeline`` is the ordering authority.
EXPECTED_ORDER = [
    "background", "stage_decorations", "columns", "stage_lights",
    "receptors_under", "notes", "combo_and_judgment", "receptors_over",
    "hit_error_popups", "hit_strip", "progress_bar",
    "hp_bar", "banner", "hud", "key_counter", "top_chrome", "flashlight",
    "ur_summary",
    "break_overlay",
    "miss_break_wash", "fade_to_black",
    "intro_logo",
    "results_overlay", "watermark",
]


def test_registry_populates_in_canonical_order():
    import osu_mania_renderer_v2.render.pipeline  # noqa: F401 — populate registries
    from osu_mania_renderer_v2.render.compositor import ELEMENTS, RENDER_ORDER

    assert RENDER_ORDER == EXPECTED_ORDER
    assert all(name in ELEMENTS for name in RENDER_ORDER)
    # Every element currently draws through the shared GPU primitives; none
    # declares additional SkinPair-resolved inputs yet.
    for name in RENDER_ORDER:
        spec = ELEMENTS[name]
        assert spec.asset_basenames == ()
        assert callable(spec.render_fn)


def test_monolithic_frame_draw_is_retired():
    from osu_mania_renderer_v2.gpu.renderer import FrameRenderer

    assert not hasattr(FrameRenderer, "draw")


def test_wiki_module_forwards_to_compositor():
    from osu_mania_renderer_v2 import wiki_renderer
    from osu_mania_renderer_v2.render import compositor

    assert wiki_renderer.render is compositor.render
    assert wiki_renderer.RENDER_ORDER is compositor.RENDER_ORDER
    assert wiki_renderer.ELEMENTS is compositor.ELEMENTS


async def test_package_api_forwards_to_canonical_compositor(monkeypatch, tmp_path):
    import osu_mania_renderer_v2
    from osu_mania_renderer_v2.beatmap.models import RenderOptions
    from osu_mania_renderer_v2.render import compositor

    received = None

    async def fake_render_mania(**kwargs):
        nonlocal received
        received = kwargs

    monkeypatch.setattr(compositor, "render_mania", fake_render_mania)
    await osu_mania_renderer_v2.render_mania(
        osr_path=tmp_path / "input.osr",
        beatmap_dir=tmp_path,
        output_path=tmp_path / "output.mp4",
        options=RenderOptions(resolution=(640, 360), fps=30),
    )

    assert received is not None
    assert received["skin_dir"] is None


@pytest.mark.slow
def test_wiki_path_renders_argon(tmp_path, fixtures_dir):
    """The compatibility-named smoke test renders through the compositor."""
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required (GL + ffmpeg)")
    import asyncio

    import osu_mania_renderer_v2.render.pipeline  # noqa: F401 — populate registries
    from osu_mania_renderer_v2.beatmap.models import RenderOptions
    from osu_mania_renderer_v2.render.compositor import render as compositor

    osr = fixtures_dir / "ao_infinity_hard.osr"
    skin = tmp_path / "emptyskin"
    skin.mkdir()
    dflt = (
        Path(__file__).resolve().parent.parent
        / "osu_mania_renderer_v2"
        / "assets"
        / "default_skin"
    )
    out = tmp_path / "wiki.mp4"

    asyncio.run(compositor(
        osr_path=osr, beatmap_dir=fixtures_dir, output_path=out,
        options=RenderOptions(resolution=(854, 480), fps=30),
        skin_dir=skin, default_skin_dir=dflt,
    ))
    assert out.exists() and out.stat().st_size > 100_000
