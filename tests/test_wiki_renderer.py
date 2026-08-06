"""Wiki-driven renderer: registry population (fast) + byte-for-byte parity
with the legacy renderer (slow, RUN_SLOW=1).

Phase 1 elements delegate to FrameRenderer in draw()'s exact order, so the
wiki path must produce a bit-identical MP4 to render_mania for the same
inputs. This test is the guard that keeps that true as elements migrate.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

# Canonical painter order — must match gpu/renderer.py::FrameRenderer.draw().
EXPECTED_ORDER = [
    "background", "stage_decorations", "columns", "stage_lights",
    "receptors_under", "notes", "combo_and_judgment", "receptors_over",
    "hit_error_popups", "hit_strip", "progress_bar", "fail_overlay",
    "hp_bar", "banner", "hud", "key_counter", "top_chrome", "flashlight",
    "ur_summary",
    "miss_break_wash", "fade_to_black", "results_overlay", "watermark",
]


def test_registry_populates_in_canonical_order():
    import osu_mania_renderer_v2.render.pipeline  # noqa: F401 — populate registries
    from osu_mania_renderer_v2.render.compositor import ELEMENTS, RENDER_ORDER

    assert RENDER_ORDER == EXPECTED_ORDER
    assert all(name in ELEMENTS for name in RENDER_ORDER)
    # Phase 1: every element delegates (no declared assets/variables yet).
    for name in RENDER_ORDER:
        spec = ELEMENTS[name]
        assert spec.asset_basenames == ()
        assert callable(spec.render_fn)


@pytest.mark.slow
def test_wiki_path_renders_argon(tmp_path, fixtures_dir):
    """The wiki path renders the fixture (Argon default look) to a valid,
    non-trivial MP4 without error. The wiki path intentionally diverges from
    the legacy renderer now (Argon default vs the classic fallback), so this
    is a render smoke test, not a byte-identical parity check."""
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required (GL + ffmpeg)")
    import asyncio

    from osu_mania_renderer_v2.models import RenderOptions
    import osu_mania_renderer_v2.render.pipeline  # noqa: F401 — populate registries
    from osu_mania_renderer_v2.render.compositor import render as wiki

    osr = fixtures_dir / "ao_infinity_hard.osr"
    skin = tmp_path / "emptyskin"
    skin.mkdir()
    dflt = Path(__file__).resolve().parent.parent / "osu_mania_renderer_v2" / "assets" / "default_skin"
    out = tmp_path / "wiki.mp4"

    asyncio.run(wiki(
        osr_path=osr, beatmap_dir=fixtures_dir, output_path=out,
        options=RenderOptions(resolution=(854, 480), fps=30),
        skin_dir=skin, default_skin_dir=dflt,
    ))
    assert out.exists() and out.stat().st_size > 100_000
