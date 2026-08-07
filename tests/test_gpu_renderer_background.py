import os
from pathlib import Path

import pytest
from PIL import Image

from osu_mania_renderer_v2.beatmap.models import VisualMods
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.render.scene import SceneState


@pytest.mark.slow
def test_background_image_visible(tmp_path: Path):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    bg_path = tmp_path / "bg.png"
    Image.new("RGB", (128, 128), (200, 60, 60)).save(bg_path)
    W, H = 256, 256
    with HeadlessGl(width=W, height=H) as gl:
        rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=W, height=H, key_count=4)
        fr = FrameRenderer(rc)
        fr.set_background(bg_path)
        scene = SceneState(
            t_ms=0, visible_notes=(), keys_held=(False,)*4,
            visual_mods=VisualMods(),
        )
        fr.draw(scene)
        data = gl.fbo.read(components=3)
        # Background is dimmed, so red channel should dominate but be reduced.
        r, g, b = data[0], data[1], data[2]
        assert r > g and r > b
        # Dimmed: not full intensity.
        assert r < 200
