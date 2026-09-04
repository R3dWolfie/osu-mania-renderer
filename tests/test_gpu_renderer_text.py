import os

import pytest

from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.gpu.text import text_to_texture
from osu_mania_renderer_v2.models import VisualMods
from osu_mania_renderer_v2.scene import SceneState


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
        scene = SceneState(
            t_ms=0, visible_notes=(), keys_held=(False,)*4,
            visual_mods=VisualMods(),
            score=865_612, combo=1305, max_combo=1305, accuracy=98.45,
        )
        fr.draw(scene)
        data = gl.fbo.read(components=3)
        assert max(data) > 50
