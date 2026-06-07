import os

import pytest

from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.models import VisualMods
from osu_mania_renderer_v2.scene import JudgmentPopup, SceneState


@pytest.mark.slow
def test_receptors_drawn_at_bottom():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    W, H = 320, 240
    with HeadlessGl(width=W, height=H) as gl:
        rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=W, height=H, key_count=4)
        fr = FrameRenderer(rc)
        scene = SceneState(
            t_ms=0, visible_notes=(),
            keys_held=(True, False, True, False),
            visual_mods=VisualMods(),
        )
        fr.draw(scene)
        data = gl.fbo.read(components=3)
        assert max(data) > 50


@pytest.mark.slow
def test_judgment_popup_drawn():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    W, H = 320, 240
    with HeadlessGl(width=W, height=H) as gl:
        rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=W, height=H, key_count=4)
        fr = FrameRenderer(rc)
        scene = SceneState(
            t_ms=0, visible_notes=(), keys_held=(False,)*4,
            visual_mods=VisualMods(),
            active_judgments=(JudgmentPopup(column=2, judgment="300", age_ms=100),),
        )
        fr.draw(scene)
        data = gl.fbo.read(components=3)
        assert max(data) > 50
