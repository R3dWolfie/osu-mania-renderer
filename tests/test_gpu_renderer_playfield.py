import os

import pytest

from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.models import VisualMods
from osu_mania_renderer_v2.scene import SceneState, VisibleNote


@pytest.mark.slow
def test_renders_a_single_note():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    W, H = 256, 256
    with HeadlessGl(width=W, height=H) as gl:
        rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, width=W, height=H, key_count=4)
        fr = FrameRenderer(rc)
        scene = SceneState(
            t_ms=0,
            visible_notes=(
                VisibleNote(column=1, is_hold=False, y_fraction=0.5,
                            head_y_fraction=0.5, tail_y_fraction=0.5),
            ),
            keys_held=(False, False, False, False),
            visual_mods=VisualMods(),
        )
        fr.draw(scene)
        data = gl.fbo.read(components=3)
        # At least one pixel should be brighter than the background.
        assert max(data) > 50
