import os

import pytest

from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.readback import FrameReader


@pytest.mark.slow
def test_readback_returns_correct_size():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    W, H = 128, 64
    with HeadlessGl(width=W, height=H) as gl:
        reader = FrameReader(gl.ctx, gl.fbo, components=3)
        gl.fbo.clear(0.1, 0.2, 0.3, 1.0)
        frame = reader.read()
        assert len(frame) == W * H * 3


@pytest.mark.slow
def test_readback_double_buffered():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    W, H = 64, 64
    with HeadlessGl(width=W, height=H) as gl:
        reader = FrameReader(gl.ctx, gl.fbo, components=3, ring=2)
        for color in [(1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)]:
            gl.fbo.clear(*color, 1.0)
            frame = reader.read()
            assert len(frame) == W * H * 3
