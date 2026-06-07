import os

import pytest

from osu_mania_renderer_v2.gpu.context import HeadlessGl


@pytest.mark.slow
def test_open_context_and_fbo():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("Set RUN_SLOW=1 to run GL smoke tests")
    with HeadlessGl(width=128, height=64) as ctx:
        assert ctx.fbo.size == (128, 64)
        ctx.fbo.clear(0.25, 0.5, 0.75, 1.0)
        data = ctx.fbo.read(components=3)
        # Sample any pixel: should be ~(64, 128, 191) for rgb24.
        r, g, b = data[0], data[1], data[2]
        assert abs(r - 64) <= 2
        assert abs(g - 128) <= 2
        assert abs(b - 191) <= 2


def test_context_close_idempotent():
    # Without a real GL, we still want the API to be safe to call twice.
    h = HeadlessGl.__new__(HeadlessGl)
    h._ctx = None
    h._fbo = None
    h._color = None
    h._depth = None
    h.close()
    h.close()  # should not raise
