import os

import pytest

from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.shaders import load_programs


@pytest.mark.slow
def test_load_programs():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    with HeadlessGl(width=64, height=64) as gl:
        progs = load_programs(gl.ctx)
        assert "sprite" in progs
        assert "flashlight" in progs


def test_shader_source_files_exist():
    from osu_mania_renderer_v2.gpu import shaders
    for name in ("sprite.vert", "sprite.frag", "flashlight.frag"):
        assert (shaders.SHADERS_DIR / name).is_file()
