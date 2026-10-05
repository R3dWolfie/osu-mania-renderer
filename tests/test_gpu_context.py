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


@pytest.mark.parametrize('platform',['win32','darwin'])
def test_platform_default_never_loads_egl_even_when_egl_env_is_set(monkeypatch,platform):
    from unittest.mock import MagicMock
    from osu_mania_renderer_v2.gpu import context
    gl=MagicMock();gl.info={'GL_RENDERER':'platform-default'}
    standalone=MagicMock(return_value=gl)
    monkeypatch.setattr(context.sys,'platform',platform)
    monkeypatch.setenv('R3D_EGL_DEVICE_INDEX','not-an-egl-device')
    monkeypatch.setenv('MODERNGL_BACKEND','egl')
    monkeypatch.setattr(context.moderngl,'create_standalone_context',standalone)
    def forbidden(*args,**kwargs):
        raise AssertionError('EGL must not be loaded on Windows/macOS')
    monkeypatch.setattr(context,'_create_pinned_egl_context',forbidden)
    monkeypatch.setattr(context.ctypes,'CDLL',forbidden)
    with context.HeadlessGl(16,16):
        standalone.assert_called_once_with(require=330)
    gl.release.assert_called_once()


@pytest.mark.parametrize('device',[None,'2'])
def test_linux_retains_default_egl_and_device_pinning(monkeypatch,device):
    from unittest.mock import MagicMock
    from osu_mania_renderer_v2.gpu import context
    gl=MagicMock();gl.info={'GL_RENDERER':'egl'}
    standalone=MagicMock(return_value=gl);pinned=MagicMock(return_value=gl)
    monkeypatch.setattr(context.sys,'platform','linux')
    monkeypatch.delenv('MODERNGL_BACKEND',raising=False)
    if device is None: monkeypatch.delenv('R3D_EGL_DEVICE_INDEX',raising=False)
    else: monkeypatch.setenv('R3D_EGL_DEVICE_INDEX',device)
    monkeypatch.setattr(context.moderngl,'create_standalone_context',standalone)
    monkeypatch.setattr(context,'_create_pinned_egl_context',pinned)
    with context.HeadlessGl(16,16): pass
    if device is None:
        standalone.assert_called_once_with(backend='egl',require=330)
        pinned.assert_not_called()
    else:
        pinned.assert_called_once_with(2)
        standalone.assert_not_called()
