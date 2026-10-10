"""What the readback pool's buffers are declared to be for (gpu/readback.py,
the switch R3D_MANIA_STREAM_READ).

What must hold:
  * the default is on on macOS (where it was measured) and off anywhere else;
    R3D_MANIA_STREAM_READ forces it either way;
  * with it on, every pool buffer is GL_STREAM_READ and still one frame long;
    with it off the pool is what it always was;
  * the frames that come out are the same bytes on and off: RGB and yuv420p,
    mapped and copied, frame after frame past the size of the pool, and the
    drained tail.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from osu_mania_renderer_v2.gpu import readback
from osu_mania_renderer_v2.gpu import yuv as Y


def test_the_switch(monkeypatch):
    monkeypatch.delenv("R3D_MANIA_STREAM_READ", raising=False)
    monkeypatch.setattr(readback.sys, "platform", "darwin")
    assert readback._stream_read() is True
    monkeypatch.setattr(readback.sys, "platform", "linux")
    assert readback._stream_read() is False
    for v, want in (("1", True), ("on", True), ("0", False), ("off", False), ("", False)):
        monkeypatch.setenv("R3D_MANIA_STREAM_READ", v)
        monkeypatch.setattr(readback.sys, "platform", "linux" if want else "darwin")
        assert readback._stream_read() is want, v


def _gl(width, height):
    from osu_mania_renderer_v2.gpu.context import HeadlessGl
    return HeadlessGl(width=width, height=height)


def _paint(gl, rgb_bottom_up):
    h, w = rgb_bottom_up.shape[:2]
    rgba = np.dstack([rgb_bottom_up, np.full((h, w, 1), 255, np.uint8)])
    gl.fbo.color_attachments[0].write(np.ascontiguousarray(rgba).tobytes())


def _pool(reader):
    """(usage, size) of every pool buffer, asked of GL itself."""
    from OpenGL import GL
    out = []
    for slot in reader._slots:
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, slot.pbo.glo)
        out.append(tuple(
            int(np.asarray(GL.glGetBufferParameteriv(GL.GL_PIXEL_PACK_BUFFER, what)).reshape(-1)[0])
            for what in (GL.GL_BUFFER_USAGE, GL.GL_BUFFER_SIZE)))
    GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
    return out


def _frames(gl, reader, pictures):
    out = []
    for rgb in pictures:
        _paint(gl, rgb)
        f = reader.read()
        if isinstance(f, readback.MappedFrame):
            out.append(bytes(f.mv))
            f.done.set()                      # what the ffmpeg writer does
        else:
            out.append(bytes(f))
    return out + [bytes(f) for f in reader.drain()]


@pytest.mark.slow
@pytest.mark.parametrize("on", [True, False])
def test_the_pool_is_declared_as_asked(on, monkeypatch):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run real GL checks")
    from OpenGL import GL
    monkeypatch.setenv("R3D_MANIA_STREAM_READ", "1" if on else "0")
    width, height = 128, 96
    with _gl(width, height) as gl:
        for reader in (readback.FrameReader(gl.ctx, gl.fbo, components=3),
                       readback.FrameReader(gl.ctx, gl.fbo, components=3,
                                            yuv=Y.YuvConverter(gl.ctx, width, height))):
            assert reader.stream_read is on
            pool = _pool(reader)
            assert len(pool) == reader.pool_size
            want = GL.GL_STREAM_READ if on else GL.GL_DYNAMIC_DRAW
            assert pool == [(int(want), reader.frame_size)] * reader.pool_size


@pytest.mark.slow
@pytest.mark.parametrize("mapped", [True, False])
@pytest.mark.parametrize("gpu_yuv", [True, False])
def test_the_same_frames_on_and_off(mapped, gpu_yuv, monkeypatch):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run real GL pixel checks")
    if mapped:
        monkeypatch.delenv("R3D_MANIA_NO_MAPPED_READBACK", raising=False)
    else:
        monkeypatch.setenv("R3D_MANIA_NO_MAPPED_READBACK", "1")
    width, height, n = 128, 96, 20            # 20 frames: every pool buffer is reused
    rng = np.random.default_rng(11)
    pictures = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(n)]
    got = {}
    for on in (True, False):
        monkeypatch.setenv("R3D_MANIA_STREAM_READ", "1" if on else "0")
        with _gl(width, height) as gl:
            conv = Y.YuvConverter(gl.ctx, width, height) if gpu_yuv else None
            reader = readback.FrameReader(gl.ctx, gl.fbo, components=3, yuv=conv)
            assert reader.stream_read is on
            assert n > reader.pool_size
            got[on] = _frames(gl, reader, pictures)
    assert got[True] == got[False]
    # and they are the right frames, not merely the same ones
    blank = Y.black_frame(width, height) if gpu_yuv else bytes(width * height * 3)
    assert len(got[True]) == n + readback._LAG
    assert got[True][:readback._LAG] == [blank] * readback._LAG
    for i, rgb in enumerate(pictures):
        want = Y.rgb_to_yuv420p(rgb).tobytes() if gpu_yuv else rgb.tobytes()
        assert got[True][readback._LAG + i] == want, f"frame {i}"


@pytest.mark.slow
def test_a_failed_declaration_keeps_the_pool(monkeypatch):
    """The hint is an optimisation: if GL refuses it, readback goes on as before."""
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run real GL checks")
    monkeypatch.setenv("R3D_MANIA_STREAM_READ", "1")

    def boom(self):
        raise RuntimeError("no")
    monkeypatch.setattr(readback.FrameReader, "_declare_stream_read", boom)
    width, height = 128, 96
    rng = np.random.default_rng(3)
    pictures = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(5)]
    with _gl(width, height) as gl:
        reader = readback.FrameReader(gl.ctx, gl.fbo, components=3)
        assert reader.stream_read is False and reader._slots
        out = _frames(gl, reader, pictures)
    assert out[readback._LAG:] == [p.tobytes() for p in pictures]
