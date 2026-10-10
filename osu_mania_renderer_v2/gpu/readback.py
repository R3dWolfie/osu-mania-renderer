"""Framebuffer → CPU readback with a pool of Pixel Buffer Objects.

Without PBOs, ``glReadPixels`` is synchronous: every frame's draw chain
must fully complete on the GPU before the call returns, so the CPU and
GPU end up alternating instead of overlapping. With PBOs we issue the
read into a pool buffer (the GPU starts copying asynchronously) and only
consume it two frames later, when its data has settled in pinned
host-visible memory, so the GPU is never stalled.

Two consume paths, same output bytes and the same 2-frame pipeline lag
(two black warm-up frames at the start, two drained frames at the end):

  * mapped (default) — glMapBufferRange the settled PBO and hand the
    mapped memoryview straight to the ffmpeg writer thread as a
    ``MappedFrame`` lease; the writer os.write()s from pinned memory and
    sets ``done``, and the PBO is unmapped/reused on the GL thread once
    that event is set. Skips the 6 MB CPU copy every frame.
  * copy — read_into() a rotated pool of reusable bytearrays (the old
    behaviour). Fallback when PyOpenGL/mapping is unavailable or
    ``R3D_MANIA_NO_MAPPED_READBACK=1``.

Falls back to the fully synchronous ``fbo.read()`` if PyOpenGL isn't
available — same wire shape, just no overlap.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from collections import deque

import moderngl

log = logging.getLogger("osu_mania_renderer_v2.gpu.readback")

try:
    from OpenGL import GL as _GL
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False

# How many frames a readback stays in flight before we consume it. This is
# part of the OUTPUT contract (that many black frames lead the video, and
# the same count is flushed by drain()) — do not change it casually.
_LAG = 2

# Seconds to wait for the ffmpeg writer to release a mapped frame before
# declaring the render wedged. Generous — the writer normally completes a
# frame in ~2 ms.
_LEASE_TIMEOUT_S = 30.0


# What the pool's buffers are declared to be for. ctx.buffer(dynamic=True)
# asks for GL_DYNAMIC_DRAW ("the application writes it, the GPU draws from
# it"). A readback buffer is the opposite: the GPU writes it and the
# application reads it once, which is GL_STREAM_READ.
#
# On macOS the hint decides what a map costs. With DYNAMIC_DRAW every
# glMapBufferRange took 0.6 ms a frame at 1280x720 on the draw thread (2.6 s
# of a 70 s replay), with the GPU already finished (a fence on the read was
# signalled every time). With STREAM_READ the map returns at once, and
# reading the mapped bytes afterwards is cheap as well (0.03 ms a frame): the
# time was the map call itself, not the bytes. Same bytes either way; only
# the usage hint of the pool changes.
#
# On by default on macOS, where that was measured. Anywhere else it is off
# until it has been measured there; R3D_MANIA_STREAM_READ=1 / =0 forces it.
def _stream_read_default() -> bool:
    return sys.platform == "darwin"


def _stream_read() -> bool:
    v = os.environ.get("R3D_MANIA_STREAM_READ")
    if v is None:
        return _stream_read_default()
    return v.strip().lower() not in ("", "0", "false", "no", "off")


class MappedFrame:
    """A frame whose bytes live in mapped GL memory. The ffmpeg writer
    thread writes ``mv`` to the pipe and then sets ``done``; only after
    that may the GL thread unmap and reuse the underlying PBO."""

    __slots__ = ("mv", "done")

    def __init__(self, mv: memoryview) -> None:
        self.mv = mv
        self.done = threading.Event()

    def __len__(self) -> int:  # parity with bytes for callers that log sizes
        return len(self.mv)


class _Slot:
    __slots__ = ("pbo", "lease")

    def __init__(self, pbo: moderngl.Buffer) -> None:
        self.pbo = pbo
        self.lease: MappedFrame | None = None  # non-None ⇒ currently mapped


class FrameReader:
    def __init__(
        self,
        ctx: moderngl.Context,
        fbo: moderngl.Framebuffer,
        components: int = 3,
        ring: int = 3,
        yuv=None,
    ) -> None:
        self.ctx = ctx
        self.fbo = fbo
        self.components = components
        # `yuv` (a gpu.yuv.YuvConverter, R3D_MANIA_GPU_YUV): every frame is
        # converted to yuv420p on the GPU before it is read back, so a frame
        # is 1.5 bytes a pixel instead of 3. Same pool, same lag, same leases.
        self.yuv = yuv
        # `ring` kept for API compatibility; the pool is sized to cover the
        # 2-frame lag + every lease the ffmpeg writer can hold in flight
        # (2 queued + 1 writing) + slack.
        self.pool_size = max(6, ring + 3)
        self.w, self.h = fbo.size
        self._frame_idx = 0
        if yuv is not None:
            from osu_mania_renderer_v2.gpu.yuv import black_frame
            self.frame_size = self.w * self.h * 3 // 2
            # black in yuv is Y 16, U/V 128; zeros would be a green frame
            self._warmup_blank = black_frame(self.w, self.h)
        else:
            self.frame_size = self.w * self.h * components
            self._warmup_blank = bytes(self.frame_size)
        self._mapped_mode = False
        self.stream_read = False
        self._free: deque[_Slot] = deque()
        self._pending: deque[_Slot] = deque()   # issued reads, oldest first
        self._leased: deque[_Slot] = deque()    # mapped + handed out, oldest first
        self._slots: list[_Slot] = []
        # Copy-path state (fallback): rotated reusable output buffers.
        self._out_pool: list[bytearray] = []
        self._out_idx = 0
        if _GL_AVAILABLE:
            try:
                self._slots = [
                    _Slot(ctx.buffer(reserve=self.frame_size, dynamic=True))
                    for _ in range(self.pool_size)
                ]
                self._free.extend(self._slots)
                self._mapped_mode = (
                    os.environ.get("R3D_MANIA_NO_MAPPED_READBACK") != "1"
                )
                self.stream_read = _stream_read()
                if self.stream_read:
                    try:
                        self._declare_stream_read()
                    except Exception as e:  # noqa: BLE001 — a hint, never the pool
                        log.warning("pbo_stream_read_failed",
                                    extra={"err": str(e)})
                        self.stream_read = False
                log.info("pbo_readback_enabled",
                         extra={"pool": self.pool_size,
                                "mapped": self._mapped_mode,
                                "stream_read": self.stream_read})
            except Exception as e:  # noqa: BLE001
                log.warning("pbo_alloc_failed_fallback_sync",
                            extra={"err": str(e)})
                self._slots = []
                self._free.clear()
        else:
            log.warning("pyopengl_missing_fallback_sync")
        if self._slots and not self._mapped_mode:
            self._out_pool = [bytearray(self.frame_size)
                              for _ in range(self.pool_size)]

    # ---- GL helpers (GL thread only) ----

    def _declare_stream_read(self) -> None:
        """Give every pool buffer storage of the same size declared
        GL_STREAM_READ (see _stream_read). Nothing has been read into the
        pool yet, so there are no contents to keep."""
        for slot in self._slots:
            _GL.glBindBuffer(_GL.GL_PIXEL_PACK_BUFFER, slot.pbo.glo)
            _GL.glBufferData(_GL.GL_PIXEL_PACK_BUFFER, self.frame_size, None,
                             _GL.GL_STREAM_READ)
        _GL.glBindBuffer(_GL.GL_PIXEL_PACK_BUFFER, 0)

    def _issue_read(self, slot: _Slot) -> None:
        if self.yuv is not None:
            self.yuv.run(self.fbo.color_attachments[0])
            self.yuv.read_into(slot.pbo)
            self.fbo.use()          # the renderer expects its own target bound
            return
        self.fbo.use()
        _GL.glBindBuffer(_GL.GL_PIXEL_PACK_BUFFER, slot.pbo.glo)
        # offset=0 (use bound PBO instead of CPU pointer)
        _GL.glReadPixels(
            0, 0, self.w, self.h, _GL.GL_RGB, _GL.GL_UNSIGNED_BYTE, 0,
        )
        _GL.glBindBuffer(_GL.GL_PIXEL_PACK_BUFFER, 0)

    def _map_slot(self, slot: _Slot) -> memoryview | None:
        _GL.glBindBuffer(_GL.GL_PIXEL_PACK_BUFFER, slot.pbo.glo)
        ptr = _GL.glMapBufferRange(
            _GL.GL_PIXEL_PACK_BUFFER, 0, self.frame_size, _GL.GL_MAP_READ_BIT,
        )
        _GL.glBindBuffer(_GL.GL_PIXEL_PACK_BUFFER, 0)
        addr = getattr(ptr, "value", ptr) if not isinstance(ptr, int) else ptr
        if not addr:
            return None
        arr = (ctypes.c_ubyte * self.frame_size).from_address(addr)
        return memoryview(arr)

    def _unmap_slot(self, slot: _Slot) -> None:
        _GL.glBindBuffer(_GL.GL_PIXEL_PACK_BUFFER, slot.pbo.glo)
        _GL.glUnmapBuffer(_GL.GL_PIXEL_PACK_BUFFER)
        _GL.glBindBuffer(_GL.GL_PIXEL_PACK_BUFFER, 0)
        slot.lease = None

    def _reclaim(self, *, block: bool) -> None:
        """Unmap every leased slot the writer has finished with. When
        ``block`` and nothing is free, wait for the OLDEST lease —
        that's the pipe's natural backpressure point."""
        while self._leased and self._leased[0].lease.done.is_set():
            slot = self._leased.popleft()
            self._unmap_slot(slot)
            self._free.append(slot)
        if block and not self._free:
            slot = self._leased.popleft()
            if not slot.lease.done.wait(_LEASE_TIMEOUT_S):
                raise RuntimeError(
                    "ffmpeg writer did not release a mapped frame within "
                    f"{_LEASE_TIMEOUT_S}s — encoder wedged?"
                )
            self._unmap_slot(slot)
            self._free.append(slot)

    # ---- public API (GL thread only) ----

    def read(self) -> bytes | bytearray | MappedFrame:
        """Return one frame's worth of RGB bytes (or a MappedFrame lease in
        mapped mode). The bytes returned are from the read issued _LAG
        frames ago — fine for streaming to ffmpeg since each frame is
        independent rawvideo, but it does introduce a 2-frame latency at
        the start (filled with black) and end (flushed by drain())."""
        if not self._slots:
            if self.yuv is not None:
                self.yuv.run(self.fbo.color_attachments[0])
                frame = self.yuv.read_bytes()
                self.fbo.use()
                return frame
            return self.fbo.read(components=self.components)

        if self._mapped_mode:
            self._reclaim(block=not self._free)
            slot = self._free.popleft()
        else:
            slot = self._free.popleft()
            self._free.append(slot)  # copy path never holds slots long
        self._issue_read(slot)
        self._pending.append(slot)

        self._frame_idx += 1
        if self._frame_idx <= _LAG:
            # Pipeline warming up — emit black; the real frames come out
            # at the end via drain(). Same contract as the old ring.
            return self._warmup_blank

        settled = self._pending.popleft()
        if not self._mapped_mode:
            out = self._next_out()
            settled.pbo.read_into(out)
            return out
        mv = self._map_slot(settled)
        if mv is None:
            # Mapping failed — permanently fall back to the copy path.
            log.warning("pbo_map_failed_fallback_copy")
            self._mapped_mode = False
            self._out_pool = [bytearray(self.frame_size)
                              for _ in range(self.pool_size)]
            out = self._next_out()
            settled.pbo.read_into(out)
            self._free.append(settled)
            return out
        lease = MappedFrame(mv)
        settled.lease = lease
        self._leased.append(settled)
        return lease

    def _next_out(self) -> bytearray:
        buf = self._out_pool[self._out_idx]
        self._out_idx = (self._out_idx + 1) % len(self._out_pool)
        return buf

    def drain(self) -> list[bytes]:
        """Flush the reads still in flight (the last _LAG frames), AS
        COPIES, and release every mapped lease. After drain() returns no
        GL memory is referenced by anyone — required because the caller
        tears the GL context down before ffmpeg finishes the tail writes."""
        if not self._slots:
            return []
        # 1. Wait out + unmap everything the writer still holds.
        while self._leased:
            slot = self._leased.popleft()
            if not slot.lease.done.wait(_LEASE_TIMEOUT_S):
                raise RuntimeError(
                    "ffmpeg writer did not release a mapped frame within "
                    f"{_LEASE_TIMEOUT_S}s during drain — encoder wedged?"
                )
            self._unmap_slot(slot)
            self._free.append(slot)
        # 2. Copy out the still-pending tail reads, oldest first. Plain
        # CPU copies on purpose: nothing may reference GL memory after
        # drain() returns.
        out: list[bytes | bytearray] = []
        while self._pending:
            slot = self._pending.popleft()
            buf = bytearray(self.frame_size)
            slot.pbo.read_into(buf)
            out.append(buf)
            if self._mapped_mode and slot not in self._free:
                self._free.append(slot)
        return out


def reader_for_plan(ctx: moderngl.Context, fbo: moderngl.Framebuffer, *,
                    gpu_yuv: bool):
    """The frame reader that matches the ffmpeg command a render plan was
    built with. EVERY render loop must take its reader from here: the plan
    decides whether ffmpeg expects yuv420p frames (``plan.gpu_yuv``), and a
    loop that hands it RGB instead sends frames of the wrong size.

    With ``gpu_yuv`` the frames are converted on the GPU (gpu/yuv.py). If
    those passes cannot be built on this machine, ffmpeg is already waiting
    for yuv420p, so the frames are read the stock way and converted on the
    CPU with the same arithmetic: late, not lost."""
    if not gpu_yuv:
        return FrameReader(ctx, fbo, components=3)
    from osu_mania_renderer_v2.gpu import yuv as _yuv
    CpuTwinReader = _yuv.CpuTwinReader
    w, h = fbo.size
    try:
        return FrameReader(ctx, fbo, components=3, yuv=_yuv.make_converter(ctx, w, h))
    except Exception as e:  # noqa: BLE001
        log.warning("gpu_yuv_unavailable_cpu_twin", extra={"err": str(e)})
        return CpuTwinReader(FrameReader(ctx, fbo, components=3), w, h)

