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
    ) -> None:
        self.ctx = ctx
        self.fbo = fbo
        self.components = components
        # `ring` kept for API compatibility; the pool is sized to cover the
        # 2-frame lag + every lease the ffmpeg writer can hold in flight
        # (2 queued + 1 writing) + slack.
        self.pool_size = max(6, ring + 3)
        self.w, self.h = fbo.size
        self.frame_size = self.w * self.h * components
        self._frame_idx = 0
        self._warmup_blank = bytes(self.frame_size)
        self._mapped_mode = False
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
                log.info("pbo_readback_enabled",
                         extra={"pool": self.pool_size,
                                "mapped": self._mapped_mode})
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

    def _issue_read(self, slot: _Slot) -> None:
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
