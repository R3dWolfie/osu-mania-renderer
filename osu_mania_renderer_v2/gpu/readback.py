"""Framebuffer → CPU readback with a ring of Pixel Buffer Objects.

Without PBOs, ``glReadPixels`` is synchronous: every frame's draw chain
must fully complete on the GPU before the call returns, so the CPU and
GPU end up alternating instead of overlapping. With a ring of PBOs we
issue the read into PBO[N] (the GPU starts copying asynchronously) and
then map PBO[N-K] from K frames ago to actually pull bytes back to CPU.
By the time we map an older PBO its data is already settled in pinned
host-visible memory, so the GPU is never stalled.

Falls back to the synchronous ``fbo.read()`` if PyOpenGL isn't available
or the GL driver doesn't honour async pack — same wire shape, just no
overlap.
"""
from __future__ import annotations

import logging

import moderngl

log = logging.getLogger("osu_mania_renderer_v2.gpu.readback")

try:
    from OpenGL import GL as _GL
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False


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
        self.ring_size = max(1, ring)
        self.w, self.h = fbo.size
        self.frame_size = self.w * self.h * components
        self._pbos: list[moderngl.Buffer] = []
        self._frame_idx = 0
        # Empty placeholder bytes returned for the first (ring - 1) reads
        # while the pipeline is warming up. Discarded by the caller? No —
        # they're real frames; the caller doesn't care about ordering, only
        # that we hand back N frames total. To keep frame count correct we
        # emit a black frame for the warmup slots; the user-visible video
        # starts with a black fade-in anyway.
        self._warmup_blank = bytes(self.frame_size)
        if _GL_AVAILABLE:
            try:
                # Allocate the ring of pack-side PBOs. moderngl's Buffer
                # carries a GL name (`.glo`) we can bind as a pixel-pack
                # buffer via PyOpenGL.
                self._pbos = [
                    ctx.buffer(reserve=self.frame_size, dynamic=True)
                    for _ in range(self.ring_size)
                ]
                log.info("pbo_readback_enabled", extra={"ring": self.ring_size})
            except Exception as e:  # noqa: BLE001
                log.warning("pbo_alloc_failed_fallback_sync",
                            extra={"err": str(e)})
                self._pbos = []
        else:
            log.warning("pyopengl_missing_fallback_sync")

    def read(self) -> bytes:
        """Return one frame's worth of RGB bytes. With PBO mode active,
        the bytes returned are from the read issued K=ring_size frames
        ago — perfectly fine for streaming to ffmpeg since each frame is
        independent rawvideo, but it does introduce a `ring_size`-frame
        latency at the start (filled with black) and end (truncated)."""
        if not self._pbos:
            return self.fbo.read(components=self.components)

        ring = self.ring_size
        current = self._frame_idx % ring
        next_to_map = (self._frame_idx - (ring - 1)) % ring

        # Issue an async read into the current PBO.
        pbo = self._pbos[current]
        self.fbo.use()
        _GL.glBindBuffer(_GL.GL_PIXEL_PACK_BUFFER, pbo.glo)
        # offset=0 (use bound PBO instead of CPU pointer)
        _GL.glReadPixels(
            0, 0, self.w, self.h, _GL.GL_RGB, _GL.GL_UNSIGNED_BYTE, 0,
        )
        _GL.glBindBuffer(_GL.GL_PIXEL_PACK_BUFFER, 0)

        out: bytes
        if self._frame_idx < ring - 1:
            # Pipeline still warming up — emit black for these initial
            # slots; the actual frames will come out at the end.
            out = self._warmup_blank
        else:
            # Map the buffer that's been settled for (ring - 1) frames.
            out = bytes(self._pbos[next_to_map].read())

        self._frame_idx += 1
        return out

    def drain(self) -> list[bytes]:
        """Flush any frames still in flight in the PBO ring. Call once
        after the main render loop so the last `ring_size - 1` frames'
        reads complete and we can hand them to ffmpeg."""
        if not self._pbos:
            return []
        ring = self.ring_size
        out: list[bytes] = []
        # The most recently-issued read is at (self._frame_idx - 1) % ring.
        # We've already mapped buffers that are (ring - 1) older than the
        # newest issued. Remaining unread = newest, newest-1, ... newest-
        # (ring-2). Map them in chronological order.
        for k in range(ring - 1, 0, -1):
            idx = (self._frame_idx - k) % ring
            if self._frame_idx >= k:  # actually issued
                out.append(bytes(self._pbos[idx].read()))
        return out
