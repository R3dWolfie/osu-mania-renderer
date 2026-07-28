"""Background-video decode for the mania renderer.

osu! maps can ship a background video (the ``Video,<start>,"file"`` event).
When ``RenderOptions.load_video`` is on we play it behind gameplay. We decode
through ffmpeg — already a hard dependency for output encoding, so no OpenCV —
scaling+cropping each frame to *cover* the canvas (same framing as the static
background) and resampling to the render fps, then read raw RGBA frames over a
pipe.

A small reader thread pulls frames into a bounded queue so ffmpeg decodes a few
frames ahead while the GPU draws the previous one (the pipe's own buffer is too
small to overlap on its own). The render loop calls :meth:`frame_for` once per
output frame with that frame's song time; the decoder advances 1:1 (both sides
run at the same fps after the ``fps`` filter) and returns the current frame's
bytes — or ``None`` before the video's start offset, when decode failed, or
once it ends (the caller then falls back to the static background image).
"""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
from pathlib import Path

from osu_mania_renderer_v2.render.encode import _ffmpeg_prefix

log = logging.getLogger("osu_mania_renderer_v2")

# Frames buffered ahead of the render loop. Enough to overlap decode with the
# GPU draw without hoarding memory — a 4K RGBA frame is ~33 MB, so 4 frames is
# ~130 MB worst case at 2160p, a few MB at 720p.
_QUEUE_DEPTH = 4


class VideoBackground:
    """Streaming RGBA decode of a beatmap background video, frame-indexed by
    song time. One per render; call :meth:`close` when the render finishes."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        start_ms: int,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self.start_ms = start_ms
        self.frame_size = width * height * 4
        self._q: queue.Queue[bytes | None] = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._eof = False
        self._failed = False
        self._cur_idx = -1            # index of the frame currently held
        self._cur_bytes: bytes | None = None
        self._frames_seen = 0

        # cover = fill the canvas preserving aspect, crop the overflow (the
        # static-bg path does the identical thing in PIL). fps= resamples the
        # source to the render rate so frame N here == render frame N once the
        # video has started.
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},fps={self.fps},format=rgba"
        )
        cmd = [
            *_ffmpeg_prefix(),
            "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-an", "-sn",
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0,
            )
        except OSError as e:
            log.warning("video_bg_spawn_failed err=%s", e)
            self._proc = None
            self._failed = True
            return
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    # ── decode thread ──
    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        fs = self.frame_size
        try:
            while True:
                buf = self._read_exact(stdout, fs)
                if buf is None:
                    break
                self._q.put(buf)      # blocks when full → backpressures ffmpeg
        finally:
            self._q.put(None)         # EOF sentinel

    @staticmethod
    def _read_exact(stream, n: int) -> bytes | None:
        """Read exactly n bytes (a full frame) or None at EOF/short read."""
        chunks: list[bytes] = []
        got = 0
        while got < n:
            chunk = stream.read(n - got)
            if not chunk:
                return None
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    # ── render-loop API ──
    def active_at(self, t_ms: int) -> bool:
        """True once the video has started (and decode hasn't failed)."""
        return (not self._failed) and t_ms >= self.start_ms

    def frame_for(self, t_ms: int) -> bytes | None:
        """RGBA bytes for the output frame at song time ``t_ms`` (top-left
        origin, width*height*4), or ``None`` before the video starts, on
        failure, before the first frame is available, or once the video has
        ended (the caller then shows the static background image — matching
        osu!, which drops back to the bg image when the video finishes)."""
        if self._failed or t_ms < self.start_ms:
            return None
        target = int(round((t_ms - self.start_ms) * self.fps / 1000.0))
        while self._cur_idx < target and not self._eof:
            try:
                # Small timeout so a stalled decode can't wedge the render
                # loop forever; on timeout we just reuse the current frame.
                buf = self._q.get(timeout=30.0)
            except queue.Empty:
                log.warning("video_bg_decode_stall idx=%s", self._cur_idx)
                break
            if buf is None:
                self._eof = True
                break
            self._cur_bytes = buf
            self._cur_idx += 1
            self._frames_seen += 1
        # Video has run out of frames and the song has moved past its end →
        # let the static background take over rather than freezing on the last
        # frame. (A transient decode stall keeps the current frame instead.)
        if self._eof and self._cur_idx < target:
            return None
        return self._cur_bytes

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.kill()
        except Exception:  # noqa: BLE001
            pass
        # Drain so a reader blocked on a full queue can finish and exit.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if self._frames_seen == 0 and not self._failed:
            log.warning("video_bg_no_frames (decode produced nothing)")
