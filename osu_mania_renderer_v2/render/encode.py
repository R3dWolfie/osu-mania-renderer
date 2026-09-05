"""ffmpeg subprocess: probe encoder, build command, spawn, manage stdin frames.

When the renderer runs inside a toolbox/distrobox container (where Mesa
gives us OpenGL but the in-container ffmpeg's VAAPI driver is broken), we
route the ffmpeg subprocess to the HOST's ffmpeg via `flatpak-spawn
--host`. stdin/stdout still proxy correctly so the frame pipe works
unchanged, and the host ffmpeg gets full VAAPI access — turning a
CPU-encoded 4× real-time render into a hardware-encoded 1× one.
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
from pathlib import Path

from osu_mania_renderer_v2.errors import EncoderError

# Single-pass loudnorm applied to the MUSIC ALONE (the 2026-07-12 #17 duck
# fix — normalising the song before hits are mixed on top). The SAME string is
# used as (a) the fused filter in build_ffmpeg_cmd below and (b) part of the
# shared loudnorm-cache key (loudnorm_cache.py). It MUST stay byte-identical to
# the sibling engines' _LOUDNORM_FILTER (osu-std record/audio.py) or the shared
# cache key diverges and cross-engine reuse silently stops.
LOUDNORM = "loudnorm=I=-18:TP=-1.5:LRA=11"
# Format of the shared loudnorm-cache artifact: raw headerless little-endian
# float32 PCM, 48 kHz, stereo — IDENTICAL to the sibling engines so the
# `{key}.f32le` files interoperate. build_ffmpeg_cmd must tell ffmpeg the raw
# input geometry (there is no header) when it consumes such a file.
LOUDNORM_CACHE_SR = 48000
LOUDNORM_CACHE_CH = 2


def _ffmpeg_prefix() -> list[str]:
    """Prefix command to escape to host ffmpeg when we're inside a
    toolbox/distrobox container; empty list otherwise."""
    if Path("/run/host/etc/os-release").exists() and Path("/usr/bin/flatpak-spawn").exists():
        return ["/usr/bin/flatpak-spawn", "--host", "/usr/bin/ffmpeg"]
    return ["ffmpeg"]


async def probe_encoder(encoder: str, device: str | None) -> str:
    """Resolve 'auto' → preferred encoder available on this system.

    Preference order: h264_vaapi (if device provided) → libx264 →
    libopenh264 (Fedora ffmpeg-free) → libx264 as last-resort name.

    Non-'auto' values pass through unchanged.
    """
    if encoder != "auto":
        return encoder
    # Probe through the same prefix the encode will use (host ffmpeg when
    # we're in a toolbox), so we don't pick an encoder the toolbox ffmpeg
    # has but the host's doesn't (or vice versa).
    probe_cmd = _ffmpeg_prefix() + ["-hide_banner", "-encoders"]
    proc = await asyncio.create_subprocess_exec(
        *probe_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    text = out.decode(errors="ignore")
    # When no device was specified, try to auto-pick the standard VAAPI
    # render node. Host AMD GPUs always expose /dev/dri/renderD128.
    # nvenc FIRST: R3D is an NVIDIA box; the old order never tried nvenc, so a
    # dropped R3D_ENCODER env silently fell to vaapi/libx264 (5-10x slower).
    if "h264_nvenc" in text:
        return "h264_nvenc"
    if device is None and Path("/dev/dri/renderD128").exists():
        device = "/dev/dri/renderD128"
    if device is not None and "h264_vaapi" in text:
        return "h264_vaapi"
    if "libx264" in text:
        return "libx264"
    if "libopenh264" in text:
        return "libopenh264"
    return "libx264"  # let ffmpeg raise a clear error if nothing is available


def nvenc_target_bps(w: int, h: int, fps: float) -> int:
    """Resolution-scaled NVENC bitrate ladder (R3D cross-engine policy, 2026-07).

    Replaces the flat per-engine bitrate: scale a 4 Mbps 720p30 reference
    by pixel rate with a perceptual exponent (0.70 -- deliberately NOT
    linear), clamped to [2.5, 16] Mbps.  Anchors: 720p30=4.0M,
    720p60=6.5M, 1080p30=7.1M, 1080p60=11.5M, 1440p60/1080p120+=16M cap.
    Callers pair the target with maxrate=1.5x / bufsize=2x for NVENC VBR.
    Same formula in all four engines (catch/taiko/std/mania v2).
    """
    ref = 1280.0 * 720.0 * 30.0
    target = 4_000_000.0 * ((float(w) * float(h) * float(fps)) / ref) ** 0.70
    return int(min(16_000_000.0, max(2_500_000.0, target)))


def build_ffmpeg_cmd(
    *,
    encoder: str,
    encoder_device: str | None,
    resolution: tuple[int, int],
    fps: int,
    audio_path: Path | None,
    audio_rate: float,
    audio_pitch: bool = False,
    prenormalized_audio_path: Path | None = None,
    audio_lead_in_ms: int,
    video_bitrate: str,
    video_bitrate_override: int | None = None,
    audio_bitrate: str,
    output_path: Path,
    total_duration_ms: int | None = None,
    hitsound_path: Path | None = None,
    frames_fifo_path: Path | None = None,
    music_volume: float = 1.0,
    hitsound_volume: float = 1.0,
) -> list[str]:
    """Build the ffmpeg argv. Audio is optional.

    When ``frames_fifo_path`` is given, raw frames are read from that FIFO
    rather than stdin — required for the host-ffmpeg case (we route via
    flatpak-spawn, which proxies stdin through D-Bus and is far too slow
    for raw-video throughput, but a FIFO on a shared tmpfs is direct).

    ``prenormalized_audio_path`` — when set, it is a cached PCM file that has
    ALREADY had the rate/pitch change AND ``LOUDNORM`` baked in (see
    :mod:`osu_mania_renderer_v2.render.loudnorm_cache`). We feed it as the song
    input and the filtergraph SKIPS both the rate/pitch filters and loudnorm,
    keeping only the per-render lead-in/volume/fade + hitsound mix. ``None``
    (kill-switch off / cache miss failure) is the unchanged fused path where
    loudnorm runs inline every render.
    """
    w, h = resolution
    cmd: list[str] = [*_ffmpeg_prefix(), "-y", "-hide_banner", "-loglevel", "error"]

    if encoder == "h264_vaapi":
        # `-vaapi_device` initialises the device for the FILTER graph
        # (hwupload). `-hwaccel`/`-hwaccel_device` would only affect decode
        # acceleration on inputs — without `-vaapi_device`, hwupload errors
        # out with "A hardware device reference is required to upload".
        if encoder_device:
            cmd += ["-vaapi_device", encoder_device]
        cmd += ["-hwaccel", "vaapi"]

    # Video input: raw frames on stdin OR a FIFO file. FIFO is required
    # when ffmpeg is being run on the host via flatpak-spawn (stdin would
    # go through D-Bus). Same `-f rawvideo -pix_fmt rgb24 -s WxH -r FPS`
    # input args either way.
    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", str(frames_fifo_path) if frames_fifo_path is not None else "pipe:0",
    ]

    # Audio inputs (song, optionally hitsound track). We always handle them
    # through `-filter_complex` instead of `-filter:a` so a second audio
    # stream can be mixed in cleanly.
    if audio_path is not None:
        if prenormalized_audio_path is not None:
            # Cached artifact is HEADERLESS raw f32le PCM (rate/pitch + loudnorm
            # already baked in) — declare its geometry so ffmpeg can read it.
            cmd += ["-f", "f32le",
                    "-ar", str(LOUDNORM_CACHE_SR),
                    "-ac", str(LOUDNORM_CACHE_CH),
                    "-i", str(prenormalized_audio_path)]
        else:
            cmd += ["-i", str(audio_path)]  # raw source (fused path)
        song_label = "1:a"
    if audio_path is not None and hitsound_path is not None:
        cmd += ["-i", str(hitsound_path)]
        hit_label = "2:a"
    else:
        hit_label = None

    audio_out_label: str | None = None
    if audio_path is not None:
        # `_prenorm` = we were handed a cached PCM file with rate/pitch AND
        # loudnorm already applied. In that case the rate/pitch filters and the
        # inline loudnorm are OMITTED (they are baked into the input); only the
        # per-render lead-in / volume / fade remain. When `_prenorm` is False
        # the chain below is byte-for-byte the original fused pipeline.
        _prenorm = prenormalized_audio_path is not None
        song_chain: list[str] = []
        if not _prenorm and audio_rate != 1.0:
            if audio_pitch:
                # Nightcore: rate change — pitch rises with speed (stable
                # NC semantics; the resampled "nightcore" sound). Normalise
                # to 44100 FIRST: asetrate's factor is relative to the
                # stream's actual sample rate, and the old bare
                # `asetrate=44100*rate` on a 48 kHz mp3 produced
                # 66150/48000 = 1.378x instead of 1.5x — audibly flat AND
                # ~9% out of sync with the note timeline.
                song_chain.append("aresample=44100")
                song_chain.append(f"asetrate=44100*{audio_rate}")
                song_chain.append("aresample=44100")
            else:
                # DT / HT: stable plays these pitch-PRESERVING (BASS FX
                # tempo shift) — only NC pitches up. The old code ran the
                # asetrate branch for every rate mod, which made DT sound
                # like Nightcore. atempo is sample-rate-agnostic and
                # handles any factor in [0.5, 100] in one stage, which
                # covers our only rates (0.75 / 1.5).
                song_chain.append(f"atempo={audio_rate}")
        if audio_lead_in_ms > 0:
            song_chain.append(f"adelay={audio_lead_in_ms}|{audio_lead_in_ms}")
        # Apply music gain. 1.0 = no-op (filter omitted). ffmpeg accepts plain
        # multipliers.
        if music_volume != 1.0:
            song_chain.append(f"volume={music_volume:.3f}")
        # Audio fade-out for the last 600 ms so the song tucks under the results
        # overlay instead of cutting abruptly. The `-t` flag bounds the file
        # overall, so we anchor on that length.
        if total_duration_ms is not None and total_duration_ms > 700:
            fade_dur = 0.6
            fade_start = (total_duration_ms / 1000.0) - fade_dur
            song_chain.append(f"afade=t=out:st={fade_start:.3f}:d={fade_dur:.3f}")

        def _song_producer(out_label: str) -> str | None:
            """`[song]`/`[aout]` producer for the song. Fuses ``LOUDNORM`` at
            the tail ONLY when not using a pre-normalised cache file. Returns
            ``None`` when there is nothing to apply (prenorm + no per-render
            filters) so the caller can map the stream straight through."""
            parts = list(song_chain)
            if not _prenorm:
                # LOUDNORM FIX (#17): normalise the SONG ALONE (before hits are
                # mixed) so its gain never reacts to hitsound transients — the
                # song+hits mix used to duck ~4 dB under every hit peak.
                parts.append(LOUDNORM)
            if parts:
                return f"[{song_label}]{','.join(parts)}[{out_label}]"
            return None

        if hit_label is not None:
            # Song → [song]; if nothing to apply (prenorm, no per-render
            # filters) still expose a labelled [song] for amix via anull.
            song_chain_str = _song_producer("song") or f"[{song_label}]anull[song]"
            # Build hit chain: optional adelay → optional volume → label. The
            # premixed hitsound track is already in the modded timeline, so it
            # needs no rate filter — just the same lead-in delay as the song.
            hit_filters: list[str] = []
            if audio_lead_in_ms > 0:
                hit_filters.append(f"adelay={audio_lead_in_ms}|{audio_lead_in_ms}")
            if hitsound_volume != 1.0:
                hit_filters.append(f"volume={hitsound_volume:.3f}")
            hit_chain = (
                f"[{hit_label}]{','.join(hit_filters)}[hits]"
                if hit_filters
                else f"[{hit_label}]anull[hits]"
            )
            # Mix the hits ON TOP of the already-normalised song, then a gentle
            # true-peak limiter (level=disabled = clamp only, no makeup gain, no
            # re-normalisation) to catch summed peaks WITHOUT ducking the song.
            mix_chain = (
                "[song][hits]amix=inputs=2:duration=first:"
                "normalize=0:weights=1 1,"
                "alimiter=limit=0.95:level=disabled:attack=1:release=20[aout]"
            )
            filter_complex = ";".join([song_chain_str, hit_chain, mix_chain])
            cmd += ["-filter_complex", filter_complex]
            audio_out_label = "aout"
        else:
            sp = _song_producer("aout")
            if sp is None:
                # Pre-normalised audio with no per-render filters: map the
                # cached stream straight through (already loudnorm'd).
                audio_out_label = song_label
            else:
                cmd += ["-filter_complex", sp]
                audio_out_label = "aout"

    # OpenGL's framebuffer has row 0 at the BOTTOM of the viewport, but MP4
    # (and PNG, and PIL) expect row 0 at the TOP. Without vflip the entire
    # video reads upside-down. Force YUV 4:2:0 limited-range — that's what
    # 99% of consumer pipelines (browsers, Discord, OBS, x264 defaults)
    # expect. Earlier we emitted full-range yuv420p (== yuvj420p) and
    # tagged it `-color_range pc`; Discord's transcoder treated it as
    # full-range and re-encoded to limited without rescaling, washing
    # colors / shifting them green. Limited-range here means the same
    # bits land as the same Y'CbCr levels on every player.
    vf_chain = ["vflip"]
    if encoder == "h264_vaapi":
        vf_chain += ["format=nv12", "hwupload"]
    else:
        vf_chain += ["scale=in_range=full:out_range=limited", "format=yuv420p"]
    cmd += ["-vf", ",".join(vf_chain)]

    if encoder in ("h264_nvenc", "hevc_nvenc"):
        # #87 (quality-approved): default to CQ23 (constant quality) instead
        # of a flat bitrate. On osu gameplay this lands ~3-4x smaller than
        # the ladder target -> much smaller masters -> faster node->
        # coordinator upload (the real "Finalizing" cost on remote renders).
        # The resolution ladder becomes a maxrate CAP. Explicit
        # video_bitrate_override still pins an exact bitrate.
        if video_bitrate_override:
            _tgt = int(video_bitrate_override)
            cmd += ["-c:v", encoder, "-b:v", str(_tgt),
                    "-maxrate", str(int(_tgt * 1.5)), "-bufsize", str(_tgt * 2)]
        else:
            _cap = nvenc_target_bps(w, h, fps)
            cmd += ["-c:v", encoder, "-rc", "vbr", "-cq", "23", "-b:v", "0",
                    "-maxrate", str(_cap), "-bufsize", str(_cap * 2)]
    else:
        # #87 default-quality: crf 23 (x264 family) / CQP qp 23 (vaapi).
        # Explicit override pins the bitrate. VAAPI CQP is driver-dependent
        # -> Aussie to validate on AMD before the fleet bundle.
        if video_bitrate_override:
            cmd += ["-c:v", encoder, "-b:v", str(video_bitrate_override)]
        elif encoder in ("libx264", "libx265", "libopenh264"):
            cmd += ["-c:v", encoder, "-crf", "23"]
        elif encoder == "h264_vaapi":
            cmd += ["-c:v", encoder, "-rc_mode", "CQP", "-qp", "23"]
        else:
            cmd += ["-c:v", encoder, "-b:v", video_bitrate]
        if encoder in ("libx264", "libx265", "libopenh264"):
            # CPU-encode thread cap (R3D host-governance, 2026-09): leave >=2
            # logical cores free for the machine's owner. Uncapped, libx264
            # spawns threads on EVERY core at normal priority and can freeze a
            # contributor's desktop (the rel/Stella "semi-crash"). Harmless on
            # dedicated render boxes: CPU encode is only the no-HW fallback.
            # Same cap in all four engines (catch/taiko/std/mania v2).
            cmd += ["-threads", str(max(2, (os.cpu_count() or 4) - 2))]
    # Pin BT.709 + limited-range tags on the SPS so downstream players
    # don't have to guess. (Limited range matches the scale=out_range
    # conversion above; both must agree or you get a brightness shift.)
    if encoder in ("h264_nvenc", "hevc_nvenc", "libx264", "libx265",
                   "libopenh264"):
        cmd += [
            "-color_range", "tv",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
        ]

    # +faststart moves the MP4 moov atom to the file's beginning so HTML5
    # players (incl. Discord's inline embed) can start playback as soon as a
    # tiny prefix has downloaded, instead of waiting on the entire file.
    cmd += ["-movflags", "+faststart"]

    if audio_path is not None:
        cmd += ["-c:a", "aac", "-ar", "48000", "-b:a", audio_bitrate, "-map", "0:v"]
        # `audio_out_label` is either a stream selector like "1:a" (use bare)
        # or a filter-complex output label like "aout" (use [aout]).
        if audio_out_label is None:
            pass  # no audio mapping; fall through to video-only
        elif ":" in audio_out_label:
            cmd += ["-map", audio_out_label]
        else:
            cmd += ["-map", f"[{audio_out_label}]"]
    else:
        cmd += ["-map", "0:v"]

    # We want the output to be exactly the video duration (gameplay + the
    # post-game results card). With `-shortest` ffmpeg cuts to whichever
    # stream ends first, which would chop the results overlay off when the
    # song's audio runs out a few seconds before the video does — and once
    # ffmpeg closes stdin, the renderer hits BrokenPipeError on the next
    # frame write. `-t` bounds the output by an explicit duration instead.
    if total_duration_ms is not None:
        cmd += ["-t", f"{total_duration_ms / 1000:.3f}"]
    cmd += [str(output_path)]
    return cmd


class FfmpegPipe:
    """Owns a running ffmpeg process; bytes written via write_frame end up
    encoded. Two ingestion modes:

      * stdin (default) — `cmd` uses `-i pipe:0`, frames flow through the
        Python child stdin. Best for in-toolbox / single-process renders.
      * FIFO — when `fifo_path` is given, the FIFO file is mkfifo'd, the
        cmd points its video input at the FIFO path, ffmpeg opens it for
        read, and Python writes raw bytes to the same path. Required when
        ffmpeg runs on the host via flatpak-spawn (D-Bus stdin proxying
        is way too slow for raw-video throughput).
    """

    def __init__(self, cmd: list[str], *, fifo_path: Path | None = None) -> None:
        self.cmd = cmd
        self.fifo_path = fifo_path
        self.proc: asyncio.subprocess.Process | None = None
        self._stderr_log: bytes = b""
        self._fifo_fd: int | None = None
        self._stdin_fd: int | None = None
        # Dedicated writer thread + bounded queue (see write_frame). A
        # depth-2 queue decouples the render loop from ffmpeg's per-frame
        # read/filter cadence without unbounded buffering; the thread is
        # plain `queue`/`os.write` — no asyncio in the hot write path.
        self._q: queue.Queue | None = None
        self._writer: threading.Thread | None = None
        self._werr: BaseException | None = None

    async def start(self) -> None:
        if self.fifo_path is not None:
            # Create the FIFO before spawning ffmpeg so the consumer can
            # open it for read. Open our write side as O_RDWR so the call
            # returns immediately without waiting for a reader.
            try:
                os.mkfifo(str(self.fifo_path))
            except FileExistsError:
                pass
            self.proc = await asyncio.create_subprocess_exec(
                *self.cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            self._fifo_fd = os.open(str(self.fifo_path), os.O_RDWR)
            _grow_pipe(self._fifo_fd)
        else:
            # Hand ffmpeg a plain os.pipe() instead of asyncio's stdin
            # transport. The asyncio writer chops each 6 MB rawvideo frame
            # into ~95 pipe-capacity (64 KiB) chunks, each with an epoll
            # wakeup + _write_ready dispatch — measured at ~60% of the
            # whole render loop. A raw fd grown to pipe-max-size (1 MiB)
            # takes the same bytes in ~6 blocking writes issued from a
            # worker thread, no event-loop churn. Byte stream is identical.
            rfd, wfd = os.pipe()
            try:
                self.proc = await asyncio.create_subprocess_exec(
                    *self.cmd,
                    stdin=rfd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
            finally:
                # Child owns its dup of the read end; drop ours either way.
                os.close(rfd)
            self._stdin_fd = wfd
            _grow_pipe(wfd)

    def _ensure_writer(self, fd: int) -> queue.Queue:
        if self._q is None:
            self._q = queue.Queue(maxsize=2)
            self._writer = threading.Thread(
                target=self._writer_loop, args=(fd,),
                name="ffmpeg-frame-writer", daemon=True,
            )
            self._writer.start()
        return self._q

    def _writer_loop(self, fd: int) -> None:
        """Drain the frame queue into the pipe. On any write error, record
        it (surfaced loudly by the NEXT write_frame/close) and keep
        consuming so the producer can never block on a dead pipe."""
        q = self._q
        assert q is not None
        while True:
            item = q.get()
            if item is None:
                return
            # A frame is either a plain bytes-like or a MappedFrame lease
            # (readback.py): bytes live in mapped GL memory as `.mv`, and
            # `.done` MUST be set once we're finished with them — even on
            # error — or the GL thread waits forever to reuse the buffer.
            data = getattr(item, "mv", item)
            try:
                if self._werr is None:
                    _blocking_write_all(fd, data)
            except BaseException as e:  # noqa: BLE001 — must not kill thread
                self._werr = e
            finally:
                done = getattr(item, "done", None)
                if done is not None:
                    done.set()

    async def write_frame(self, data: bytes) -> None:
        if self.proc is None:
            raise EncoderError("ffmpeg not started")
        if self._werr is not None:
            raise EncoderError(
                f"ffmpeg pipe write failed: {self._werr!r}") from self._werr
        fd = self._fifo_fd if self._fifo_fd is not None else self._stdin_fd
        if fd is None:
            raise EncoderError("ffmpeg stdin closed")
        # Hand the frame to the writer thread. The bounded queue gives
        # depth-2 pipelining — the GL/Python side renders the next frame
        # while the thread pushes this one — and `put` blocking when full
        # is exactly the old stdin backpressure. Frames are immutable bytes
        # or caller-rotated buffers (FrameReader pool > queue depth + 2),
        # so an in-flight buffer can't be mutated underneath the writer.
        self._ensure_writer(fd).put(data)

    def _join_writer(self) -> None:
        if self._writer is not None:
            assert self._q is not None
            self._q.put(None)
            self._writer.join()
            self._writer = None
            self._q = None

    async def close(self, output_path: Path) -> None:
        if self.proc is None:
            return
        self._join_writer()
        if self._werr is not None:
            raise EncoderError(
                f"ffmpeg pipe write failed: {self._werr!r}") from self._werr
        if self._fifo_fd is not None:
            os.close(self._fifo_fd)
            self._fifo_fd = None
            if self.fifo_path is not None:
                try:
                    os.unlink(str(self.fifo_path))
                except OSError:
                    pass
        elif self._stdin_fd is not None:
            os.close(self._stdin_fd)
            self._stdin_fd = None
        _, stderr = await self.proc.communicate()
        self._stderr_log = stderr or b""
        if self.proc.returncode != 0:
            raise EncoderError(
                f"ffmpeg exit code {self.proc.returncode}: "
                f"{self._stderr_log.decode(errors='ignore')[-1024:]}"
            )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise EncoderError(f"output MP4 missing or empty: {output_path}")


def _grow_pipe(fd: int) -> None:
    """Best-effort: grow the kernel pipe buffer to /proc/sys/fs/pipe-max-size
    (1 MiB default) so a 6 MB raw frame moves in ~6 write() calls instead of
    ~95 at the 64 KiB default. Purely a syscall-count optimisation — the byte
    stream is unchanged — so any failure (EPERM, non-Linux) is ignored."""
    try:
        import fcntl
        f_setpipe_sz = getattr(fcntl, "F_SETPIPE_SZ", 1031)  # Linux
        try:
            max_size = int(
                Path("/proc/sys/fs/pipe-max-size").read_text().strip())
        except (OSError, ValueError):
            max_size = 1 << 20
        fcntl.fcntl(fd, f_setpipe_sz, max_size)
    except OSError:
        pass


def _blocking_write_all(fd: int, data) -> None:
    """Loop os.write until every byte of `data` is committed to `fd`. A
    partial write can happen on pipes/FIFOs once the kernel buffer fills."""
    view = memoryview(data)
    offset = 0
    n = len(view)
    while offset < n:
        offset += os.write(fd, view[offset:])
