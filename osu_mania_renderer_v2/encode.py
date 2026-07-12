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
from pathlib import Path

from osu_mania_renderer_v2.errors import EncoderError


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


def build_ffmpeg_cmd(
    *,
    encoder: str,
    encoder_device: str | None,
    resolution: tuple[int, int],
    fps: int,
    audio_path: Path | None,
    audio_rate: float,
    audio_lead_in_ms: int,
    video_bitrate: str,
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
        cmd += ["-i", str(audio_path)]
        song_label = "1:a"
    if audio_path is not None and hitsound_path is not None:
        cmd += ["-i", str(hitsound_path)]
        hit_label = "2:a"
    else:
        hit_label = None

    audio_out_label: str | None = None
    if audio_path is not None:
        song_chain: list[str] = []
        if audio_rate != 1.0:
            # asetrate changes both speed AND pitch (matches osu! DT/HT).
            song_chain.append(f"asetrate=44100*{audio_rate}")
            song_chain.append("aresample=44100")
        if audio_lead_in_ms > 0:
            song_chain.append(f"adelay={audio_lead_in_ms}|{audio_lead_in_ms}")
        # Apply music gain. 1.0 = no-op; ffmpeg accepts plain multipliers.
        # We always emit the filter even at 1.0 so the chain length is
        # deterministic — easier to follow when diffing logs.
        if music_volume != 1.0:
            song_chain.append(f"volume={music_volume:.3f}")
        # NB: `loudnorm=I=-16:LRA=11:TP=-1.5` would normalise across maps
        # but one-pass loudnorm slows ffmpeg by ~6× and pushed our renders
        # past the 600 s timeout, leaving truncated MP4s with no moov atom.
        # Drop it for now; if cross-map consistency becomes an issue we'll
        # use a quick ffprobe `volumedetect` pre-pass to set a fixed
        # `volume=N dB` adjustment, which is cheap.
        # Audio fade-out for the last 600 ms of the gameplay so the song
        # tucks under the results overlay instead of cutting abruptly. The
        # `-t` flag bounds the file overall, so we anchor on that length.
        if total_duration_ms is not None and total_duration_ms > 700:
            fade_dur = 0.6
            fade_start = (total_duration_ms / 1000.0) - fade_dur
            song_chain.append(f"afade=t=out:st={fade_start:.3f}:d={fade_dur:.3f}")

        if hit_label is not None:
            # Pre-mixed hitsound track is already in the modded timeline (the
            # renderer applied DT/HT to note positions before generating it),
            # so it doesn't need asetrate. Just delay it by the same lead-in
            # the song uses so the two stay in sync.
            # LOUDNORM FIX (2026-07-12, #17): loudnorm the SONG ALONE, before
            # mixing, so its gain never reacts to the hitsound transients.
            # Loudnorm on the song+HITS mix (the old norm_chain) let its
            # limiter/gain duck the whole mix -- song included -- ~4 dB under
            # every hit peak ("song ducks to hitsounds"). Song target -10 LUFS
            # (== the no-hit branches); hits are transient (~+0.2 dB to the
            # integrated), so the mix still lands ~-10 with them on top.
            song_chain_str = (
                f"[{song_label}]{','.join(song_chain)},"
                "loudnorm=I=-10:TP=-1.5:LRA=11[song]"
                if song_chain
                else f"[{song_label}]loudnorm=I=-10:TP=-1.5:LRA=11[song]"
            )
            # Build hit chain: optional adelay → optional volume → label.
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
            # No loudnorm on the mix -- that was what ducked the song under hits.
            mix_chain = (
                "[song][hits]amix=inputs=2:duration=first:"
                "normalize=0:weights=1 1,"
                "alimiter=limit=0.95:level=disabled:attack=1:release=20[aout]"
            )
            filter_complex = ";".join([song_chain_str, hit_chain, mix_chain])
            cmd += ["-filter_complex", filter_complex]
            audio_out_label = "aout"
        elif song_chain:
            # LOUDNORM CONSOLIDATION (perf 2026-07-12): append the single
            # surviving loudnorm (-10 LUFS) after song_chain.
            filter_complex = (
                f"[{song_label}]{','.join(song_chain)},loudnorm=I=-10:TP=-1.5:LRA=11[aout]"
            )
            cmd += ["-filter_complex", filter_complex]
            audio_out_label = "aout"
        else:
            # LOUDNORM CONSOLIDATION (perf 2026-07-12): a bare song with no
            # song_chain and no hitsounds still needs the single surviving
            # loudnorm (-10 LUFS), so give this branch its own filtergraph.
            filter_complex = (
                f"[{song_label}]loudnorm=I=-10:TP=-1.5:LRA=11[aout]"
            )
            cmd += ["-filter_complex", filter_complex]
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

    cmd += ["-c:v", encoder, "-b:v", video_bitrate]
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
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-map", "0:v"]
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
        else:
            self.proc = await asyncio.create_subprocess_exec(
                *self.cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

    async def write_frame(self, data: bytes) -> None:
        if self.proc is None:
            raise EncoderError("ffmpeg not started")
        if self._fifo_fd is not None:
            # Off-loop write — writing 165 MB/s of raw frames blocks until
            # ffmpeg drains, so we must not stall the asyncio loop.
            await asyncio.to_thread(_blocking_write_all, self._fifo_fd, data)
        else:
            assert self.proc.stdin is not None
            self.proc.stdin.write(data)
            await self.proc.stdin.drain()

    async def close(self, output_path: Path) -> None:
        if self.proc is None:
            return
        if self._fifo_fd is not None:
            os.close(self._fifo_fd)
            self._fifo_fd = None
            if self.fifo_path is not None:
                try:
                    os.unlink(str(self.fifo_path))
                except OSError:
                    pass
        elif self.proc.stdin is not None:
            self.proc.stdin.close()
        _, stderr = await self.proc.communicate()
        self._stderr_log = stderr or b""
        if self.proc.returncode != 0:
            raise EncoderError(
                f"ffmpeg exit code {self.proc.returncode}: "
                f"{self._stderr_log.decode(errors='ignore')[-1024:]}"
            )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise EncoderError(f"output MP4 missing or empty: {output_path}")


def _blocking_write_all(fd: int, data: bytes) -> None:
    """Loop os.write until every byte of `data` is committed to `fd`. A
    partial write can happen on FIFOs once the kernel pipe buffer fills."""
    offset = 0
    n = len(data)
    while offset < n:
        offset += os.write(fd, data[offset:])
