"""The inline preview on the Mac's media engine (VideoToolbox).

Ported from std (osu_std_renderer/record/encode.py, std PR #11), where the
settings were chosen and the numbers below were measured, by way of taiko and
catch (bundle 0.1.33); the code is the same so the engines cannot drift, and
they share the node-wide markers (the probe result and the day-long
switch-off live beside the other r3d caches)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _envflag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("", "0", "false", "no", "off")


# With no switch set the hardware preview is used on a Mac (where the CPU is
# the master's encoder and the media engine is otherwise idle) and nowhere
# else. R3D_PREVIEW_VT=0 turns it off, =1 asks for it on any platform; either
# way the probe below has the last word.
_MAC_DEFAULT = sys.platform == "darwin"

# ---- inline preview on the Mac's media engine --------------------------------
# The preview is a 720p30 side output of the SAME ffmpeg process as the master.
# On a Mac the CPU is the encoder, so from two renders at once the preview's
# own x264 competes with the masters': on an M1 Max, encoding the preview with
# VideoToolbox instead (the media engine is otherwise idle, the master stays on
# x264) measured +16 to +23% aggregate throughput at 2-4 concurrent std renders
# at 720p and 1080p, +8% for one 1080p render, and nothing for one 720p render.
# The master is byte-for-byte the same file either way.
#
#   R3D_PREVIEW_VT=0   the CPU preview, as before
#   R3D_PREVIEW_VT=1   ask for the media engine on any platform
#   unset              the media engine on a Mac (perf.FAST_DEFAULT, so
#                      R3D_STD_STOCK=1 also means the CPU preview)
#
# Asking is not getting: the local ffmpeg must first open a hardware session
# (`_vt_probe`); where it cannot, the preview stays on libx264. One failed
# output kills the whole ffmpeg process and with it the render, so when an
# ffmpeg that carried a hardware preview dies naming videotoolbox, the media
# engine is switched off for a day (`note_preview_failure`) and the next render
# on this node is on the CPU preview without anyone touching a setting.
#
# Settings (chosen on lossless captures of all four modes, see the PR): the
# same target bitrate as the x264 preview, in the encoder's CONSTANT-RATE mode
# with B-frames, behind a half-second LEAD-IN.
#
# Why each: in its default mode, and worse under a peak cap, this encoder
# makes a small first keyframe and then starves the rest of the first second
# (worst frame VMAF 28 with x264's 1.25x cap). Constant-rate mode with B-frames
# cures that at 1400 kbit/s, but a long replay gets a lower preview bitrate
# from the size budget (507 kbit/s at five minutes) and there the first
# keyframe is too small again: first half-second VMAF 42 against 66 for the
# rest. The lead-in is the cure that holds at every bitrate: the first frame
# is repeated for _VT_LEAD_S seconds in front of the video, so the rate
# control has banked about what x264 spends on its first keyframe by the time
# the real video starts; a keyframe is forced on the first real frame, and two
# bitstream filters drop the lead-in's packets and shift the rest back, so the
# file starts on that keyframe at time 0 with exactly the frames it had before
# (42 -> 73 on that replay, with nothing taken from the seconds after).
# Constant-rate mode also makes the file size exact, which is what
# preview_video_bps budgets for. Frames are scaled on the CPU exactly as
# before, so the encoder is the only thing that changes.
_VT_RATE = 1.0
_VT_BFRAMES = 2
_VT_LEAD_S = 0.5


def _vt_lead_frames(pfps: int) -> int:
    return max(1, int(round(pfps * _VT_LEAD_S)))


def vt_lead_in_filter(pfps: int) -> str:
    """Tail of the preview branch's filter chain for the hardware preview: the
    first frame repeated in front (see the block comment above)."""
    return f"tpad=start={_vt_lead_frames(pfps)}:start_mode=clone"


def _vt_codec_args(bps: int, pfps: int = 30) -> "list[str]":
    """The hardware preview's codec arguments. The probe uses these too, so a
    machine that cannot do one of them (constant-rate mode needs macOS 13; the
    packet filters need ffmpeg 5.1) fails the probe and keeps the CPU preview
    instead of failing a render."""
    n = _vt_lead_frames(pfps)
    cut = n / float(pfps)
    return ["-c:v", "h264_videotoolbox", "-allow_sw", "0", "-realtime", "0",
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-b:v", str(int(bps)),
            "-constant_bit_rate", "1", "-g", str(int(pfps)),
            "-bf", str(_VT_BFRAMES),
            # the cut must land on a keyframe whatever cadence the encoder
            # keeps (with B-frames it is 29 frames, not 30): force one on the
            # first real frame, by frame NUMBER (a time would round)
            "-force_key_frames", f"expr:eq(n,{n})",
            "-bsf:v", (f"noise=drop=lt(pts*tb\\,{cut - 0.001:.6f}),"
                       f"setts=pts=PTS-{n}/{int(pfps)}/TB:dts=DTS-{n}/{int(pfps)}/TB")]


_VT_OFF_FOR_S = 24 * 3600
_vt_decision: "bool | None" = None


def _r3d_cache_dir() -> str:
    import tempfile
    return (os.path.expanduser("~/Library/Caches/r3d") if sys.platform == "darwin"
            else os.path.join(tempfile.gettempdir(), "r3d-cache"))


def _vt_probe(ignore_off: bool = False) -> bool:
    """Can the ffmpeg on THIS machine open a VideoToolbox H.264 session right
    now? A half-second gray clip through it; a yes is remembered per ffmpeg
    binary (path, size, mtime), a no is asked again next time. Any trouble
    counts as no: a probe must never fail a render."""
    import hashlib
    import time
    try:
        exe = shutil.which("ffmpeg")
        if not exe:
            return False
        base = _r3d_cache_dir()
        if not ignore_off:
            try:
                with open(os.path.join(base, "vt-preview-off")) as fh:
                    if time.time() < float(fh.read().strip() or 0):
                        return False
            except (OSError, ValueError):
                pass
        st = os.stat(os.path.realpath(exe))
        key = hashlib.sha1(f"{os.path.realpath(exe)}|{st.st_size}|{st.st_mtime_ns}|v3"
                           .encode()).hexdigest()[:16]
        mark = os.path.join(base, f"vt-preview-probe-{key}")
        if os.path.exists(mark):
            return True
        p = subprocess.run(
            [exe, "-v", "error", "-f", "lavfi", "-i",
             "color=c=gray:s=320x180:r=30:d=1", "-vf", vt_lead_in_filter(30)]
            + _vt_codec_args(300_000) + ["-f", "null", "-"],
            capture_output=True, timeout=20)
        if p.returncode != 0:
            return False
        try:
            os.makedirs(base, exist_ok=True)
            with open(mark, "w") as fh:
                fh.write("1")
        except OSError:
            pass
        return True
    except Exception:  # noqa: BLE001
        return False


def preview_on_media_engine() -> bool:
    """Should this render's inline preview be encoded by VideoToolbox? Decided
    once per process (see the block comment above for the rule)."""
    global _vt_decision
    if _vt_decision is None:
        asked = os.environ.get("R3D_PREVIEW_VT") is not None
        want = _envflag("R3D_PREVIEW_VT") if asked else _MAC_DEFAULT
        # an explicit =1 overrides the day-long switch-off, never the probe
        _vt_decision = bool(want) and _vt_probe(ignore_off=asked)
    return _vt_decision


def note_preview_failure(cmd: "list[str]", err: bytes) -> bool:
    """ffmpeg exited non-zero. If it carried a hardware preview and its last
    words name videotoolbox, switch the media engine off for a day on this
    node. Returns whether it did."""
    import time
    try:
        if "h264_videotoolbox" not in cmd:
            return False
        low = (err or b"").lower()
        if b"videotoolbox" not in low and b"vtenc" not in low:
            return False
        base = _r3d_cache_dir()
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "vt-preview-off"), "w") as fh:
            fh.write(str(int(time.time()) + _VT_OFF_FOR_S))
        return True
    except Exception:  # noqa: BLE001
        return False


def hw_video_args(vbps: int, pfps: int = 30) -> "list[str]":
    """Video codec arguments of the inline preview on the media engine."""
    return _vt_codec_args(vbps * _VT_RATE, pfps)
