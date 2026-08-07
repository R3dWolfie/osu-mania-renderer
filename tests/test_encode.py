from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from osu_mania_renderer_v2.render.encode import build_ffmpeg_cmd, probe_encoder


def test_build_cmd_vaapi():
    cmd = build_ffmpeg_cmd(
        encoder="h264_vaapi",
        encoder_device="/dev/dri/renderD128",
        resolution=(1920, 1080), fps=60,
        audio_path=Path("/x/audio.mp3"), audio_rate=1.0, audio_lead_in_ms=0,
        video_bitrate="5M", audio_bitrate="192k",
        output_path=Path("/x/out.mp4"),
    )
    joined = " ".join(cmd)
    assert "-hwaccel vaapi" in joined
    assert "-c:v h264_vaapi" in joined
    assert "-r 60" in joined
    assert "-s 1920x1080" in joined
    assert "out.mp4" in joined
    # vflip MUST come before VAAPI's format/hwupload chain.
    assert "vflip,format=nv12,hwupload" in joined


def test_build_cmd_software():
    cmd = build_ffmpeg_cmd(
        encoder="libx264", encoder_device=None,
        resolution=(1280, 720), fps=30,
        audio_path=Path("/x/audio.mp3"), audio_rate=1.5, audio_lead_in_ms=500,
        video_bitrate="3M", audio_bitrate="128k",
        output_path=Path("/x/out.mp4"),
    )
    joined = " ".join(cmd)
    assert "-c:v libx264" in joined
    assert "-hwaccel" not in joined
    # Speed mod expressed in audio filter.
    assert "asetrate=44100*1.5" in joined
    # Software path still needs the Y flip.
    assert "vflip" in joined


def test_build_cmd_no_audio():
    cmd = build_ffmpeg_cmd(
        encoder="libx264", encoder_device=None,
        resolution=(1280, 720), fps=30,
        audio_path=None, audio_rate=1.0, audio_lead_in_ms=0,
        video_bitrate="3M", audio_bitrate="128k",
        output_path=Path("/x/silent.mp4"),
    )
    joined = " ".join(cmd)
    assert "-c:a" not in joined
    assert "silent.mp4" in joined


async def test_probe_encoder_auto_prefers_vaapi(tmp_path):
    # Mock subprocess that reports h264_vaapi is available.
    async def fake_create(*args, **kw):
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b" V..... h264_vaapi  H.264 ...\n", b""))
        proc.returncode = 0
        return proc
    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
        chosen = await probe_encoder("auto", "/dev/dri/renderD128")
    assert chosen == "h264_vaapi"


async def test_probe_encoder_falls_back_to_libx264():
    async def fake_create(*args, **kw):
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b" V..... libx264  H.264 ...\n", b""))
        proc.returncode = 0
        return proc
    with patch("asyncio.create_subprocess_exec", side_effect=fake_create):
        chosen = await probe_encoder("auto", None)
    assert chosen == "libx264"


async def test_probe_encoder_explicit_passes_through():
    chosen = await probe_encoder("libx264", None)
    assert chosen == "libx264"
