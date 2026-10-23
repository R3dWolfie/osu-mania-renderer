"""AV1 master migration (Leon's 2026-10-23 design): backend-specific encoder
args, the real hardware-AV1 probe (ffprobe codec_name=av1), and fail-closed
auto selection with NO H.264/CPU fallback. No ffmpeg/GPU required — the probe
subprocess and ffprobe are both mocked, exactly like test_encoder_runtime.py."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from osu_mania_renderer_v2.errors import EncoderError
from osu_mania_renderer_v2.render import encode

# ---- build_ffmpeg_cmd: backend-specific AV1 master args ---------------------

def _build(encoder, **kw):
    base = dict(
        encoder=encoder, encoder_device="/dev/dri/renderD128",
        resolution=(1920, 1080), fps=60,
        audio_path=Path("/x/audio.mp3"), audio_rate=1.0, audio_lead_in_ms=0,
        video_bitrate="5M", audio_bitrate="192k", output_path=Path("/x/out.mp4"),
        total_duration_ms=60000,
    )
    base.update(kw)
    return encode.build_ffmpeg_cmd(**base)


def test_av1_vaapi_uses_cqp_global_quality_not_qp():
    joined = " ".join(_build("av1_vaapi"))
    assert "-c:v av1_vaapi" in joined
    # VAAPI's -qp is ignored by the AMD driver; CQP + global_quality is required.
    assert "-rc_mode CQP" in joined
    assert "-global_quality 90" in joined
    assert "-qp" not in joined
    assert "-profile:v main" in joined
    # 8-bit 4:2:0 on a GPU surface, vflip before the upload, faststart, GOP.
    assert "vflip,format=nv12,hwupload" in joined
    assert "-movflags +faststart" in joined
    assert "-g 120" in joined
    # VAAPI device is initialised for the hwupload filtergraph.
    assert "-vaapi_device /dev/dri/renderD128" in joined


def test_av1_vaapi_global_quality_env_override(monkeypatch):
    monkeypatch.setenv("R3D_AV1_VAAPI_GQ", "54")
    joined = " ".join(_build("av1_vaapi"))
    assert "-global_quality 54" in joined


def test_av1_nvenc_is_bitrate_vbr_yuv420p_and_tagged():
    joined = " ".join(_build("av1_nvenc", encoder_device=None))
    assert "-c:v av1_nvenc" in joined
    assert "-b:v " in joined and "-maxrate " in joined and "-bufsize " in joined
    # non-VAAPI AV1 takes the range-converted 8-bit 4:2:0 software branch...
    assert "scale=in_range=full:out_range=limited,format=yuv420p" in joined
    assert "hwupload" not in joined
    # ...and is colour-tagged like the software H.264 path.
    assert "-colorspace bt709" in joined
    assert "-g 120" in joined


def test_av1_amf_and_qsv_build_without_error():
    for enc in ("av1_amf", "av1_qsv"):
        joined = " ".join(_build(enc, encoder_device=None))
        assert f"-c:v {enc}" in joined
        assert "-b:v " in joined


def test_av1_bps_factor_env_scales_target(monkeypatch):
    monkeypatch.setenv("R3D_AV1_BPS_FACTOR", "1.0")
    full = " ".join(_build("av1_nvenc", encoder_device=None))
    monkeypatch.setenv("R3D_AV1_BPS_FACTOR", "0.25")
    quarter = " ".join(_build("av1_nvenc", encoder_device=None))

    def _bv(j):
        parts = j.split()
        return int(parts[parts.index("-b:v") + 1])
    assert _bv(quarter) < _bv(full)


def test_av1_codec_args_honor_bitrate_override():
    args = encode._av1_codec_args("av1_nvenc", 1920, 1080, 60, 3_000_000)
    assert args[args.index("-b:v") + 1] == "3000000"


def test_h264_master_unchanged_by_av1_support():
    # The existing H.264 VAAPI master path must be byte-for-byte as before.
    joined = " ".join(_build("h264_vaapi"))
    assert "-c:v h264_vaapi" in joined
    assert "vflip,format=nv12,hwupload" in joined
    assert "av1" not in joined


def test_av1_master_keeps_h264_inline_preview_and_compact(monkeypatch):
    # AV1 master + H.264 Discord inline preview/compact from ONE process.
    cmd = _build(
        "av1_vaapi",
        preview_path=Path("/x/out.embed.mp4"),
        compact_path=Path("/x/out.embed-sm.mp4"),
    )
    joined = " ".join(cmd)
    assert "-c:v av1_vaapi" in joined          # master is AV1
    assert "-rc_mode CQP" in joined
    # both small Discord copies stay libx264 (Discord cannot play inline AV1)
    assert joined.count("-c:v libx264") == 2
    assert "out.embed.mp4" in joined and "out.embed-sm.mp4" in joined
    # the master branch still uploads nv12 surfaces; preview/compact are CPU.
    assert "[vm0]format=nv12,hwupload[vm]" in joined


# ---- probe: real AV1 proof + fail-closed auto -------------------------------

@pytest.fixture
def av1_probes(monkeypatch):
    """Mock -encoders listing + per-encoder probe + ffprobe codec_name."""
    calls = []
    listed = ["av1_nvenc", "av1_vaapi", "av1_qsv", "h264_vaapi", "libx264",
              "libsvtav1", "libaom-av1"]
    # name -> (encode_exit, stderr, ffprobe_codec_name)
    results = {}

    async def spawn(*command, **options):
        calls.append((command, options))
        if "-encoders" in command:
            out = "\n".join(f" V..... {n} enc" for n in listed).encode()
            return SimpleNamespace(returncode=0,
                                   communicate=AsyncMock(return_value=(out, b"")))
        if command[0].endswith("ffprobe") or "ffprobe" in command[0]:
            # the output file path is the last arg
            name = spawn.last_probe_encoder
            codec = results.get(name, (0, b"", "av1"))[2]
            data = (codec or "").encode()
            return SimpleNamespace(returncode=0,
                                   communicate=AsyncMock(return_value=(data, b"")))
        # an encode probe: -c:v <name>
        name = command[command.index("-c:v") + 1]
        spawn.last_probe_encoder = name
        code, stderr, _ = results.get(name, (0, b"", "av1"))
        return SimpleNamespace(returncode=code,
                               communicate=AsyncMock(return_value=(b"", stderr)))

    spawn.last_probe_encoder = None
    monkeypatch.setattr(encode, "_ffmpeg_prefix", lambda: ["host-ffmpeg"])
    monkeypatch.setattr(encode, "_ffprobe_prefix", lambda: ["host-ffprobe"])
    monkeypatch.setattr(encode.asyncio, "create_subprocess_exec", spawn)
    return calls, results, listed


def _attempted(calls):
    return [c[c.index("-c:v") + 1] for c, _ in calls if "-c:v" in c]


async def test_av1_auto_selects_first_working_hw_av1(av1_probes):
    # av1_nvenc compiled in but no CUDA -> next is av1_vaapi, which proves AV1.
    calls, results, _ = av1_probes
    results["av1_nvenc"] = (1, b"Cannot load libcuda.so.1", None)
    chosen = await encode.probe_encoder("av1", "/dev/dri/renderD128")
    assert chosen == "av1_vaapi"
    assert _attempted(calls) == ["av1_nvenc", "av1_vaapi"]


async def test_av1_probe_rejects_non_av1_output(av1_probes):
    # Encoder "starts" (exit 0) but ffprobe says the stream is NOT av1:
    # compiled-in support is insufficient, so it must be rejected.
    calls, results, _ = av1_probes
    results["av1_nvenc"] = (0, b"", "hevc")     # lies / wrong codec
    results["av1_vaapi"] = (0, b"", "av1")
    chosen = await encode.probe_encoder("av1", "/dev/dri/renderD128")
    assert chosen == "av1_vaapi"


async def test_av1_auto_fails_closed_without_hw(av1_probes):
    # Every hardware AV1 encoder fails; there is a compiled libsvtav1/libaom,
    # but AV1 masters must NOT fall back to it. Expect a clear EncoderError.
    calls, results, _ = av1_probes
    for n in ("av1_nvenc", "av1_vaapi", "av1_qsv"):
        results[n] = (1, f"no {n}".encode(), None)
    with pytest.raises(EncoderError, match="do not fall back to H.264/CPU"):
        await encode.probe_encoder("av1", "/dev/dri/renderD128")
    # libsvtav1 / libaom-av1 were never even probed.
    assert "libsvtav1" not in _attempted(calls)
    assert "libaom-av1" not in _attempted(calls)


async def test_av1_explicit_passes_through_without_probe(av1_probes):
    calls, _, _ = av1_probes
    for enc in ("av1_vaapi", "av1_nvenc", "av1_amf", "av1_qsv"):
        assert await encode.probe_encoder(enc, None) == enc
    assert not calls


async def test_plain_auto_still_picks_h264_not_av1(av1_probes):
    # "auto" is unchanged: it must never return an AV1 encoder.
    calls, _, listed = av1_probes
    listed[:] = ["h264_nvenc", "h264_vaapi", "libx264"]
    chosen = await encode.probe_encoder("auto", "/dev/dri/renderD128")
    assert chosen == "h264_nvenc"


async def test_av1_probe_command_shape_and_ffprobe(av1_probes):
    # The av1_vaapi probe must encode to a real file through the production
    # recipe and ffprobe it (codec_name), not use `-f null`.
    calls, results, _ = av1_probes
    results["av1_nvenc"] = (1, b"no cuda", None)
    await encode.probe_encoder("av1", "/dev/dri/renderD128")
    vaapi_encode = next(c for c, _ in calls
                        if "-c:v" in c and c[c.index("-c:v") + 1] == "av1_vaapi")
    assert "-rc_mode" in vaapi_encode and "CQP" in vaapi_encode
    assert vaapi_encode[vaapi_encode.index("-vaapi_device") + 1] == "/dev/dri/renderD128"
    assert "format=nv12,hwupload" in vaapi_encode
    # written to a real .mp4 (so ffprobe can read codec_name), NOT `-f null`.
    assert "null" not in vaapi_encode
    assert vaapi_encode[-2] == "-y" and vaapi_encode[-1].endswith(".mp4")
    # an ffprobe call was made
    assert any(c[0] == "host-ffprobe" for c, _ in calls)


async def test_av1_encoder_usable_timeout_is_bounded(monkeypatch):
    # A stalled AV1 encode must time out, be killed/reaped, and be reported.
    process = SimpleNamespace(returncode=None, killed=False)

    async def communicate(_=None):
        if not process.killed:
            raise TimeoutError
        process.returncode = -9
        return b"", b"driver stalled"

    process.communicate = AsyncMock(side_effect=communicate)
    process.kill = lambda: setattr(process, "killed", True)
    monkeypatch.setattr(encode, "_ffmpeg_prefix", lambda: ["host-ffmpeg"])
    monkeypatch.setattr(encode.asyncio, "create_subprocess_exec",
                        AsyncMock(return_value=process))
    usable, reason = await encode._av1_encoder_usable("av1_vaapi", "/dev/dri/renderD128")
    assert not usable and "timed out" in reason
    assert process.killed
