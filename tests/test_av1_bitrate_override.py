"""Published AV1 override must reach the encoder; no GPU or ffmpeg required."""
from pathlib import Path

import pytest

from osu_mania_renderer_v2.cli import _build_parser, _render_options_from_args
from osu_mania_renderer_v2.render import encode


def _command(override, *, preview=False):
    return encode.build_ffmpeg_cmd(
        encoder="av1_vaapi", encoder_device="/dev/dri/renderD129",
        resolution=(1920, 1080), fps=60, audio_path=None, audio_rate=1.0,
        audio_lead_in_ms=0, video_bitrate="2500k", video_bitrate_override=override,
        audio_bitrate="160k", output_path=Path("master.mp4"),
        preview_path=Path("preview.mp4") if preview else None,
        compact_path=Path("compact.mp4") if preview else None,
    )


def test_explicit_vaapi_override_selects_vbr_instead_of_cqp():
    args = encode._av1_codec_args("av1_vaapi", 1920, 1080, 60, 3_000_000)
    assert args == ["-c:v", "av1_vaapi", "-rc_mode", "VBR", "-b:v", "3000000",
                    "-maxrate", "4500000", "-profile:v", "main", "-g", "120"]


def test_omitted_override_preserves_existing_quality_recipe(monkeypatch):
    monkeypatch.delenv("R3D_AV1_VAAPI_GQ", raising=False)
    assert encode._av1_codec_args("av1_vaapi", 1920, 1080, 60, None) == [
        "-c:v", "av1_vaapi", "-rc_mode", "CQP", "-global_quality", "90",
        "-profile:v", "main", "-g", "120",
    ]
    monkeypatch.setenv("R3D_AV1_VAAPI_GQ", "54")
    assert "54" in encode._av1_codec_args("av1_vaapi", 1920, 1080, 60, None)


@pytest.mark.parametrize(("literal", "bps"), [
    ("3000000", 3_000_000), ("3000k", 3_000_000), ("3M", 3_000_000),
    ("8.5M", 8_500_000), ("10m", 10_000_000),
])
def test_cli_bitrate_reaches_actual_av1_master_command(literal, bps):
    args = _build_parser().parse_args([
        "replay.osr", "beatmap", "--encoder", "av1_vaapi", "--video-bitrate", literal,
    ])
    options = _render_options_from_args(args)
    assert options.video_bitrate_override == bps
    cmd = _command(options.video_bitrate_override)
    assert cmd[cmd.index("-c:v") + 1] == "av1_vaapi"
    assert cmd[cmd.index("-b:v") + 1] == str(bps)
    assert "-global_quality" not in cmd


def test_cli_omission_does_not_create_an_override():
    options = _render_options_from_args(_build_parser().parse_args(["replay.osr", "beatmap"]))
    assert options.video_bitrate_override is None


@pytest.mark.parametrize("literal", ["0", "-1", "0M", "nan", "inf", "1e999M", "garbage"])
def test_cli_rejects_nonpositive_or_nonfinite_bitrate(literal):
    with pytest.raises(SystemExit) as exc:
        _build_parser().parse_args(["replay.osr", "beatmap", "--video-bitrate", literal])
    assert exc.value.code == 2


def test_override_changes_only_av1_master_not_h264_companions():
    current = _command(None, preview=True)
    explicit = _command(3_000_000, preview=True)
    assert explicit[explicit.index("-c:v") + 1] == "av1_vaapi"
    assert "[vm0]format=nv12,hwupload[vm]" in explicit[explicit.index("-filter_complex") + 1]
    assert current[current.index("master.mp4") + 1:] == explicit[explicit.index("master.mp4") + 1:]
    codecs = [explicit[i + 1] for i, value in enumerate(explicit) if value == "-c:v"]
    assert codecs == ["av1_vaapi", "libx264", "libx264"]
