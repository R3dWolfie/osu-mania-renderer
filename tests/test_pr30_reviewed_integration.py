"""PR #30 coexists with the reviewed e3ca887 renderer, without GPU resources."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from osu_mania_renderer_v2.beatmap.models import RenderOptions
from osu_mania_renderer_v2.render import encode as enc
from osu_mania_renderer_v2.render import render as rendering
from tests.test_mania_converted_replay_autodetect import mania_replay, source_map

FEATURE_ENV = (
    "R3D_STREAM_MASTER", "R3D_COMPACT_INLINE", "R3D_PREVIEW_LIVE", "R3D_PREVIEW_INLINE",
    "R3D_X264_PRESET", "R3D_X264_CRF", "R3D_X264_THREADS", "R3D_X264_PARAMS",
    "R3D_COMPACT_IF_OVER_BYTES", "R3D_COMPACT_EXPECT_BPS",
)
DEFAULT_CASES = (
    "x264-silent", "x264-song", "x264-mixed", "x264-cached", "nvenc-mixed",
    "vaapi-song", "amf-song", "qsv-song", "openh264-silent", "nc-mixed", "ht-mixed",
    "x264-override", "fifo",
)
# SHA-256 of complete JSON argv arrays captured from trusted e3ca887, with a
# fixed ffmpeg prefix. These are immutable baseline evidence, not current outputs.
DEFAULT_HASHES = {
    "x264-silent": "248a828c3819a6a65bf8ec77e15420a615d957cf54f935cdb075e10300859612",
    "x264-song": "ea2e819cf98ff792b40f8a12d6f92b82526ab6d5e5519dabf79bf749911249e5",
    "x264-mixed": "c5f32d29442be935bb6a368825276864863b7784a19eb5be872a747d81058cca",
    "x264-cached": "8aaa42bbc0f23631f01d06db2fc146222553d703e70e2937aa14749d080399c1",
    "nvenc-mixed": "a61c61404724b8e6c115e6e81ebe8146ff6de18d2356d3abf5eb6c8c30d39cd0",
    "vaapi-song": "3e52501f1a42583d8abbf38f112dd9a322ae9c77a6ff1ecdc913bb412d5cb9c3",
    "amf-song": "4ba46837d9245504e8aa9385b8548a30a1048d437b16bab8e88dd6e5db4d0efa",
    "qsv-song": "4aedfacdd001f8ac57d2fce641553a12c8d97865a20e534c29f0d7f0aa0e321d",
    "openh264-silent": "8e7492d6fbf3bc8d1515207fad2a3f08314ec49cb47f636a570d8fc1e7daa5f5",
    "nc-mixed": "4156f240fb487005268f545013cc58e311c2d01594fd32572276e7cca0d7ebaa",
    "ht-mixed": "02e492903168a61964f1d75bb73cbfa5e200642b798f83618669f8c01da0b435",
    "x264-override": "4d7662b0b133d89a58cdafc912bb70b6ac20ad963655f87fd2fa1b84793f2a44",
    "fifo": "d130a50a8a084cf5ec5eab12fbc32d806a44fa1f5010b55f7208d8ceaddb3460"
}


@pytest.fixture(autouse=True)
def clean_features(monkeypatch):
    for name in FEATURE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(enc, "_ffmpeg_prefix", lambda: ["ffmpeg"])
    monkeypatch.setattr(enc.os, "cpu_count", lambda: 8)


def ffmpeg_arguments(case):
    codec = {"nvenc": "h264_nvenc", "vaapi": "h264_vaapi", "amf": "h264_amf",
             "qsv": "h264_qsv", "openh264": "libopenh264"}.get(case.split("-")[0], "libx264")
    mixed = "mixed" in case or case in ("x264-cached", "x264-override")
    with_audio = mixed or "song" in case
    return dict(
        encoder=codec, encoder_device="/dev/dri/renderD128" if codec == "h264_vaapi" else None,
        resolution=(1280, 720), fps=60,
        audio_path=Path("/fixtures/song.mp3") if with_audio else None,
        audio_rate=1.5 if case == "nc-mixed" else .75 if case == "ht-mixed" else 1.0,
        audio_pitch=case == "nc-mixed", audio_lead_in_ms=1200 if mixed else 0,
        video_bitrate="2500k",
        video_bitrate_override=6_000_000 if case == "x264-override" else None,
        audio_bitrate="160k", output_path=Path("/renders/master.mp4"), total_duration_ms=4000,
        hitsound_path=Path("/renders/hits.wav") if mixed else None,
        prenormalized_audio_path=Path("/cache/song.f32le") if case == "x264-cached" else None,
        frames_fifo_path=Path("/tmp/frames.fifo") if case == "fifo" else None,
        music_volume=.8 if mixed else 1, hitsound_volume=.7 if mixed else 1,
    )


@pytest.mark.parametrize("case", DEFAULT_CASES)
def test_default_off_complete_argv_matches_reviewed_e3ca887(case):
    command = enc.build_ffmpeg_cmd(**ffmpeg_arguments(case))
    digest = hashlib.sha256(json.dumps(command).encode()).hexdigest()
    assert digest == DEFAULT_HASHES[case], command


async def test_runtime_fallback_and_per_job_x264_knobs_coexist(monkeypatch):
    monkeypatch.setattr(enc, "_run_probe", AsyncMock(return_value=(
        0, b" V..... h264_nvenc\n V..... libx264\n", b"")))
    attempts = []

    async def usable(encoder, device):
        attempts.append(encoder)
        return encoder == "libx264", "unusable hardware"

    monkeypatch.setattr(enc, "_encoder_usable", usable)
    monkeypatch.setenv("R3D_X264_CRF", "20")
    monkeypatch.setenv("R3D_X264_PRESET", "veryfast")
    chosen = await enc.probe_encoder("auto", None)
    assert chosen == "libx264" and attempts == ["h264_nvenc", "libx264"]
    command = enc.build_ffmpeg_cmd(**{**ffmpeg_arguments("x264-silent"), "encoder": chosen})
    assert command[command.index("-crf") + 1] == "20"
    assert command[command.index("-preset") + 1] == "veryfast"
    assert "-b:v" not in command


@pytest.mark.parametrize("preview,compact", [(False, False), (True, False), (True, True)])
def test_stream_master_normalises_final_mix_at_48k_without_faststart(preview, compact):
    options = ffmpeg_arguments("x264-mixed")
    if preview:
        options["preview_path"] = Path("/renders/master.embed.mp4")
    if compact:
        options["compact_path"] = Path("/renders/master.embed-sm.mp4")
    command = enc.build_ffmpeg_cmd(**options, stream_master=True)
    master = command[:command.index(str(options["output_path"]))]
    assert "+faststart" not in master
    assert master[master.index("-ar") + 1] == "48000"
    graph = command[command.index("-filter_complex") + 1]
    assert graph.index("[song][hits]amix=") < graph.index("aresample," + enc.LOUDNORM)
    assert graph.count(enc.LOUDNORM) == 2  # song first, final mixed output second
    if preview:
        assert "fps=30,scale=-2:720" in graph
        assert "aformat=sample_rates=48000,asplit=" in graph
    if compact:
        assert "split=3[vm0][vp0][vc0]" in graph
        assert "[vc0]fps=60,scale=-2:720:flags=bilinear[vc]" in graph
        assert str(options["compact_path"]) in command
        assert command[command.index("-color_range") + 1] == "tv"


def test_compact_request_and_budget_remain_opt_in(monkeypatch):
    assert not enc.compact_wanted(120, 1920, 1080, 60, .6)
    monkeypatch.setenv("R3D_COMPACT_INLINE", "1")
    assert enc.compact_wanted(120, 1920, 1080, 60, .6)
    monkeypatch.setenv("R3D_COMPACT_IF_OVER_BYTES", "10000000")
    monkeypatch.setenv("R3D_COMPACT_EXPECT_BPS", "100000")
    assert not enc.compact_wanted(120, 1920, 1080, 60, .6)
    monkeypatch.setenv("R3D_COMPACT_EXPECT_BPS", "1000000")
    assert enc.compact_wanted(120, 1920, 1080, 60, .6)


@pytest.mark.parametrize("duration,height,fps", [(0, 1080, 60), (60, 1080, 60),
                                               (300, 720, 60), (2000, 720, 30)])
def test_compact_plan_uses_source_duration_budget(duration, height, fps):
    actual_height, maxrate, audio, actual_fps = enc.compact_plan(duration)
    assert (actual_height, actual_fps) == (height, fps)
    assert maxrate > 0 and audio > 0


def test_live_preview_only_creates_its_requested_hls_sink(tmp_path, monkeypatch):
    preview = tmp_path / "master.embed.mp4"
    assert enc._preview_sink_args(preview) == ["-movflags", "+faststart", str(preview)]
    live = tmp_path / "master.live"
    assert not live.exists()
    live.mkdir()
    stale = live / "seg_00000.m4s"
    stale.write_bytes(b"stale segment")
    unrelated = tmp_path / "preserve.mp4"
    unrelated.write_bytes(b"preserve")
    monkeypatch.setenv("R3D_PREVIEW_LIVE", "1")
    sink = enc._preview_sink_args(preview)
    assert sink[:2] == ["-f", "hls"] and sink[-1] == str(live / "live.m3u8")
    assert "independent_segments+temp_file" in sink
    assert str(live / "seg_%05d.m4s") in sink
    assert not stale.exists() and unrelated.read_bytes() == b"preserve"


@pytest.mark.parametrize("mode,lazer", [(0, False), (0, True), (3, False), (3, True)])
@pytest.mark.parametrize("feature", ["stream", "compact", "live", "all"])
async def test_output_features_preserve_replay_plan_and_source_audio_inputs(
        tmp_path, monkeypatch, mode, lazer, feature):
    source = source_map(tmp_path, mode=mode, cs=4, od=6.5, total=10, duration=2)
    source.write_text(source.read_text().replace("Mode:", "AudioFilename:song.wav\nMode:"))
    (tmp_path / "song.wav").write_bytes(b"mocked audio runtime")
    skin = tmp_path / "skin"
    skin.mkdir()
    replay = replace(mania_replay(source), is_lazer_replay=lazer)
    monkeypatch.setattr(rendering, "parse_replay", lambda _: replay)
    monkeypatch.setattr(rendering, "probe_encoder", AsyncMock(return_value="libx264"))
    monkeypatch.setattr(rendering.loudnorm_cache, "get_or_build_normalized",
                        AsyncMock(return_value=None))
    monkeypatch.setattr(rendering, "require_hitsound_runtime", lambda: None)
    calls = []

    def hitsounds(**kwargs):
        calls.append(kwargs)
        return kwargs["output_wav"]

    monkeypatch.setattr(rendering, "build_hitsound_track", hitsounds)
    options = RenderOptions((320, 240), 60, use_skin_hitsounds=True)
    async def build():
        return await rendering.build_render_plan(osr_path=tmp_path / "synthetic.osr",
            beatmap_dir=tmp_path, output_path=tmp_path / "out.mp4", options=options, skin_dir=skin)

    baseline = await build()
    if feature in ("stream", "all"):
        monkeypatch.setenv("R3D_STREAM_MASTER", "1")
    if feature in ("compact", "live", "all"):
        monkeypatch.setenv("R3D_PREVIEW_INLINE", "1")
    if feature in ("compact", "all"):
        monkeypatch.setenv("R3D_COMPACT_INLINE", "1")
    if feature in ("live", "all"):
        monkeypatch.setenv("R3D_PREVIEW_LIVE", "1")
    if feature == "all":
        monkeypatch.setenv("R3D_X264_CRF", "21")
        monkeypatch.setenv("R3D_X264_PRESET", "veryfast")
    enabled = await build()
    assert calls[0] == calls[1]  # Includes raw notes, source facts and overlay skin directories.
    assert calls[1]["overlay_skin_dirs"] == (skin,)
    assert calls[1]["is_lazer_replay"] == lazer
    assert bool(calls[1]["lazer_facts"]) == lazer
    assert enabled.replay == baseline.replay and enabled.modded == baseline.modded
    assert enabled.modded.source_mode == mode and enabled.key_count == (7 if mode == 0 else 4)
    assert enabled.judgment_events == baseline.judgment_events
    assert enabled.legacy_presentation == baseline.legacy_presentation
    assert enabled.audio_path == baseline.audio_path
    assert enabled.hitsound_wav == baseline.hitsound_wav
    marker = tmp_path / "out.stream.json"
    assert marker.exists() == (feature in ("stream", "all"))
    if marker.exists():
        assert json.loads(marker.read_text())["compact"] == (feature == "all")
    if feature in ("compact", "all"):
        assert str(tmp_path / "out.embed-sm.mp4") in enabled.ffmpeg_cmd
    if feature in ("live", "all"):
        assert str(tmp_path / "out.live/live.m3u8") in enabled.ffmpeg_cmd
