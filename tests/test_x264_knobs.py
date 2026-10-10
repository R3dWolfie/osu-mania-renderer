"""R3D_X264_* knobs on the libx264 master (render/encode.py). With none set the
command must be what it always was; set, they replace the flat bitrate target
and add the preset / thread cap; they never touch another encoder."""
import os
from pathlib import Path

from osu_mania_renderer_v2.render import encode as enc

_ENV = {"_X264_PRESET": "R3D_X264_PRESET", "_X264_CRF": "R3D_X264_CRF",
        "_X264_THREADS": "R3D_X264_THREADS", "_X264_PARAMS": "R3D_X264_PARAMS"}


def _build(encoder="libx264", override=None):
    cmd = enc.build_ffmpeg_cmd(
        encoder=encoder, encoder_device=None, resolution=(1920, 1080),
        fps=60, audio_path=None, audio_rate=1.0, audio_lead_in_ms=0,
        video_bitrate="2500k", video_bitrate_override=override,
        audio_bitrate="160k", output_path=Path("/x/out.mp4"))
    i = cmd.index("-c:v")
    return cmd[i:cmd.index("-color_range")]


def _video_args(encoder="libx264", override=None, **knobs):
    # through the ENVIRONMENT, which is how a node sets them: setting module
    # attributes would pass even if the engine only read the environment once
    saved = {e: os.environ.get(e) for e in _ENV.values()}
    try:
        for k, e in _ENV.items():
            if knobs.get(k):
                os.environ[e] = knobs[k]
            else:
                os.environ.pop(e, None)
        return _build(encoder, override)
    finally:
        for e, v in saved.items():
            if v is None:
                os.environ.pop(e, None)
            else:
                os.environ[e] = v


def test_a_value_set_after_import_is_used_and_the_next_job_can_change_it():
    # the long-lived-worker case: one process, a different environment per job
    saved = {e: os.environ.get(e) for e in _ENV.values()}
    try:
        for e in _ENV.values():
            os.environ.pop(e, None)
        assert _build() == ["-c:v", "libx264", "-b:v", "2500k"]
        os.environ["R3D_X264_PRESET"] = "veryfast"
        assert _build() == ["-c:v", "libx264", "-b:v", "2500k", "-preset", "veryfast"]
        os.environ["R3D_X264_PRESET"] = "faster"
        os.environ["R3D_X264_CRF"] = "21"
        assert _build() == ["-c:v", "libx264", "-crf", "21", "-preset", "faster"]
        for e in _ENV.values():
            os.environ.pop(e, None)
        assert _build() == ["-c:v", "libx264", "-b:v", "2500k"]
    finally:
        for e, v in saved.items():
            if v is None:
                os.environ.pop(e, None)
            else:
                os.environ[e] = v


def test_unset_is_the_command_as_it_was():
    assert _video_args() == ["-c:v", "libx264", "-b:v", "2500k"]


def test_crf_replaces_the_bitrate_target():
    assert _video_args(_X264_CRF="20", _X264_PRESET="veryfast",
                       _X264_THREADS="6") == [
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-threads", "6"]


def test_preset_alone_keeps_the_bitrate_target():
    assert _video_args(_X264_PRESET="veryfast") == [
        "-c:v", "libx264", "-b:v", "2500k", "-preset", "veryfast"]


def test_an_explicit_bitrate_override_wins_over_crf():
    assert _video_args(override=6_000_000, _X264_CRF="20") == [
        "-c:v", "libx264", "-b:v", "6000000"]


def test_params_are_passed_through():
    assert _video_args(_X264_PARAMS="ref=2:bframes=1")[-2:] == [
        "-x264-params", "ref=2:bframes=1"]


def test_other_encoders_are_untouched():
    assert _video_args(encoder="libopenh264", _X264_CRF="20",
                       _X264_PRESET="veryfast", _X264_THREADS="6") == [
        "-c:v", "libopenh264", "-b:v", "2500k"]


def test_the_engines_own_preset_name_wins_over_the_node_wide_one(monkeypatch):
    for e in _ENV.values():
        monkeypatch.delenv(e, raising=False)
    monkeypatch.delenv("R3D_MANIA_X264_PRESET", raising=False)
    assert _build() == ["-c:v", "libx264", "-b:v", "2500k"]                 # nothing set: as it always was
    monkeypatch.setenv("R3D_MANIA_X264_PRESET", "veryfast")
    assert _build() == ["-c:v", "libx264", "-b:v", "2500k", "-preset", "veryfast"]
    monkeypatch.setenv("R3D_X264_PRESET", "faster")                          # the node-wide name is set too
    assert _build() == ["-c:v", "libx264", "-b:v", "2500k", "-preset", "veryfast"]
    monkeypatch.delenv("R3D_MANIA_X264_PRESET")
    assert _build() == ["-c:v", "libx264", "-b:v", "2500k", "-preset", "faster"]
    monkeypatch.setenv("R3D_MANIA_X264_PRESET", "  ")                        # blank is not a preset
    assert _build() == ["-c:v", "libx264", "-b:v", "2500k", "-preset", "faster"]
    monkeypatch.setenv("R3D_MANIA_X264_PRESET", "veryfast")
    assert _build("h264_nvenc")[:2] == ["-c:v", "h264_nvenc"] and "-preset" not in _build("h264_nvenc")
