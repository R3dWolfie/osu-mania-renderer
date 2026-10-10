"""The inline preview on the Mac's media engine (render/preview_hw.py, wired
into render/encode.py build_ffmpeg_cmd as `preview_hw`).

What must hold:
  * not asked for, the command is what it was;
  * asked for beside a libx264 master with a known length: the preview is
    encoded by VideoToolbox behind a half-second lead-in that is cut off again,
    its audio is ended at the video's length, its `-t` covers the lead-in, and
    THE MASTER'S ARGUMENTS ARE UNCHANGED;
  * asked for beside any other master encoder, or with no known length: ignored;
  * the switch: R3D_PREVIEW_VT=0 off, =1 ask anywhere, unset on a Mac; asking is
    not getting (the probe decides);
  * an ffmpeg that carried a hardware preview and died naming VideoToolbox
    switches the media engine off for a day; any other death does not.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.render import encode as enc
from osu_mania_renderer_v2.render import preview_hw as P


def _cmd(tmp=Path("/x"), **kw):
    args = dict(
        encoder="libx264", encoder_device=None, resolution=(1280, 720), fps=60,
        audio_path=tmp / "song.wav", audio_rate=1.0, audio_lead_in_ms=0,
        video_bitrate="2500k", video_bitrate_override=None, audio_bitrate="160k",
        output_path=tmp / "out.mp4", total_duration_ms=70192,
        preview_path=tmp / "out.embed.mp4")
    args.update(kw)
    return enc.build_ffmpeg_cmd(**args)


def _split(cmd, tmp=Path("/x")):
    """(everything up to and including the master's path, the preview's part)."""
    i = cmd.index(str(tmp / "out.mp4")) + 1
    return cmd[:i], cmd[i:]


def test_not_asked_for_is_the_command_as_it_was():
    assert _cmd() == _cmd(preview_hw=False)
    _, preview = _split(_cmd())
    assert preview[preview.index("-c:v") + 1] == "libx264" and "h264_videotoolbox" not in _cmd()


def test_asked_for_changes_only_the_preview():
    sw_master, sw_prev = _split(_cmd())
    hw_master, hw_prev = _split(_cmd(preview_hw=True))
    # the master: same arguments in the same order; only the shared filter graph differs
    g = sw_master.index("-filter_complex") + 1
    assert sw_master[:g] == hw_master[:g] and sw_master[g + 1:] == hw_master[g + 1:]
    sw_graph, hw_graph = sw_master[g], hw_master[g]
    lead = P.vt_lead_in_filter(30)
    assert hw_graph.replace("," + lead, "").replace("[ap]", "[AP]").replace("[ap_full]", "[ap]") \
        .replace(";[ap]atrim=end=70.192000[AP]", "") == sw_graph
    assert f"scale=-2:720,{lead}[vp]" in hw_graph
    # the preview: the media engine, its audio ended at the video's length, -t covering the lead-in
    assert hw_prev[hw_prev.index("-c:v") + 1] == "h264_videotoolbox"
    assert P.hw_video_args(enc.preview_video_bps(70.192), 30) == hw_prev[hw_prev.index("-c:v"):hw_prev.index("-color_range")]
    assert hw_prev[hw_prev.index("-t") + 1] == "70.692" and sw_prev[sw_prev.index("-t") + 1] == "70.192"
    assert hw_master[len(hw_master) - 3:len(hw_master) - 1] == ["-t", "70.192"]     # the master's own bound
    assert "-shortest" not in hw_prev


@pytest.mark.parametrize("kw", [
    dict(encoder="h264_vaapi", encoder_device="/dev/dri/renderD128"),
    dict(encoder="h264_nvenc"),
    dict(total_duration_ms=None),
])
def test_asked_for_where_it_cannot_apply_is_ignored(kw):
    assert _cmd(preview_hw=True, **kw) == _cmd(preview_hw=False, **kw)


def test_asked_for_without_a_preview_is_ignored():
    assert _cmd(preview_hw=True, preview_path=None) == _cmd(preview_path=None)


def test_the_switch(monkeypatch):
    calls = []
    monkeypatch.setattr(P, "_vt_probe", lambda ignore_off=False: calls.append(ignore_off) or True)

    def decide(value, platform_default):
        monkeypatch.setattr(P, "_vt_decision", None)
        monkeypatch.setattr(P, "_MAC_DEFAULT", platform_default)
        if value is None:
            monkeypatch.delenv("R3D_PREVIEW_VT", raising=False)
        else:
            monkeypatch.setenv("R3D_PREVIEW_VT", value)
        return P.preview_on_media_engine()

    assert decide(None, True) is True and calls[-1] is False       # a Mac, unset: ask, respecting the day-off
    assert decide(None, False) is False                              # not a Mac, unset: never
    assert decide("0", True) is False
    assert decide("1", False) is True and calls[-1] is True          # asked for by name: the day-off is ignored
    monkeypatch.setattr(P, "_vt_probe", lambda ignore_off=False: False)
    assert decide("1", True) is False                                # asking is not getting


def test_a_dead_hardware_preview_switches_itself_off(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "_r3d_cache_dir", lambda: str(tmp_path))
    mark = tmp_path / "vt-preview-off"

    def pipe(cmd, code, err):
        p = enc.FfmpegPipe.__new__(enc.FfmpegPipe)
        p.cmd, p.proc, p._stderr_log = cmd, SimpleNamespace(returncode=code), err
        return p

    hw = _cmd(preview_hw=True)
    assert pipe(hw, 0, b"") ._hw_preview_note() == "" and not mark.exists()
    assert pipe(hw, 1, b"Error opening output: no space left") ._hw_preview_note() == "" and not mark.exists()
    assert pipe(_cmd(), 1, b"[h264_videotoolbox] cannot create session") ._hw_preview_note() == "" and not mark.exists()
    note = pipe(hw, 187, b"[h264_videotoolbox @ 0x1] Error: cannot create compression session")._hw_preview_note()
    assert "off for 24 h" in note and mark.exists()
    monkeypatch.setattr(P, "_vt_decision", None)
    monkeypatch.setattr(P, "_MAC_DEFAULT", True)
    monkeypatch.delenv("R3D_PREVIEW_VT", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/sh")     # any real file: the marker is read first
    assert P.preview_on_media_engine() is False                        # the next render is on the CPU preview


@pytest.mark.slow
def test_a_real_two_output_encode_where_a_hardware_session_opens(tmp_path, monkeypatch):
    """Frames in, master and preview out, through the command as built."""
    if not shutil.which("ffmpeg") or not P._vt_probe(ignore_off=True):
        pytest.skip("no ffmpeg with a VideoToolbox session on this machine")
    import numpy as np
    w, h, fps, seconds = 320, 180, 60, 2
    song = tmp_path / "song.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds + 1}",
                    "-ac", "2", str(song)], check=True)
    out = {}
    for name, hw in (("sw", False), ("hw", True)):
        d = tmp_path / name
        d.mkdir()
        cmd = enc.build_ffmpeg_cmd(
            encoder="libx264", encoder_device=None, resolution=(w, h), fps=fps,
            audio_path=song, audio_rate=1.0, audio_lead_in_ms=0, video_bitrate="800k",
            video_bitrate_override=None, audio_bitrate="160k", output_path=d / "out.mp4",
            total_duration_ms=seconds * 1000, preview_path=d / "out.embed.mp4", preview_hw=hw)
        rng = np.random.default_rng(0)
        frames = b"".join(np.full((h, w, 3), rng.integers(0, 256, 3), np.uint8).tobytes() for _ in range(fps * seconds))
        r = subprocess.run(cmd, input=frames, capture_output=True)
        assert r.returncode == 0, r.stderr.decode()[-600:]
        probe = lambda f, s: subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", s, "-show_entries",
             "stream=codec_name,nb_frames,start_time,duration", "-of", "csv=p=0", str(f)],
            capture_output=True, text=True).stdout.strip()
        first = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                                "-of", "csv=p=0", str(d / "out.embed.mp4")], capture_output=True, text=True).stdout.split()[0]
        out[name] = ((d / "out.mp4").read_bytes(), probe(d / "out.embed.mp4", "v:0"), probe(d / "out.embed.mp4", "a:0"), first)
    assert out["sw"][0] == out["hw"][0]                       # the master is the same file
    assert out["sw"][1] == out["hw"][1]                       # same frames, start and length in the preview
    assert out["sw"][2] == out["hw"][2]                       # and the same audio shape
    assert out["hw"][3].startswith("0.000000,K")              # it starts on a keyframe at time 0
