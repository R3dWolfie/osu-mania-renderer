"""Colour conversion on the GPU (gpu/yuv.py, R3D_MANIA_GPU_YUV).

What must hold:
  * the arithmetic IS ffmpeg's: the CPU twin equals what ffmpeg makes of the
    same frames, sample for sample (noise at several sizes, one-pixel stripes,
    hard colour edges), and so does the frame the stock command's filter makes;
  * converting and then flipping is flipping and then converting, which is
    what lets the frames stay bottom-up;
  * the warm-up frame is black in yuv, not zeros;
  * the switch: on by default on macOS only and only where this machine's
    ffmpeg agrees, forced either way, never with VAAPI or an unsupported size;
  * the ffmpeg command: untouched when off; when on, the input is declared
    yuv420p / tv / bt709 and the conversion is gone from the chain while the
    flip stays, with and without the preview and the Discord copy;
  * real pixels (RUN_SLOW=1): the shader pair equals the twin, and the frame
    reader hands out those frames with its usual lag, on both its consume
    paths.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from osu_mania_renderer_v2.gpu import yuv as Y
from osu_mania_renderer_v2.render import encode as enc


def _ffmpeg(rgb, vf):
    h, w = rgb.shape[:2]
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-i", "pipe:0", "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"],
        input=np.ascontiguousarray(rgb).tobytes(), capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8)


def _frames():
    rng = np.random.default_rng(20261010)
    out = {f"noise {w}x{h}": rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
           for w, h in ((128, 96), (1280, 720), (1024, 768), (64, 12))}
    stripes = np.zeros((96, 128, 3), np.uint8)
    stripes[:, ::2] = (255, 0, 255)
    stripes[1::2] = stripes[1::2][:, ::-1]
    out["one-pixel stripes"] = stripes
    edges = np.zeros((96, 128, 3), np.uint8)
    edges[:, :] = (0, 255, 0)
    edges[20:40, 30:90] = (255, 0, 0)
    edges[60:62] = (255, 255, 255)
    out["hard colour edges"] = edges
    return out


def test_the_twin_is_ffmpegs_conversion_sample_for_sample():
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    if not Y.ffmpeg_matches_twin():
        pytest.skip("this machine's ffmpeg does not use this arithmetic; "
                    "the engine then keeps the conversion in ffmpeg")
    for name, rgb in _frames().items():
        want = _ffmpeg(rgb, Y.FFMPEG_REFERENCE_VF)
        got = Y.rgb_to_yuv420p(rgb)
        assert np.array_equal(got, want), f"{name}: {int(np.count_nonzero(got != want))} samples differ"
    # and the SD matrix is a different conversion, so the comparison can fail
    rgb = _frames()["noise 128x96"]
    assert not np.array_equal(Y.rgb_to_yuv420p(rgb), _ffmpeg(rgb, "format=yuv420p"))


def _planes(buf, w, h):
    y = buf[:w * h].reshape(h, w)
    u = buf[w * h:w * h + w * h // 4].reshape(h // 2, w // 2)
    v = buf[w * h + w * h // 4:].reshape(h // 2, w // 2)
    return y, u, v


def test_convert_then_flip_is_flip_then_convert():
    for name, rgb in _frames().items():
        h, w = rgb.shape[:2]
        a = _planes(Y.rgb_to_yuv420p(rgb[::-1]), w, h)             # flip, then convert
        b = [p[::-1] for p in _planes(Y.rgb_to_yuv420p(rgb), w, h)]  # convert, then flip
        for pa, pb in zip(a, b):
            assert np.array_equal(pa, pb), name


def test_the_warm_up_frame_is_black_not_zeros():
    for w, h in ((128, 96), (1280, 720)):
        black = Y.black_frame(w, h)
        assert len(black) == w * h * 3 // 2
        assert black == Y.rgb_to_yuv420p(np.zeros((h, w, 3), np.uint8)).tobytes()
        assert set(black[:w * h]) == {16} and set(black[w * h:]) == {128}


def test_the_switch(monkeypatch):
    seen = []
    monkeypatch.setattr(Y, "ffmpeg_matches_twin", lambda prefix=("ffmpeg",): seen.append(1) or True)
    monkeypatch.delenv("R3D_MANIA_GPU_YUV", raising=False)
    monkeypatch.setattr(Y.sys, "platform", "darwin")
    assert Y.wanted(1280, 720, "libx264") is True and seen
    for platform in ("linux", "win32"):                 # unset: macOS only
        monkeypatch.setattr(Y.sys, "platform", platform)
        assert Y.wanted(1280, 720, "libx264") is False
    monkeypatch.setattr(Y.sys, "platform", "darwin")
    monkeypatch.setattr(Y, "ffmpeg_matches_twin", lambda prefix=("ffmpeg",): False)
    assert Y.wanted(1280, 720, "libx264") is False      # ffmpeg disagrees: off
    monkeypatch.setenv("R3D_MANIA_GPU_YUV", "1")        # forced: no probe, any platform
    for platform in ("darwin", "linux"):
        monkeypatch.setattr(Y.sys, "platform", platform)
        assert Y.wanted(1280, 720, "libx264") is True
        assert Y.wanted(1280, 720, "h264_nvenc") is True
        assert Y.wanted(1280, 720, "h264_vaapi") is False       # never with VAAPI
        assert Y.wanted(1281, 720, "libx264") is False          # odd width
        assert Y.wanted(1280, 10, "libx264") is False           # under 12 rows
    monkeypatch.setattr(Y, "ffmpeg_matches_twin", lambda prefix=("ffmpeg",): True)
    for off in ("0", "off", "false"):
        monkeypatch.setenv("R3D_MANIA_GPU_YUV", off)
        monkeypatch.setattr(Y.sys, "platform", "darwin")
        assert Y.wanted(1280, 720, "libx264") is False


def _cmd(**kw):
    base = dict(encoder="libx264", encoder_device=None, resolution=(1280, 720),
                fps=60, audio_path=None, audio_rate=1.0, audio_lead_in_ms=0,
                video_bitrate="2500k", audio_bitrate="160k",
                output_path=Path("/x/out.mp4"))
    base.update(kw)
    return enc.build_ffmpeg_cmd(**base)


_VARIANTS = (
    {},
    {"preview_path": Path("/x/out.embed.mp4"), "total_duration_ms": 60000},
    {"preview_path": Path("/x/out.embed.mp4"), "compact_path": Path("/x/out.sm.mp4"),
     "total_duration_ms": 60000},
    {"stream_master": True},
    {"encoder": "h264_nvenc"},
    {"audio_path": Path("/x/a.mp3"), "hitsound_path": Path("/x/h.wav")},
)


def test_off_builds_the_command_it_always_did():
    for kw in _VARIANTS:
        assert _cmd(**kw) == _cmd(frames_yuv420p=False, **kw)
        joined = " ".join(_cmd(**kw))
        assert "-pix_fmt rgb24" in joined and "in_range=full:out_range=limited" in joined


def test_on_declares_the_frames_and_drops_only_the_conversion():
    for kw in _VARIANTS:
        stock, new = _cmd(**kw), _cmd(frames_yuv420p=True, **kw)
        i = new.index("-i")
        head = new[:i]
        # the frames are declared for what they are, before the input
        k = head.index("-pix_fmt")
        assert head[k:k + 6] == ["-pix_fmt", "yuv420p", "-color_range", "tv",
                                 "-colorspace", "bt709"], kw
        joined = " ".join(new)
        assert "rgb24" not in joined and "in_range=full" not in joined, kw
        assert "vflip" in joined, kw
        # nothing else moved: put the two differences back and it is stock
        back = list(new)
        del back[k + 2:k + 6]
        back[k + 1] = "rgb24"
        back = [a.replace("vflip,split", "vflip,scale=in_range=full:out_range=limited,format=yuv420p,split")
                if "split" in a else
                ("vflip,scale=in_range=full:out_range=limited,format=yuv420p" if a == "vflip" else a)
                for a in back]
        assert back == stock, kw
    with pytest.raises(ValueError):
        _cmd(frames_yuv420p=True, encoder="h264_vaapi", encoder_device="/dev/dri/renderD128")


def _gl(width, height):
    from osu_mania_renderer_v2.gpu.context import HeadlessGl
    return HeadlessGl(width=width, height=height)


def _paint(gl, rgb_bottom_up):
    """Put known pixels in the scene's colour texture (row 0 = bottom)."""
    h, w = rgb_bottom_up.shape[:2]
    rgba = np.dstack([rgb_bottom_up, np.full((h, w, 1), 255, np.uint8)])
    gl.fbo.color_attachments[0].write(np.ascontiguousarray(rgba).tobytes())


@pytest.mark.slow
@pytest.mark.parametrize("width,height", [(1280, 720), (1920, 1080), (1024, 768), (64, 12)])
def test_the_shader_pair_is_the_twin(width, height):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run real GL pixel checks")
    rng = np.random.default_rng(width * 7 + height)
    with _gl(width, height) as gl:
        conv = Y.YuvConverter(gl.ctx, width, height)
        for trial in range(3):
            rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            if trial == 2:                                   # hard edges and stripes
                rgb[:] = (0, 255, 0)
                rgb[:, ::2] = (255, 0, 255)
                rgb[height // 3: height // 3 + 2] = (255, 255, 255)
            _paint(gl, rgb)
            conv.run(gl.fbo.color_attachments[0])
            got = np.frombuffer(bytes(conv.read_bytes()), np.uint8)
            want = Y.rgb_to_yuv420p(rgb)
            assert np.array_equal(got, want), (
                f"{width}x{height} trial {trial}: {int(np.count_nonzero(got != want))} samples differ")


@pytest.mark.slow
@pytest.mark.parametrize("mapped", [True, False])
def test_the_frame_reader_hands_out_converted_frames_with_its_usual_lag(mapped, monkeypatch):
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run real GL pixel checks")
    from osu_mania_renderer_v2.gpu import readback
    if mapped:
        monkeypatch.delenv("R3D_MANIA_NO_MAPPED_READBACK", raising=False)
    else:
        monkeypatch.setenv("R3D_MANIA_NO_MAPPED_READBACK", "1")
    width, height, n = 128, 96, 7
    rng = np.random.default_rng(5)
    frames = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(n)]
    with _gl(width, height) as gl:
        reader = readback.FrameReader(gl.ctx, gl.fbo, components=3,
                                      yuv=Y.YuvConverter(gl.ctx, width, height))
        assert reader.frame_size == width * height * 3 // 2
        out = []
        for rgb in frames:
            _paint(gl, rgb)
            f = reader.read()
            if isinstance(f, readback.MappedFrame):
                out.append(bytes(f.mv))
                f.done.set()                      # what the ffmpeg writer does
            else:
                out.append(bytes(f))
        out += [bytes(f) for f in reader.drain()]
    black = Y.black_frame(width, height)
    assert len(out) == n + readback._LAG
    assert out[:readback._LAG] == [black] * readback._LAG
    for i, rgb in enumerate(frames):
        assert out[readback._LAG + i] == Y.rgb_to_yuv420p(rgb).tobytes(), f"frame {i}"


def test_the_cpu_way_out_gives_the_same_frames():
    """If the GPU passes cannot be built after ffmpeg was started for yuv420p,
    the stock RGB reader is wrapped and every frame converted by the twin."""
    import threading
    w, h = 64, 12
    rng = np.random.default_rng(9)
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(4)]

    class _Lease:                                    # stands in for MappedFrame
        def __init__(self, data):
            self.mv = memoryview(data)
            self.done = threading.Event()

    class _Stock:                                    # stands in for FrameReader
        def __init__(self):
            self.handed = []

        def read(self):
            i = len(self.handed)
            if i == 0:
                f = bytes(w * h * 3)                 # the RGB warm-up blank
            elif i % 2:
                f = _Lease(frames[i - 1].tobytes())
            else:
                f = bytearray(frames[i - 1].tobytes())
            self.handed.append(f)
            return f

        def drain(self):
            return [bytearray(frames[-1].tobytes())]

    stock = _Stock()
    reader = Y.CpuTwinReader(stock, w, h)
    assert reader.frame_size == w * h * 3 // 2
    out = [reader.read() for _ in range(4)] + reader.drain()
    assert out[0] == Y.black_frame(w, h)             # zeros in, black out
    for i in range(3):
        assert out[1 + i] == Y.rgb_to_yuv420p(frames[i]).tobytes()
    assert out[4] == Y.rgb_to_yuv420p(frames[-1]).tobytes()
    assert all(isinstance(o, bytes) for o in out)
    # every lease was released, or the stock reader would wait on it for ever
    assert all(f.done.is_set() for f in stock.handed if isinstance(f, _Lease))


def test_every_render_loop_takes_its_reader_from_the_plan():
    """The plan decides what ffmpeg expects; a loop that built its own RGB
    reader would hand it frames of the wrong size (this broke the wiki render
    path while this change was being written)."""
    import re
    root = Path(enc.__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "readback.py":
            continue
        src = path.read_text()
        if re.search(r"\bFrameReader\(", src):
            offenders.append(str(path.relative_to(root)))
        if "build_render_plan(" in src and path.name != "render.py":
            assert "reader_for_plan(" in src, path.name
    assert offenders == [], offenders
    assert "reader_for_plan(gl.ctx, gl.fbo, gpu_yuv=plan.gpu_yuv)" in (root / "render" / "render.py").read_text()


def test_reader_for_plan_picks_by_the_plan(monkeypatch):
    from osu_mania_renderer_v2.gpu import readback

    class _Fbo:
        size = (64, 12)
    monkeypatch.setattr(readback, "FrameReader",
                        lambda ctx, fbo, components=3, yuv=None: ("reader", yuv))
    assert readback.reader_for_plan(None, _Fbo(), gpu_yuv=False) == ("reader", None)
    monkeypatch.setattr(Y, "YuvConverter", lambda ctx, w, h: ("converter", w, h))
    assert readback.reader_for_plan(None, _Fbo(), gpu_yuv=True) == ("reader", ("converter", 64, 12))

    def boom(ctx, w, h):
        raise RuntimeError("no passes here")
    monkeypatch.setattr(Y, "YuvConverter", boom)
    fallback = readback.reader_for_plan(None, _Fbo(), gpu_yuv=True)
    assert isinstance(fallback, Y.CpuTwinReader) and fallback.frame_size == 64 * 12 * 3 // 2

