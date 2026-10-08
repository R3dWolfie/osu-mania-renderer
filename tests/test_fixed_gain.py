"""Song loudness by one measured gain (render/loudnorm_cache.py, the switch
R3D_MANIA_FIXED_GAIN / R3D_FIXED_GAIN).

What must hold:
  * with nothing set the cache builds what it always built, under the key it
    always used;
  * the engine's own switch wins over the node-wide one;
  * with the switch on, the artifact has its own key, is the plain decode times
    ONE constant, lands on the target loudness and stays under the ceiling;
  * a second call is a hit (nothing is rebuilt);
  * an ffmpeg whose summary cannot be read gives the stock artifact.

Runnable two ways:  pytest tests/test_fixed_gain.py   OR   python tests/test_fixed_gain.py
"""
from __future__ import annotations

import asyncio
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_mania_renderer_v2.render import loudnorm_cache as lc_mania  # noqa: E402

COPIES = (lc_mania,)
_ENV = ("R3D_MANIA_FIXED_GAIN", "R3D_FIXED_GAIN", "R3D_LOUDNORM_CACHE_DIR",
        "R3D_NO_LOUDNORM_CACHE")


def _run(fn, *a, **kw):
    return asyncio.run(fn(*a, **kw))


class _Env:
    """Set / clear the switches for one block, and put the old values back."""
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in _ENV}
        for k in _ENV:
            os.environ.pop(k, None)
        for k, v in self.kw.items():
            if v is not None:
                os.environ[k] = str(v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _wav(path, seconds, level, rate=48000):
    """A tone with a slow swell (something a gain rider would ride)."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    x = level * (0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t))) \
        * np.sin(2 * np.pi * 440.0 * t)
    data = np.repeat(x[:, None], 2, axis=1).astype("<f4").tobytes()
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt ")
        fh.write(struct.pack("<IHHIIHH", 16, 3, 2, rate, rate * 8, 8, 32))
        fh.write(b"data" + struct.pack("<I", len(data)) + data)
    return Path(path)


def _plain(src):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i",
                        str(src), "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
                        "-ar", "48000", "-ac", "2", "-"], capture_output=True)
    return np.frombuffer(r.stdout, dtype="<f4")


def _lufs(lc, pcm):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
                        "-f", "f32le", "-ar", "48000", "-ac", "2", "-i", "-",
                        "-af", lc.EBUR128, "-f", "null", "-"],
                       input=pcm.astype("<f4").tobytes(), capture_output=True)
    return lc.parse_integrated_lufs(r.stderr.decode(errors="replace"))


def test_the_switch():
    for lc in COPIES:
        with _Env():
            assert lc.fixed_gain_on() is False
        with _Env(R3D_FIXED_GAIN=1):
            assert lc.fixed_gain_on() is True
        with _Env(R3D_MANIA_FIXED_GAIN=1):
            assert lc.fixed_gain_on() is True
        # the engine's own switch wins, both ways
        with _Env(R3D_FIXED_GAIN=1, R3D_MANIA_FIXED_GAIN=0):
            assert lc.fixed_gain_on() is False
        with _Env(R3D_FIXED_GAIN=0, R3D_MANIA_FIXED_GAIN=1):
            assert lc.fixed_gain_on() is True
        with _Env(R3D_FIXED_GAIN="off"):
            assert lc.fixed_gain_on() is False


def test_gain_arithmetic_and_the_summary():
    for lc in COPIES:
        f = lc.fixed_gain_db
        assert abs(f(-7.6, 1.0) - (-10.4)) < 1e-9
        assert abs(f(-30.0, 0.1) - 12.0) < 1e-9
        want = lc.PEAK_CEILING_DB - 20 * math.log10(0.9)
        assert abs(f(-30.0, 0.9) - want) < 1e-9 and f(-30.0, 0.9) < 12.0
        assert f(None, 0.5) == 0.0 and f(-70.0, 0.5) == 0.0 and f(-18.0, 0.0) == 0.0
        p = lc.parse_integrated_lufs
        s = ("  Integrated loudness:\n    I:          -7.6 LUFS\n"
             "    Threshold: -17.8 LUFS\n\n  Loudness range:\n"
             "    LRA:         2.7 LU\n    LRA low:    -9.2 LUFS\n")
        assert p(s) == -7.6 and p("") is None
        assert p(s.replace("    I:          -7.6 LUFS\n", "")) is None
        # the two artifacts can never share a key, and std writes the same name
        assert lc.FIXED_GAIN_PARAM == "fixedgain:I=-18:P=-1.5"
        assert "loudnorm" not in lc.FIXED_GAIN_PARAM


def test_nothing_set_builds_what_it_always_built():
    if not shutil.which("ffmpeg"):
        return
    for lc in COPIES:
        d = Path(tempfile.mkdtemp(prefix="r3d-gain-"))
        try:
            src = _wav(d / "song.wav", 6.0, 0.6)
            with _Env(R3D_LOUDNORM_CACHE_DIR=d / "cache"):
                got = _run(lc.get_or_build_normalized, src, rate=1.0, pitch=False)
                want = d / "want.f32le"
                assert _run(lc._build, src, 1.0, False, want)
                assert got == d / "cache" / (lc.compute_key(src, 1.0, False) + ".f32le")
                assert got.read_bytes() == want.read_bytes()
                assert lc.compute_key(src, 1.0, False) == \
                    lc.compute_key(src, 1.0, False, lc.LOUDNORM)
                assert [p.name for p in (d / "cache").iterdir()] == [got.name]
        finally:
            shutil.rmtree(d, ignore_errors=True)


def test_switch_on_one_gain_on_the_target():
    if not shutil.which("ffmpeg"):
        return
    for lc in COPIES:
        d = Path(tempfile.mkdtemp(prefix="r3d-gain-"))
        try:
            with _Env(R3D_LOUDNORM_CACHE_DIR=d / "cache", R3D_MANIA_FIXED_GAIN=1):
                for level, name in ((0.9, "loud"), (0.02, "quiet")):
                    src = _wav(d / f"{name}.wav", 8.0, level)
                    got = _run(lc.get_or_build_normalized, src, rate=1.0, pitch=False)
                    assert got is not None and got.name == \
                        lc.compute_key(src, 1.0, False, lc.FIXED_GAIN_PARAM) + ".f32le"
                    assert got.name != lc.compute_key(src, 1.0, False) + ".f32le"
                    fixed, plain = np.fromfile(got, dtype="<f4"), _plain(src)
                    assert fixed.shape == plain.shape
                    k = float(np.abs(fixed).max() / np.abs(plain).max())
                    assert np.allclose(fixed, plain * k, rtol=0, atol=2e-6), name
                    assert (k < 1.0) == (name == "loud"), (name, k)
                    lufs = _lufs(lc, fixed)
                    assert abs(lufs - lc.TARGET_LUFS) <= 0.15, (name, lufs)
                    assert float(np.abs(fixed).max()) <= \
                        10 ** (lc.PEAK_CEILING_DB / 20) + 1e-6
                    # a second call is a hit: the file is not written again
                    before = got.stat().st_mtime_ns
                    assert _run(lc.get_or_build_normalized, src, rate=1.0, pitch=False) == got
                    assert got.stat().st_mtime_ns == before
                # the argument beats the switch
                src = d / "loud.wav"
                stock = _run(lc.get_or_build_normalized, src, rate=1.0, pitch=False,
                                                   fixed_gain=False)
                assert stock.name == lc.compute_key(src, 1.0, False) + ".f32le"
        finally:
            shutil.rmtree(d, ignore_errors=True)


def _noise(path, seconds, level=0.25, rate=48000, seed=7):
    """Steady band-limited noise: nothing periodic, so two time-stretches of it
    only line up if they are the same stretch."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * rate))
    x = np.convolve(x, np.ones(8) / 8.0, mode="same") * level * 2.0
    data = np.repeat(x[:, None], 2, axis=1).astype("<f4").tobytes()
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt ")
        fh.write(struct.pack("<IHHIIHH", 16, 3, 2, rate, rate * 8, 8, 32))
        fh.write(b"data" + struct.pack("<I", len(data)) + data)
    return Path(path)


def _as_aac(wav):
    """The same sound as an .m4a. Songs are mp3 or ogg, whose decoders hand
    ffmpeg PLANAR samples; that is the case in which the stock chain stretches
    at 192 kHz (a float wav does not trigger it). AAC decodes planar too, and
    every ffmpeg can encode it."""
    out = Path(str(wav)[:-4] + ".m4a")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                    str(wav), "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
    return out


def _corr0(a, b):
    """Correlation at lag 0 of two equally long signals."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def test_a_speed_changed_song_is_stretched_exactly_as_stock():
    for lc in COPIES:
        # where the chain is pinned to loudnorm's rate, and where it is not
        assert lc.fixed_gain_chain([]) == lc.EBUR128
        assert lc.fixed_gain_chain(["atempo=1.5"]) == \
            "atempo=1.5,aformat=sample_rates=192000," + lc.EBUR128
        nc = lc._rate_filters(1.5, True)
        assert lc.fixed_gain_chain(nc) == ",".join(nc + [lc.EBUR128])
    if not shutil.which("ffmpeg"):
        return
    for lc in COPIES:
        d = Path(tempfile.mkdtemp(prefix="r3d-gain-"))
        try:
            src = _as_aac(_noise(d / "noise.wav", 6.0))
            with _Env(R3D_LOUDNORM_CACHE_DIR=d / "cache"):
                for rate in (1.5, 0.75):
                    stock = np.fromfile(_run(lc.get_or_build_normalized, src, rate=rate, pitch=False,
                                        fixed_gain=False), dtype="<f4")
                    fixed = np.fromfile(_run(lc.get_or_build_normalized, src, rate=rate, pitch=False,
                                        fixed_gain=True), dtype="<f4")
                    # the same stretch: the same length to the sample, lined up
                    assert fixed.shape == stock.shape, (rate, fixed.shape, stock.shape)
                    assert _corr0(fixed, stock) >= 0.97, (rate, _corr0(fixed, stock))
        finally:
            shutil.rmtree(d, ignore_errors=True)


def test_an_unreadable_summary_gives_the_stock_artifact():
    if not shutil.which("ffmpeg"):
        return
    for lc in COPIES:
        d = Path(tempfile.mkdtemp(prefix="r3d-gain-"))
        real = lc.parse_integrated_lufs
        try:
            src = _wav(d / "song.wav", 5.0, 0.5)
            lc.parse_integrated_lufs = lambda text: None
            with _Env(R3D_LOUDNORM_CACHE_DIR=d / "cache", R3D_FIXED_GAIN=1):
                got = _run(lc.get_or_build_normalized, src, rate=1.0, pitch=False)
            assert got is not None
            assert got.name == lc.compute_key(src, 1.0, False) + ".f32le"
            assert [p.name for p in (d / "cache").iterdir()] == [got.name]
        finally:
            lc.parse_integrated_lufs = real
            shutil.rmtree(d, ignore_errors=True)


def test_silence_stays_silence():
    if not shutil.which("ffmpeg"):
        return
    for lc in COPIES:
        d = Path(tempfile.mkdtemp(prefix="r3d-gain-"))
        try:
            src = _wav(d / "silent.wav", 3.0, 0.0)
            with _Env(R3D_LOUDNORM_CACHE_DIR=d / "cache", R3D_MANIA_FIXED_GAIN=1):
                got = _run(lc.get_or_build_normalized, src, rate=1.0, pitch=False)
            assert got is not None and not np.fromfile(got, dtype="<f4").any()
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok  ", _n)
