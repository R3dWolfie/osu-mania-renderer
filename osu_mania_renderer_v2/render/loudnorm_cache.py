"""Shared loudnorm PCM cache — cross-engine, box-local.

The single-pass loudnorm pass (``LOUDNORM``) is deterministic in
(source bytes, playback rate, pitch mode, param string) yet reruns the full
ffmpeg normalise on every render of the same track (~2-3 s for a typical song).
Memoise its f32le PCM output on the fast local SSD so a repeat render — or
another in-house engine rendering the same track — skips the pass.

This is the SAME cache the osu-std engine already uses
(``osu_std_renderer/record/audio.py``): identical directory, key recipe,
artifact format (``{key}.f32le`` = raw little-endian float32, 48 kHz stereo)
and kill-switch, so a track normalised by one mode is reused by another. Any
divergence here silently breaks that interop, so keep the four contract points
below (dir, key, format, kill-switch) byte-for-byte in step with the sibling.

Best-effort throughout: a missing / truncated / unreadable cache entry, or ANY
build failure, returns ``None`` so the caller falls back to the inline
(fused-loudnorm) encode path and the render still succeeds.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from pathlib import Path

from osu_mania_renderer_v2.render.encode import (
    LOUDNORM, LOUDNORM_CACHE_CH, LOUDNORM_CACHE_SR, _ffmpeg_prefix,
)

# --- contract shared with the sibling engines (osu-std record/audio.py) -------
DEFAULT_CACHE_DIR = "/data/r3d/loudnorm-cache"
CACHE_EXT = "f32le"                 # raw little-endian float32, 48 kHz stereo
_STRIDE = LOUDNORM_CACHE_CH * 4     # bytes per PCM frame (float32 * channels)
_CHUNK = 1 << 20                    # 1 MiB source-hash read chunk


def cache_disabled() -> bool:
    """Kill-switch, matching the sibling engines: ``R3D_NO_LOUDNORM_CACHE``.

    Default ON (unset / 0 / false / no / off = enabled); any other value
    disables the whole path — one env var kills the cache across every engine."""
    return os.environ.get("R3D_NO_LOUDNORM_CACHE", "").strip().lower() \
        not in ("", "0", "false", "no", "off")


def cache_dir() -> Path:
    return Path(os.environ.get("R3D_LOUDNORM_CACHE_DIR", DEFAULT_CACHE_DIR))


def compute_key(source: Path, rate: float, pitch: bool) -> str:
    """Stable hash of everything determining the loudnorm OUTPUT — sha256 of the
    SOURCE audio bytes + playback rate + pitch mode + the exact loudnorm param
    string. Byte-for-byte the sibling engines' recipe (double sha256, ``rate``
    via ``repr(float)``) so the ``{key}.f32le`` artifacts are shared.

    Raises OSError if the source can't be read (caller then runs uncached)."""
    h = hashlib.sha256()
    with open(source, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    material = "\n".join((
        f"src={h.hexdigest()}",
        f"rate={float(rate)!r}",
        f"pitch={1 if pitch else 0}",
        f"param={LOUDNORM}",
    )).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _rate_filters(rate: float, pitch: bool) -> list[str]:
    """The rate/pitch ffmpeg filters — IDENTICAL to build_ffmpeg_cmd's song
    chain, so the cached artifact is exactly what the fused pipeline would have
    fed into loudnorm. (For NoMod/DT/HT this also matches the sibling engines,
    making the artifact byte-shared; mania's NC resample chain differs from
    theirs, so NC entries are perceptually-equal but not byte-shared — see the
    engines' rate filters if strict cross-engine NC parity is ever needed.)"""
    if rate == 1.0:
        return []
    if pitch:  # NC — resample-based pitch-up (see encode.py)
        return ["aresample=44100", f"asetrate=44100*{rate}", "aresample=44100"]
    return [f"atempo={rate}"]  # DT/HT — pitch-preserving tempo


def _valid(path: Path) -> bool:
    """A cache file is usable iff it is a whole number of PCM frames and
    non-empty (mirrors the sibling's load-side stride check → treats an empty or
    truncated file as a miss)."""
    try:
        sz = path.stat().st_size
    except OSError:
        return False
    return sz > 0 and (sz % _STRIDE) == 0


async def _build(source: Path, rate: float, pitch: bool, target: Path) -> bool:
    """Run the loudnorm pre-pass → atomically publish raw f32le PCM to
    ``target``. Returns True on success. The ffmpeg command matches the sibling
    engines' decode (``-af <rate>,loudnorm -f f32le -ar 48000 -ac 2``) so the
    bytes are cross-engine-compatible for NoMod/DT/HT."""
    af = ",".join(_rate_filters(rate, pitch) + [LOUDNORM])
    cmd = [
        *_ffmpeg_prefix(), "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-vn", "-af", af,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(LOUDNORM_CACHE_SR), "-ac", str(LOUDNORM_CACHE_CH),
        "pipe:1",
    ]
    tmp: str | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        raw, err = await proc.communicate()
        if proc.returncode != 0 or not raw or (len(raw) % _STRIDE) != 0:
            raise RuntimeError(
                (err or b"").decode(errors="ignore")[-400:] or "bad pcm length")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(target.parent), prefix=".tmp-", suffix="." + CACHE_EXT)
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, target)  # atomic on the same filesystem
        return True
    except Exception:  # noqa: BLE001 — cache build is best-effort
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


async def get_or_build_normalized(
    source: Path, *, rate: float, pitch: bool
) -> Path | None:
    """Return a cached, loudness-normalised raw-f32le PCM file for ``source`` at
    the given rate/pitch — building it (and populating the shared cache) on a
    miss. Returns ``None`` when the cache is disabled or on ANY failure, so the
    caller falls back to the inline (fused) loudnorm path. Never raises."""
    try:
        if cache_disabled():
            return None
        target = cache_dir() / f"{compute_key(source, rate, pitch)}.{CACHE_EXT}"
        if _valid(target):
            return target  # HIT
        if await _build(source, rate, pitch, target):
            return target  # MISS → built
        return None
    except Exception:  # noqa: BLE001 — must never break a render
        return None
