#!/usr/bin/env python3
"""Deterministic per-frame render-state baseline — the regression net for the
mania v3 reorg.

It hashes ``build_frame_state`` (the draw-instruction ``SceneState``) across a
frame range. That state is pure Python and GPU/encode-free, so a byte-identical
digest across a *structural* change (package rename, module moves, killing the
wiki fork) proves the simulation + draw-state is unchanged — a far more precise
and portable safety net than diffing encoded pixels (which carry GPU/ffmpeg
noise).

Workflow:
    # before a reorg step, on the render box (mania venv):
    python -m scripts.baseline_capture OSR BEATMAP_DIR --max-seconds 20 \
        --out /tmp/base_before.jsonl
    # ...do the reorg, then:
    python -m scripts.baseline_capture OSR BEATMAP_DIR --max-seconds 20 \
        --out /tmp/base_after.jsonl
    diff /tmp/base_before.jsonl /tmp/base_after.jsonl   # must be empty

Run it twice on the SAME code first to confirm the DIGEST is stable (mania is
byte-deterministic with PYTHONHASHSEED=0) before trusting it as a baseline.

``--start-ms`` / ``--max-seconds`` bound the hashed window; the single-pole
score/accuracy smoothers are always threaded from frame 0 (only frames inside
the window are recorded) so a windowed capture is identical to the same frames
of a full capture.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import struct
from pathlib import Path

from osu_mania_renderer_v2 import RenderOptions
from osu_mania_renderer_v2.render.render import build_frame_state, build_render_plan


def _feed(h: "hashlib._Hash", obj) -> None:
    """Feed a value into the hash in a canonical, order-stable, type-tagged way.
    Tags prevent collisions between e.g. the int 1, the str '1', and True."""
    t = type(obj)
    if obj is None:
        h.update(b"N")
    elif obj is True:
        h.update(b"T")
    elif obj is False:
        h.update(b"Fa")
    elif t is int:
        h.update(b"i")
        h.update(repr(obj).encode())
    elif t is float:
        h.update(b"f")
        h.update(struct.pack("<d", obj))  # exact bits, incl. -0.0 / NaN payloads
    elif t is str:
        h.update(b"s")
        h.update(obj.encode("utf-8", "surrogatepass"))
    elif t is bytes:
        h.update(b"y")
        h.update(obj)
    elif t in (list, tuple):
        h.update(b"[")
        for x in obj:
            _feed(h, x)
        h.update(b"]")
    elif t is dict:
        h.update(b"{")
        for k in sorted(obj):
            _feed(h, k)
            _feed(h, obj[k])
        h.update(b"}")
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(b"D")
        h.update(t.__name__.encode())
        for fld in dataclasses.fields(obj):
            h.update(fld.name.encode())
            _feed(h, getattr(obj, fld.name))
    elif hasattr(obj, "tobytes") and hasattr(obj, "dtype"):  # numpy ndarray
        h.update(b"A")
        h.update(str(obj.dtype).encode())
        h.update(repr(obj.shape).encode())
        h.update(obj.tobytes())
    else:  # last-resort: stable repr (enums, etc.). Flagged so drift is visible.
        h.update(b"r")
        h.update(repr(obj).encode())


def state_hash(scene) -> str:
    h = hashlib.sha256()
    _feed(h, scene)
    return h.hexdigest()


async def _run(a: argparse.Namespace) -> int:
    w, h = (int(x) for x in a.resolution.lower().split("x"))
    options = RenderOptions(resolution=(w, h), fps=a.fps)
    plan = await build_render_plan(
        osr_path=a.osr,
        beatmap_dir=a.beatmap_dir,
        output_path=Path("/tmp/_baseline_capture_ignored.mp4"),
        options=options,
        allow_converted=a.allow_converted,
        convert_to_keys=a.convert_to_keys,
    )
    f_start = int(a.start_ms * a.fps / 1000)
    if a.max_seconds is None:
        f_end = plan.total_frames
    else:
        f_end = min(plan.total_frames, f_start + int(round(a.max_seconds * a.fps)))

    score_smoothed, accuracy_smoothed = 0.0, 100.0
    agg = hashlib.sha256()
    records: list[dict] = []
    for frame_n in range(0, f_end):
        t_ms = int(frame_n * 1000 / a.fps)  # identical to render.py's loop
        scene, score_smoothed, accuracy_smoothed = build_frame_state(
            plan, t_ms, score_smoothed, accuracy_smoothed,
        )
        if frame_n >= f_start:
            hh = state_hash(scene)
            agg.update(hh.encode())
            records.append({"frame": frame_n, "t_ms": t_ms, "hash": hh})

    digest = agg.hexdigest()
    out = a.out or Path(f"baseline_{f_start}_{digest[:8]}.jsonl")
    with open(out, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"key_count={plan.key_count} total_frames={plan.total_frames} "
          f"fps={a.fps} res={w}x{h}")
    print(f"hashed frames {f_start}..{f_end} ({len(records)} frames)")
    print(f"DIGEST {digest}")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="deterministic mania render-state baseline")
    ap.add_argument("osr", type=Path, help=".osr replay file")
    ap.add_argument("beatmap_dir", type=Path, help="dir with the .osu (+ audio/bg)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", default="1280x720")
    ap.add_argument("--start-ms", type=int, default=0, help="window start (ms)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="window length in seconds (default: whole render)")
    ap.add_argument("--allow-converted", action="store_true")
    ap.add_argument("--convert-to-keys", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="per-frame hash JSONL")
    return asyncio.run(_run(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
