"""GPU-free regression that locks the deterministic ``build_frame_state()``
output for the post-game results period.

``build_frame_state()`` is the shared simulation/draw-state truth behind the
canonical wiki pipeline registered by ``osu_mania_renderer_v2.render.pipeline``
(see that module's ``_ORDER``) -- it is the intended anchor for regressions
like this one, independent of any GPU/ffmpeg output.

Why the smoothers are threaded from frame 0: ``score_smoothed`` and
``accuracy_smoothed`` are single-pole low-pass filters carried forward every
frame by the real renderer (see ``scripts/baseline_capture.py``, which this
test reuses ``state_hash`` from). Seeding them fresh at the results window
instead of walking them from frame 0 would hash smoother values the real
renderer never produces, making the "regression" meaningless.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import pytest

# Read-only import: populates the canonical wiki pipeline's element
# registry so this test targets the accepted production composition path
# (render/pipeline.py), not the retiring monolithic FrameRenderer.draw().
# build_frame_state() itself doesn't require the registry to be populated,
# but importing it here documents and pins the intended architecture.
import osu_mania_renderer_v2.render.pipeline  # noqa: F401
from osu_mania_renderer_v2 import RenderOptions
from osu_mania_renderer_v2.render.render import build_frame_state, build_render_plan
from scripts.baseline_capture import state_hash

FPS = 30
RESOLUTION = (1280, 720)
# Small fixed window of results-period frames -- enough to catch a
# SceneState regression (results overlay, score/accuracy smoothers, grade,
# etc.) without re-hashing the whole results period every CI run.
WINDOW_FRAMES = 20

# Pinned with PYTHONHASHSEED=0, proved identical across two independent
# runs of this exact test (see codex-summary.md for the transcript).
EXPECTED_DIGEST = "da731ad320f860d028d47c732004e91a7f94a63d20a4a2ddb7029972e7bc5986"


async def test_results_period_frame_state_is_pinned(fixtures_dir: Path, tmp_path: Path):
    # Mania's beatmap/replay conversion path has documented Python
    # hash-order sensitivity. A digest produced without PYTHONHASHSEED=0
    # pinned is not reproducible, so refuse to compare against one instead
    # of silently asserting a value that may just be this run's luck.
    if os.environ.get("PYTHONHASHSEED") != "0":
        pytest.fail(
            "test_results_composition_regression requires PYTHONHASHSEED=0 "
            "(interpreter-start env var -- a test can't set this "
            "retroactively). Run e.g.: "
            "PYTHONHASHSEED=0 python -m pytest tests/test_results_composition_regression.py"
        )

    plan = await build_render_plan(
        osr_path=fixtures_dir / "ao_infinity_hard.osr",
        beatmap_dir=fixtures_dir,
        output_path=tmp_path / "unused.mp4",  # plan phase never writes this
        options=RenderOptions(resolution=RESOLUTION, fps=FPS),
    )

    # First frame whose t_ms lands at-or-after the post-game results period.
    f_start = math.ceil(plan.results_start_ms * FPS / 1000)
    assert f_start < plan.total_frames, (
        "fixture's results period doesn't fit inside its own render plan -- "
        "pick a longer fixture or a shorter window"
    )
    f_end = min(plan.total_frames, f_start + WINDOW_FRAMES)

    score_smoothed, accuracy_smoothed = 0.0, 100.0
    agg = hashlib.sha256()
    hashed_frames = 0
    for frame_n in range(0, f_end):
        t_ms = int(frame_n * 1000 / FPS)  # identical to render.py's loop
        scene, score_smoothed, accuracy_smoothed = build_frame_state(
            plan, t_ms, score_smoothed, accuracy_smoothed,
        )
        if frame_n >= f_start:
            assert t_ms >= plan.results_start_ms, (
                "hashed a gameplay-only frame instead of a results-period one"
            )
            agg.update(state_hash(scene).encode())
            hashed_frames += 1

    assert hashed_frames > 0, "no results-period frames were hashed"
    assert agg.hexdigest() == EXPECTED_DIGEST
