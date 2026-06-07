from pathlib import Path

import pytest

from osu_mania_renderer_v2 import RenderOptions, render_mania
from osu_mania_renderer_v2.errors import NotAManiaError


async def test_orchestrator_rejects_non_mania(tmp_path: Path, fixtures_dir: Path):
    # Reuse the std fixture from T5.
    osr = fixtures_dir / "std_replay.osr"
    out = tmp_path / "out.mp4"
    options = RenderOptions(resolution=(640, 360), fps=30)
    with pytest.raises(NotAManiaError):
        await render_mania(
            osr_path=osr,
            beatmap_dir=tmp_path,
            output_path=out,
            options=options,
        )


@pytest.mark.slow
async def test_orchestrator_end_to_end(fixtures_dir: Path, tmp_path: Path):
    """Render the real AO-INFINITY replay against the cached beatmap."""
    import os
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 required")
    bm_dir = Path("/var/mnt/Synology-Reddie/Mania ORDR Bot/beatmaps/"
                  "3e37a2abc23502109072187911229864")
    if not bm_dir.exists():
        pytest.skip(f"beatmap cache not present at {bm_dir}")
    out = tmp_path / "ao_infinity.mp4"
    options = RenderOptions(resolution=(1280, 720), fps=30, timeout_seconds=600)
    progress_seen = []

    async def on_progress(f):
        progress_seen.append(f)

    await render_mania(
        osr_path=fixtures_dir / "ao_infinity_hard.osr",
        beatmap_dir=bm_dir,
        output_path=out,
        options=options,
        progress_callback=on_progress,
    )
    assert out.exists()
    assert out.stat().st_size > 0
    assert any(p > 0 for p in progress_seen)
