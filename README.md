# osu! Renderer

A Python library that turns an osu! `.osr` replay plus the beatmap files
into an MP4. GPU-rendered via ModernGL (standalone EGL context), hardware-
encoded via ffmpeg + VAAPI when available.

## Install

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_placeholder_sprites.py
```

Requires `ffmpeg` on `$PATH` (with libx264 — VAAPI optional but recommended on
AMD/Intel GPUs).

## Usage — library

```python
import asyncio
from pathlib import Path

from osu_renderer import RenderOptions, render_mania

async def main():
    await render_mania(
        osr_path=Path("play.osr"),
        beatmap_dir=Path("./beatmap/"),  # contains the .osu + audio.mp3 + bg
        output_path=Path("out.mp4"),
        options=RenderOptions(resolution=(1920, 1080), fps=60),
    )

asyncio.run(main())
```

## Usage — CLI

```bash
osu-renderer play.osr ./beatmap/ -o out.mp4 --resolution 1920x1080 --fps 60
```

## Supported mods

DT / NC / HT (speed + pitch), MR (mirror), HD (hidden), FI (fade in),
FL (flashlight), V2 (ScoreV2), 1K–9K (key count locks). NF / EZ / HR / SD / PF
have no visual effect.

**Not supported in v1:** RD (Random — replays render as NM column order with a
warning), KC (Key Coop — replays render as single playfield with a warning).

## Project status

v0.1 — alpha. The custom skin pipeline is in active development.

**For contributors:** start with [`docs/NAVIGATION.md`](docs/NAVIGATION.md) —
it's a code tour aimed at someone diving in cold, with extra detail on the
skin pipeline (`skin_ini.py`, `gpu/atlas.py`, `gpu/renderer.py`). The skin
overhaul design doc lives at
[`docs/superpowers/specs/2026-05-26-skin-fidelity-overhaul.md`](docs/superpowers/specs/2026-05-26-skin-fidelity-overhaul.md).

## License

MIT — see `LICENSE`. Note: any osu! skin you bundle (e.g. Night05) carries its
own license; check before redistributing.
