# R3D osu!mania Renderer

Turns an osu! `.osr` mania replay plus its beatmap files into an MP4 — GPU-rendered
via headless ModernGL/EGL, encoded with ffmpeg.

## Part of the R3D Renderer

This repo is the **osu!mania engine** for the [R3D Renderer](https://renderer.r3dwolfie.com),
a self-hosted osu! replay→MP4 service (Discord bot + website, package `mania_ordr`)
that dispatches each render to a per-mode engine. The core invokes this engine as a
fresh subprocess per render, so editing files here deploys on the next render with no
service restart. Production runs on the `mania-v3` branch.

## What it renders / fidelity

- osu!**mania** replays (`.osr`) → MP4, reproducing osu! timing, judgment, scoring
  and HUD behaviour ported from `osu.Game.Rulesets.Mania` / `osu.Game` (Argon HUD),
  cited in module docstrings.
- Headless GPU rendering through a standalone ModernGL **EGL** context (no window
  manager). Multi-GPU boxes can pin the device via the `R3D_EGL_DEVICE_INDEX`
  environment variable.
- ffmpeg encoding with selectable encoder: `auto`, `h264_vaapi`, `h264_nvenc`,
  or `libx264`; optional loudnorm normalization pass.
- One canonical **element-registry compositor** (`render/compositor.py` and
  `render/pipeline.py`) backed by the shared GPU sprite/text primitives. The
  sprite atlas is built once from an optional extracted `.osk` directory,
  skin.ini, and bundled fallbacks. `wiki_renderer.py` remains only as the
  worker command's compatibility module; `OSU_USE_WIKI_RENDERER` is ignored.
- No osu! game assets are bundled; osu!'s default art/skins/audio are CC BY-NC and
  are not included. This repo ships only original or procedurally-generated art.
- HUD: score/combo/accuracy/PP via digit sprites; HP bar, progress bar, and
  unstable-rate (UR) meter. Optional live PP counter and results card; official
  PP / star-rating can be injected with `--pp` / `--sr` (otherwise estimated via
  the soft `rosu_pp_py` dependency).
- Can convert std/taiko/ctb beatmaps to mania with `--allow-converted`
  `--convert-to-keys` (rough reproduction of the in-game converter).

## Usage

Installed as the `osu-renderer` console script (entry point
`osu_mania_renderer_v2.cli:main`):

```bash
osu-renderer play.osr ./beatmap/ -o out.mp4 --resolution 1920x1080 --fps 60
```

Positional args: `osr` (the `.osr` file) and `beatmap_dir` (directory containing the
`.osu`, audio, and background). Selected flags (see `cli.py` for the full set):

```
-o, --output PATH            output MP4 (default: out.mp4)
--resolution WxH             e.g. 1280x720 (default 1920x1080)
--fps N                      (default 60)
--encoder {auto,h264_vaapi,h264_nvenc,libx264}
--encoder-device PATH        VAAPI device, e.g. /dev/dri/renderD128
--timeout SECONDS            render timeout (default 600)
--skin-dir PATH              extracted .osk dir (overrides bundled sprites;
                             omitted means the empty/default fallback skin)
--scroll-speed 1-40          --bg-dim / --bg-dim-{intro,game,breaks} / --bg-blur
--pp FLOAT / --sr FLOAT      exact official PP / star rating for the results card
--allow-converted            --convert-to-keys {4,5,6,7,8,9,10}
--show-pp  --logo  --watermark TEXT  --featured-avatar-png PATH
--no-hp-bar --no-ur --no-progress --no-score --no-grade --no-key-overlay
--no-key-counter --no-result-screen --no-loudnorm ... (many HUD/audio toggles)
```

Library entry point (async):

```python
from pathlib import Path
from osu_mania_renderer_v2 import RenderOptions, render_mania

await render_mania(
    osr_path=Path("play.osr"),
    beatmap_dir=Path("./beatmap/"),
    output_path=Path("out.mp4"),
    options=RenderOptions(resolution=(1920, 1080), fps=60),
)
```

## Requirements

- Python **>=3.12**
- Runtime deps (`pyproject.toml`): `moderngl>=5.10`, `osrparse>=7.0`, `Pillow>=10.0`,
  `numpy>=2.0`
- `ffmpeg` on `$PATH` (libx264; VAAPI/NVENC optional for hardware encoding)
- A working EGL/GPU stack for the headless ModernGL context
- Optional: `rosu_pp_py` for PP / star-rating estimation (imported lazily; PP/SR fall
  back to 0 with a warning if absent — not listed in `pyproject.toml`)
- Dev extras (`.[dev]`): `pytest`, `pytest-asyncio`, `pytest-mock`, `ruff`

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Layout

```
osu_mania_renderer_v2/       # the package
  __init__.py                # public render_mania() API
  cli.py                     # osu-renderer CLI (argparse)
  wiki_renderer.py           # worker-facing compatibility shim
  gpu/                       # ModernGL/EGL: context, atlas, primitives, shaders, text
  render/                    # canonical compositor/pipeline + state/encode support
  hud/ argon/ skin/          # registered elements and skin-aware presentation
  beatmap/                   # beatmap/replay parse, mods, judgments, scoring,
                             #   converters, skin.ini, pp
  errors.py                  # renderer exception types
  assets/                    # bundled sprites, shaders, default hitsounds, logo
docs/                        # NAVIGATION.md (code tour), skinning plan, specs
scripts/                     # sprite/skin generators, wiki bootstrap
tests/
pyproject.toml
```

(There are `*.pre_recovery_bak` files and a `_v2fix_backup_*` directory left in the
tree — recovery cruft, not part of the package. Verify before relying on either.)

## License

**AGPL-3.0-or-later** — see `LICENSE` and `COPYRIGHT` (© 2026 Cool Adults).

Note: `pyproject.toml` still declares `license = "MIT"`, which contradicts
`LICENSE`/`COPYRIGHT` — treat AGPL-3.0 as authoritative and fix the metadata (verify).

Attribution: gameplay/scoring/HUD logic ported from ppy's osu! / osu-framework (MIT);
danser-go (GPL-3.0) was studied as a behavioural reference. Any osu! skin you supply
carries its own license — check before redistributing.
