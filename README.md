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
- Custom **skin pipeline** with two paths (see `__init__.py`):
  - **GPU renderer** (default) — hand-coded painter, sprite atlas built once at
    startup from an `.osk` + skin.ini + bundled fallback sprites.
  - **Wiki-driven renderer** (`wiki_renderer.py`, opt-in via
    `OSU_USE_WIKI_RENDERER=1`) — every visible pixel traces to a user-skin asset,
    a default-skin asset, or a wiki-documented default; unresolved variables raise
    rather than guess. This path **requires** `--skin-dir`. Still under active
    development (verify current prod path — the env toggle defaults to the GPU
    renderer).
- Bundled `assets/classic_mania/` sprites and the classic stage-light fallback
  are official osu! default resources by ppy, with separate upstream CC BY-NC
  terms. They are not R3D-original or MIT art. `COPYRIGHT` attributes them;
  `assets/classic_mania/provenance.json` records archive paths and byte hashes.
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
                             required for the wiki renderer)
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

## Replay audio

Replay hitsounds are enabled by default and require the declared `soundfile`
runtime dependency. Missing/broken audio dependencies fail clearly. Conventional sample bank 0 skips the beatmap; bank 1 uses its base filename;
bank 2+ uses only its numbered filename. Eligible beatmap samples fall back to
unnumbered selected-skin samples with `--skin-hitsounds`, then bundled defaults. `--no-beatmap-hitsounds` removes beatmap files from this lookup.
`--hitsound-volume` applies the final user gain. WAV output streams in bounded
chunks, including overlapping samples and NC drums. Legacy audio names are
case-insensitive, and custom paths stay within their permitted source folder.
Before PCM decoding, each sample is checked against a 64 MiB estimate covering
decoded frames, stereo expansion and resampling temporaries; oversized or invalid
candidates fall back to the next source. This is separate from the 32 MiB retained
sample cache and does not cap total process memory or overlapping sample tails.

Known audio fidelity limits: client positional stereo balance and stable
`SamplesMatchPlaybackRate` waveform changes are not implemented. Mismatched sample
rates still use nearest-neighbour conversion without an anti-alias filter.
Gameplay event timing and sample lookup tests do not imply exact client PCM parity.

Both `--no-combo-break` and its compatibility control `--no-miss-hitsound` disable
combo-break SFX. `--combo-break-threshold N` plays them when old combo is strictly
greater than N (default 20). Lazer also plays its first nonzero-to-zero break,
following its default client setting; that first-break rule is independent of N.
Stable Relax/Autopilot paths suppress combo-break audio. Visual combo effects are
independent of these audio controls.

Set `R3D_PREVIEW_INLINE=1` to write a 720p30 `.embed.mp4` alongside the master in
the same ffmpeg process. The preview includes the final song/hits mix; the master
keeps its existing audio and video behavior.

## Requirements

- Python **>=3.12**
- Runtime deps (`pyproject.toml`): `moderngl>=5.10`, `osrparse>=7.0`, `Pillow>=10.0`,
  `numpy>=2.0`, `soundfile>=0.12` (sample decoder/WAV writer; platform wheels
  normally bundle libsndfile, source installs may need system libsndfile)
- `ffmpeg` on `$PATH` (libx264; VAAPI/NVENC optional for hardware encoding)
- A working OpenGL stack: EGL on Linux, platform standalone context on Windows/macOS
- Optional: `rosu_pp_py` for PP / star-rating estimation (imported lazily; PP/SR fall
  back to 0 with a warning if absent — not listed in `pyproject.toml`)
- Dev extras (`.[dev]`): `pytest`, `pytest-asyncio`, `pytest-mock`, `ruff`, `build`

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Layout

```
osu_mania_renderer_v2/       # the package
  __init__.py                # render_mania() + GPU/wiki renderer switch
  cli.py                     # osu-renderer CLI (argparse)
  wiki_renderer.py           # wiki-driven render path
  wiki_elements/             # per-element specs (notes, stage, hud, effects…)
  gpu/                       # ModernGL/EGL: context, atlas, renderer, shaders, text
  render/                    # scene, encode, hitsounds, loudnorm, bg, logo
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

Package metadata and bundled `LICENSE`/`COPYRIGHT` declare AGPL-3.0-or-later for
the renderer code. Upstream resources retain their separately attributed licences.

Attribution: gameplay/scoring/HUD logic ported from ppy's osu! / osu-framework (MIT);
danser-go (GPL-3.0) was studied as a behavioural reference. Any osu! skin you supply
carries its own license — check before redistributing. The bundled official
classic Mania resources and their separate provenance are listed in `COPYRIGHT`.
