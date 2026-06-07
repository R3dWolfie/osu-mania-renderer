# Navigation — Code Tour for New Contributors

This is a map of the repo aimed at someone diving in cold, especially
for the **custom skin pipeline** (since that's where most active work
is right now). Read top to bottom or jump to the section that matches
what you're about to touch.

## High-level pipeline

```
.osr replay  ──┐
                ├──►  scene state per frame  ──►  GPU renderer  ──►  framebuffer
.osu beatmap  ─┘                                       │
                                                      pulls
                                                       ▼
                                            ┌─── skin.ini (skin_ini.py)
                                            ├─── sprite atlas (gpu/atlas.py)
                                            └─── bundled fallback assets
```

Per-frame rendering reads from `SceneState` (what's on screen this
millisecond) and the `SpriteAtlas` (which texture layer represents
each role). The atlas was built once at startup from the skin's
`.osk` directory + skin.ini parsed values + the bundled fallback set.

## Top-level layout

```
docs/                                # design specs, planning docs, this file
osu_renderer/
  ├── __init__.py                    # public API: render_mania()
  ├── cli.py                         # argparse entry point — `osu-mania-renderer`
  ├── render.py                      # orchestrator — wires beatmap + replay + GPU + encoder
  ├── beatmap.py                     # .osu parser, timing, SV (slider velocity) tables
  ├── replay.py                      # .osr parser, mod decoding
  ├── converter.py                   # std/taiko/ctb → mania converter (for cross-mode replays)
  ├── mods.py                        # mod bitfield → behavior dispatch
  ├── judgments.py                   # hit window logic, OD-derived timing gates
  ├── scene.py                       # per-frame SceneState builder
  ├── skin_ini.py                    # [Mania] section parser ★ skin pipeline
  ├── hitsounds.py                   # audio bake-in (per-hit ogg/wav samples)
  ├── pp.py                          # PP estimate (rosu-pp shim, optional)
  ├── encode.py                      # ffmpeg subprocess driver (libx264 / VAAPI / NVENC)
  ├── errors.py                      # typed exceptions
  ├── models.py                      # dataclasses shared across modules
  ├── gpu/
  │   ├── __init__.py
  │   ├── context.py                 # EGL/headless ModernGL setup
  │   ├── shaders.py                 # GLSL programs (sprite, sprite_instanced, flashlight)
  │   ├── atlas.py                   # SpriteAtlas ★★ skin pipeline core
  │   ├── renderer.py                # FrameRenderer ★★ per-frame draw — receptors/notes/HUD
  │   ├── readback.py                # FBO → CPU memory for ffmpeg pipe
  │   └── text.py                    # font glyph rasterisation
  └── assets/
      ├── shaders/                   # .vert / .frag
      └── sprites/                   # bundled fallback PNGs (used when skin doesn't ship one)
scripts/
  ├── generate_placeholder_sprites.py  # one-shot bootstrap of assets/sprites/
  ├── extract_night05_sprites.py        # rebuilds bundled defaults from a reference .osk
  └── parse_smoke.py                    # CI-grade smoke check for skin.ini parser
tests/                                  # pytest; some need a GPU (RUN_SLOW.md)
```

★ = touched often when working on skins.
★★ = where the skin pipeline LIVES.

## The skin pipeline — what to read first

Going from a `.osk` upload to "pixels on screen", the chain is:

1. **`skin_ini.py`** — parses the `[Mania]` block(s) of `skin.ini`.
   Each `[Mania] Keys: N` section becomes its own `ManiaSection`
   dataclass (column widths, HitPosition, NoteImage paths,
   ColumnLineWidth, NoteBodyStyle, Colour1..N, etc.). Multiple
   sections coexist — a skin with `Keys: 4` + `Keys: 7` ships two
   separate configurations.

2. **`gpu/atlas.py`** — `SpriteAtlas.load(...)` walks every sprite
   slot the renderer cares about (note_tap, note_hold_body, receptor_off,
   stage_left, judgment_300, etc.) and resolves each through the
   priority chain:
   ```
   beatmap_dir  →  skin's explicit skin.ini path override  →
                   skin's conventional filename  →
                   bundled fallback in assets/sprites/
   ```
   `@2x.png` variants are tried first, then `.png`. Per-column slots
   (`note_tap`, `receptor_off`, etc.) live at `(kind, column)` keys;
   global slots (`stage_left`, `judgment_300`) live at `name`.
   Native dimensions + animation frame counts are captured before
   the sprite is letterboxed into a fixed atlas layer.

3. **`gpu/renderer.py`** — `FrameRenderer` does the per-frame
   drawing. `_compute_playfield_geometry()` resolves column widths
   from skin.ini (or lazer defaults). `_draw_columns()` paints
   `Colour{N}` lane backgrounds + `ColumnLineWidth` dividers.
   `_draw_stage_decorations()` paints the side dim + stage chrome.
   `_draw_receptors()` reads each column's native receptor aspect
   and bottom-anchors at the hit line. `_draw_notes()` does the same
   for tap notes + hold heads/tails + the L-sprite body
   (stretched at NoteBodyStyle 0, tiled at 2/3/4).

The fundamental rule: **skin-agnostic**. One algorithm reads
sprite metrics + skin.ini values + lazer fallbacks and applies them
uniformly. Never branch on a sprite filename or skin name. See
`docs/superpowers/specs/2026-05-26-skin-fidelity-overhaul.md` for
the detailed spec from the most recent overhaul.

## What "lazer-faithful" means here

The renderer's mania geometry follows **ppy/osu's Legacy* classes**
(found in https://github.com/ppy/osu under `osu.Game.Rulesets.Mania/Skinning/Legacy/`):

- `LegacyKeyArea.cs` — receptor anchoring (BottomCentre on downscroll, X locked to column width, height = native sprite height)
- `LegacyNotePiece.cs` — tap/head/tail anchoring (same)
- `LegacyBodyPiece.cs` — hold body `NoteBodyStyle` semantics (0 = stretch+clamp, 2/3/4 = tile with repeat)
- `LegacyManiaSkinConfiguration.cs` — defaults (`DEFAULT_COLUMN_SIZE = 30` in stable 480-coord, `DEFAULT_HIT_POSITION = 402`)

The Python implementation isn't a 1:1 port — different coord system,
different draw primitives — but the **anchoring rules** and
**aspect-derivation formulas** match. When in doubt, the C# reference
is authoritative.

## Running it

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_placeholder_sprites.py    # one-shot bundled-asset setup
osu-mania-renderer ./play.osr ./beatmap/ -o out.mp4 \
  --resolution 1280x720 --fps 30 \
  --skin-dir /path/to/extracted/skin/
```

`--skin-dir` is optional; without it the renderer uses bundled
defaults (which is a Night05/Corne2Plum3 import). Mods are read
from the `.osr` automatically.

GPU required (EGL-capable). VAAPI (Intel/AMD) or NVENC (NVIDIA) makes
encoding ~10× faster but `libx264` software fallback works on CPU.

## Where to test changes

1. `scripts/parse_smoke.py` — quick check that skin.ini parsing
   doesn't choke on any of the skins under `R3D_SKIN_ROOT`. Run
   before deploying.
2. `tests/test_gpu_atlas.py` — atlas resolution + per-column lookup.
3. `tests/test_gpu_renderer_*` — pytest-based render output checks
   for specific HUD/playfield/background concerns.
4. Visual smoke test — render a `.osr` you know the expected look
   of, eyeball a mid-game frame. Reference frames + replays for
   recent overhauls live at
   `/var/mnt/Synology-Reddie/R3DRenderer testing/skin-overhaul-2026-05-26/`
   on R3D's NAS.

## When skins look wrong

Triage in this order:
1. Check the `.osk`'s `skin.ini` for the relevant `[Mania] Keys: N`
   section. Does it define the field you expect?
2. Render with `--skin-dir <skin>` and watch stderr for
   `atlas_loaded` NDJSON line — it reports per-column source
   (`user` / `beatmap` / `bundle` / `missing`) and aspect.
3. If atlas says the sprite resolved but the visual is wrong, the
   bug is in `gpu/renderer.py` — anchoring or scaling.
4. If atlas says `missing` when the file exists, the bug is in
   `gpu/atlas.py` — resolution chain or path mapping.

Never patch a single skin's behavior. If a fix only works for "the
Pii AR11 case" it's not done.

## Where the design history lives

- `docs/SKINNING_PLAN.md` — original taxonomy of mania skin assets
  and how each maps to renderer slots.
- `docs/superpowers/specs/2026-05-26-skin-fidelity-overhaul.md` —
  the 6-phase overhaul spec (playfield geometry, stage chrome,
  receptor anchoring, note aspect, NoteBodyStyle, BG dimming).
  Includes failure-mode notes from a botched first attempt.
- `git log --oneline -- osu_renderer/gpu/renderer.py` —
  per-commit history of renderer changes; commits prefixed
  `Phase N:` track the overhaul.
