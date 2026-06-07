# osu! Mania Renderer — Design

**Status:** Draft for review
**Date:** 2026-05-16
**Owner:** R3dWolfie

## Goal

A standalone Python library that takes an osu!mania `.osr` replay plus the
beatmap files and produces an MP4 of the play, with the beatmap's audio mixed
in. GPU-accelerated rendering via ModernGL (OpenGL 3.3 standalone context),
hardware-encoded via ffmpeg + VAAPI. Designed to be drop-in usable by the
existing **Mania ORDR** Discord bot but with no Discord/bot coupling — the
renderer ships its own CLI and is intended for public release on GitHub later.

## Non-goals

- Rendering other osu! modes (std/taiko/ctb). The renderer rejects them.
- Parsing arbitrary user `.osk` skins. The bundled default visual style is
  fixed at build time. Custom-skin support is a stretch goal.
- Driving a real-time gameplay session. This is offline replay rendering only.
- Producing pixel-perfect parity with osu!stable or osu!lazer. The output
  should be unambiguously recognizable as mania; exact visual parity is not
  the bar.

## Scope (v1)

**In:**

- `.osr` replay parsing (mania-only).
- `.osu` beatmap parsing (mania `[HitObjects]` only).
- Mod support: NF, EZ (no visual change), HD (notes fade out near receptors),
  HR (no visual change), DT/NC (1.5×), HT (0.75×), FL (flashlight overlay), FI
  (notes fade in from above), MR (mirror columns), V2 (ScoreV2 scoring),
  key-count locks (1K–9K), PF/SD (no visual change).
- **RD (Random)** is explicitly *not* supported. RD replays render as NM with
  a warning. (Implementing osu!stable's deterministic shuffle requires
  reimplementing .NET `System.Random`; cost outweighs benefit.)
- KC (Key Coop) is *not* supported in v1; renders as NM with a warning.
- Beatmap audio mixed in with mod-correct speed + pitch (DT pitches up, HT
  down — `ffmpeg asetrate` semantics matching osu!stable).
- Beatmap background image, dimmed ~50%.
- HUD: score, accuracy, current combo, max combo, judgment counts.
- Banner: artist - title \[difficulty\] · player.
- Per-frame progress callback for the consuming application.
- CLI: `osu-mania-renderer in.osr beatmap_dir/ -o out.mp4`.

**Out (stretch / future):**

- Parsing user `.osk` skins.
- Hit-sound sample synthesis.
- Storyboard rendering.
- RD and KC mod visualization.
- Replay editor / timestamp seeking.

## Default visual style

Visuals are based on the **Night05 v1.1 osu!mania skin by Corne2Plum3**
(`- Night05 - (v1.1) [- Night05 ⟦NM1⟧ - (v1.1)] (Corne2Plum3).osk`). At
build/install time, the build script extracts the .osk and copies the relevant
PNGs into `osu_renderer/assets/`. Any sprite Night05 doesn't ship is
filled with a fallback we author. We do **not** ship `skin.ini` parsing; the
column layout, sizing, and positioning are hardcoded based on what Night05
looks like.

**Redistribution caveat:** Night05's license needs to be checked before this
repo goes public on GitHub. If unclear, ship freely-licensed default sprites
and document Night05 as an optional drop-in replacement.

## High-level architecture

```
mania-ordr (existing Discord bot)
        │
        │ pip install -e ../Reddie/OsuManiaRenderer
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  osu_renderer  (this package)                              │
│                                                                  │
│  Public API:                                                     │
│    async render_mania(                                           │
│        osr_path: Path,                                           │
│        beatmap_dir: Path,                                        │
│        output_path: Path,                                        │
│        options: RenderOptions,                                   │
│        progress_callback: Callable[[float], Awaitable[None]]     │
│            | None = None,                                        │
│        log_path: Path | None = None,                             │
│    ) -> None                                                     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐     │
│  │ beatmap.py   │  │ replay.py    │  │ mods.py             │     │
│  │ parse .osu   │  │ decode .osr  │  │ DT/HT/MR/HD/FI/FL/  │     │
│  │ notes +      │  │ keypress     │  │ key locks/V2 →      │     │
│  │ timing       │  │ timeline     │  │ apply to notes/audio│     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────────────┘     │
│         └─────────────────┼─────────────────┘                    │
│                           ▼                                      │
│                  ┌──────────────────┐                            │
│                  │ scene.py         │                            │
│                  │ per-time snapshot│                            │
│                  │ of visible state │                            │
│                  └────────┬─────────┘                            │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────┐        │
│  │ gpu/                                                 │        │
│  │   context.py    standalone moderngl context          │        │
│  │   atlas.py      bundled sprite PNGs → GL textures    │        │
│  │   shaders.py    GLSL: sprite + flashlight post-pass  │        │
│  │   renderer.py   draw bg/playfield/notes/HUD/banner   │        │
│  │   readback.py   FBO → CPU bytes (PBO double-buffer)  │        │
│  └────────────────────────────┬─────────────────────────┘        │
│                               ▼                                  │
│  ┌──────────────────────────────────────────────────────┐        │
│  │ encode.py                                            │        │
│  │   ffmpeg -hwaccel vaapi -c:v h264_vaapi …            │        │
│  │   raw frames in via stdin                            │        │
│  │   audio.mp3 mixed via asetrate (mod speed + pitch)   │        │
│  └──────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

**Key separations:**

- `beatmap` / `replay` / `mods` are pure-Python and CPU-only. They produce
  frame-independent descriptions of "what happens when." Tested without a GPU.
- `scene` is the per-frame query layer: pure function of beatmap + replay +
  mods + time.
- `gpu/` is the only module that touches OpenGL. Everything above is
  GPU-agnostic.
- `encode` owns ffmpeg; nothing else spawns subprocesses.

## Components

Files inside `osu_renderer/`, each with one responsibility:

| Module | Responsibility | LOC est. |
|---|---|---|
| `__init__.py` | Re-exports public API (`render_mania`, `RenderOptions`, `Mod`, errors) | ~30 |
| `models.py` | Pure dataclasses: `Note`, `HoldNote`, `Judgment`, `KeyEvent`, `RenderOptions`, `BeatmapInfo`, `ReplayInfo`, `VisualMods` | ~100 |
| `beatmap.py` | Parse `.osu` → `BeatmapInfo` (notes, key count, audio filename, bg filename, timing). Mania-only. | ~220 |
| `replay.py` | Parse `.osr` via `osrparse` → `ReplayInfo` with decoded keypress timeline | ~150 |
| `mods.py` | `apply_mods(beatmap, replay) -> (modded_beatmap, audio_rate, visual_mods)` | ~220 |
| `scene.py` | `snapshot(modded_beatmap, replay, t_ms, visual_mods) -> SceneState` | ~220 |
| `gpu/context.py` | EGL headless ModernGL context + offscreen FBO | ~80 |
| `gpu/atlas.py` | Pack bundled PNGs into a texture array | ~120 |
| `gpu/shaders.py` | GLSL sprite shader + flashlight post-pass | ~150 |
| `gpu/renderer.py` | `draw_frame(scene_state, options)` — issues all GL draw calls | ~320 |
| `gpu/readback.py` | PBO double-buffered framebuffer readback to raw RGB bytes | ~80 |
| `encode.py` | Spawn ffmpeg, stream video, mux audio | ~170 |
| `errors.py` | Typed exception hierarchy rooted at `RendererError` | ~40 |
| `cli.py` | `osu-mania-renderer in.osr beatmap_dir/ -o out.mp4 [opts]` | ~100 |
| `assets/` | Bundled PNG sprites + GLSL shader sources + bitmap font | (~30 files) |

Total ~2000 LOC.

## Public API (entire surface)

```python
from pathlib import Path
from typing import Awaitable, Callable

from osu_renderer import (
    RenderOptions,
    render_mania,
    NotAManiaError,
    BeatmapParseError,
    ReplayParseError,
    GpuUnavailableError,
    EncoderError,
    RenderTimeoutError,
)


options = RenderOptions(
    resolution=(1920, 1080),
    fps=60,
    encoder="auto",                # "auto" | "h264_vaapi" | "libx264"
    encoder_device="/dev/dri/renderD128",  # optional VAAPI device
    timeout_seconds=600,
    audio_required=False,          # if True, missing audio raises MissingAudioError
)

async def on_progress(fraction: float) -> None:
    print(f"{fraction:.0%}")

await render_mania(
    osr_path=Path("play.osr"),
    beatmap_dir=Path("/cache/beatmaps/<md5>/"),
    output_path=Path("out.mp4"),
    options=options,
    progress_callback=on_progress,
    log_path=Path("/var/log/mania-render.log"),  # optional structured log
)
```

## Data flow for one render

1. **Parse `.osr`** → `ReplayInfo` (mode-checked, mods, judgment counts,
   decoded keypress timeline).
2. **Parse `.osu`** → `BeatmapInfo` (notes, key count, audio/bg filenames,
   base scroll velocity from BPM, total duration).
3. **Reject non-mania** — if either is mode ≠ 3, raise `NotAManiaError`
   immediately.
4. **Apply mods** → `mods.apply(beatmap, replay)`:
   - `modded_notes`: MR-flipped, key-count-aware.
   - `audio_rate`: 1.0 / 1.5 (DT/NC) / 0.75 (HT).
   - `visual_mods`: HD / FI / FL flags consumed by the GPU layer.
   - `total_duration_ms`: post-mod.
5. **Bootstrap GPU** → `gpu.context.headless_gl(W, H)` opens an EGL
   standalone context + offscreen FBO. ~50 ms.
6. **Load assets** — `gpu.atlas.load_default()` uploads sprite PNGs and GLSL
   shaders. ~200 ms.
7. **Load background** — if `beatmap_dir / <bg_filename>` exists, upload as a
   separate texture; dim ~50 % via the sprite shader's tint uniform.
8. **Spawn ffmpeg** — `encode.start(output_path, options, audio_path,
   audio_rate)`. ffmpeg waits for raw RGB frames on stdin, pulls audio
   independently from the file.
9. **Render loop** — for `frame_n in 0 .. ceil(total_duration_ms / 1000 *
   fps)`:
   - `t_ms = frame_n * 1000 / fps`
   - `scene = scene.snapshot(modded_notes, replay_keys, t_ms, visual_mods)`
     (CPU, ~0.5 ms)
   - `gpu.renderer.draw(scene)` (GPU, ~2 ms)
   - `gpu.readback.read_into(ffmpeg_stdin)` (PBO-overlapped, ~5–10 ms total
     per frame)
   - every ~0.5 s of rendered video AND ~0.5 s wall-clock, call
     `progress_callback(frame_n / total_frames)`.
10. **Finalize** — close ffmpeg stdin, wait for it to drain. Tear down the GL
    context.
11. **Verify** — assert the MP4 exists and is non-empty. If `ffmpeg` exit code
    ≠ 0, raise `EncoderError` with the last 1 KB of its stderr.

**Cancellation:** `render_mania` is `async`. `asyncio.CancelledError`
propagates through the render loop; ffmpeg gets SIGTERM; GL context is torn
down in a `finally` block.

## Mod implementation

| Mod | Bit | Effect | Implementation |
|---|---|---|---|
| NF | 0 | No Fail | Ignored (no visual effect). |
| EZ | 1 | Easy | Ignored. |
| HD | 3 | Hidden | Notes fade out at ~60 % of playfield height down, alpha-over-Y in fragment shader gated by `u_hd`. |
| HR | 4 | Hard Rock | Ignored in mania. |
| SD/PF | 5/14 | Sudden Death / Perfect | Ignored (scoring only). |
| DT/NC | 6/9 | Double / Nightcore | `audio_rate = 1.5`. Note times pre-divided by 1.5 in `apply_mods`. NC's click track is *not* synthesized. |
| HT | 8 | Half Time | `audio_rate = 0.75`. Same path. |
| FL | 10 | Flashlight | Post-pass shader masks everything outside a circular spotlight around the receptors. Gated by `u_fl`. |
| FI | 20 | Fade In | Inverse of HD — notes fade *in* from above. Same shader, inverted gradient. |
| RD | 21 | Random | **Not implemented in v1.** RD replays render as NM with a warning. |
| 1K–9K | 26,28,27,15,16,17,18,19,24 | Key-count override | Assert that beatmap's key count matches; reject mismatches. |
| MR | 30 | Mirror | Trivial: `new_col = (key_count - 1) - old_col`. |
| KC | 25 | Key Coop | **Not implemented in v1.** Renders as NM with a warning. |
| V2 | 29 | ScoreV2 | Score HUD uses the V2 formula instead of legacy. No visual effect on the playfield. |

## Audio handling

Replays don't contain audio. The renderer mixes the beatmap's `audio.mp3`
(referenced by `[General]/AudioFilename`) into the MP4 via ffmpeg.

**Timing alignment.** The beatmap's `[General]` section has `AudioLeadIn`
(silent pad before audio starts, ms). Note times in `.osu` are measured from
"moment audio playback begins." Frame 0 of the video = time 0 of the .osu =
moment audio starts (after lead-in silence). The renderer covers
`t = -lead_in_ms .. total_duration_ms`; ffmpeg delays the audio stream by
`lead_in_ms` so it lines up.

**Speed mods.** DT/NC = 1.5×, HT = 0.75×. Implemented via ffmpeg's
`asetrate=44100*<rate>,aresample=44100` filter — this changes the playback
rate *and* pitch, matching osu!stable. The video side gets pre-divided note
times so visual scroll stays in sync.

**Total video length** = `(max_note_end_time / audio_rate) + 2000 ms` end pad
+ `lead_in_ms` start pad.

**Failure modes.** Missing audio file → render silent video with a `WARNING`
log line by default (don't fail — some storyboard-only maps have no audio).
If `RenderOptions.audio_required=True`, raise `MissingAudioError` instead.
Unrecognized audio codec → ffmpeg surfaces the error in stderr, we propagate
as `EncoderError`.

**Encoder selection.** `encoder="auto"` probes for VAAPI at startup
(`ffmpeg -encoders | grep h264_vaapi` and a brief test-init against
`encoder_device`); falls back to `libx264` if VAAPI isn't available. Explicit
`"h264_vaapi"` or `"libx264"` skips the probe and uses what's specified
(raising `EncoderError` if it can't be loaded).

**Sample ffmpeg invocation:**

```
ffmpeg -y \
  -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 \
  -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 60 -i pipe:0 \
  -i "<beatmap>/audio.mp3" \
  -filter:a "asetrate=44100*<rate>,aresample=44100,adelay=<lead_in>|<lead_in>" \
  -vf "format=nv12,hwupload" \
  -c:v h264_vaapi -b:v 5M \
  -c:a aac -b:a 192k \
  -map 0:v -map 1:a \
  -shortest \
  out.mp4
```

## Visual style & bundled assets

**Layout (1920×1080 canvas).** Playfield is fixed-width (~440 px), horizontally
centered. Column width = 440 / key_count. Notes scroll top→bottom at a
constant velocity tuned to give ~600 ms approach time at osu!stable's default
scroll speed 20.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ artist - title [diff]    player                            R3D ORDR      │
├──────────────────────────────────────────────────────────────────────────┤
│            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                       │
│            ░  background (50% dim)              ░                        │
│            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                       │
│         ┌──────────────────────────┐                Score 865,612        │
│         │█████  notes scroll       │                  98.45 %            │
│         │  ░░                      │                Combo 1305 x         │
│         │     ░░                   │                Max   1305 x         │
│         │                          │           ┌─────────────────┐       │
│         │            PERFECT       │           │ Judgments       │       │
│         │           ░░             │           │  320  732       │       │
│         │█▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░█  │ ← receptors │  300  633     │       │
│         └──────────────────────────┘           │  200  113       │       │
│           ↑↑↑↑ key flash on press              │  100   10       │       │
│                                                │   50    1       │       │
│                                                │ MISS    2       │       │
│                                                └─────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

**Sprites** (all PNG, power-of-two dimensions, pre-multiplied alpha; sourced
from Night05 unless noted):

| File | Use | Size |
|---|---|---|
| `note_tap.png` | Solid tap note | 128×32 |
| `note_hold_head.png` / `_body.png` / `_tail.png` | 3-piece hold (head, tiled body, tail) | 128×32 each |
| `receptor_off.png` / `_on.png` | Bottom receptor; "on" glows when held | 128×64 |
| `hit_light.png` | Additive splash on hit | 256×128 |
| `judgment_geki.png` / `_300.png` / `_katu.png` / `_100.png` / `_50.png` / `_miss.png` | Color-coded judgment popups | 256×64 each |
| `column_bg.png` | Faint column background (tiled) | 128×16 |
| `font_atlas.png` + `font.json` | Bitmap font for HUD + banner | 1024×1024 |
| `playfield_frame.png` | 9-slice border around columns | — |
| `bg_vignette.png` | Soft vignette overlay on the background | 1024×1024 |

**Color palette.** Mania-canonical with Night05's tones:

- 4K outer: cool blue · inner: white.
- 5K: yellow center, surrounded by white/blue.
- 6K: alternating white/blue.
- 7K: white/blue alternating with yellow center.
- Judgments: gold (geki/320) · blue (300) · green (katu/200) · yellow (100) ·
  grey (50) · red (miss).

**Animations.**

- Receptor "on" sprite cross-fades over 80 ms on press/release.
- Hit light scales 0.7 → 1.2 and fades over 200 ms.
- Judgment text holds at scale 1.0 for 400 ms, fades over 200 ms.
- Combo number pulse-scales +10 % for 80 ms on every hit.

## Error handling, logging, testing

**Errors.** Typed hierarchy rooted at `RendererError(Exception)`:

| Exception | When | Consumer's recovery |
|---|---|---|
| `BeatmapParseError` | `.osu` malformed | Surface to user |
| `ReplayParseError` | `.osr` malformed | Surface to user |
| `NotAManiaError(mode_int)` | Either file is mode ≠ 3 | Embed: "Not a mania play" |
| `MissingAudioError` | Audio file missing AND `RenderOptions.audio_required = True` | Optional — defaults to silent + warn |
| `GpuUnavailableError` | EGL/GL init failed | Embed: "GPU unavailable" + log full error |
| `EncoderError` | ffmpeg exits non-zero / output empty | Embed: includes tail of ffmpeg stderr |
| `RenderTimeoutError` | Render exceeds `options.timeout_seconds` | Embed: "Render timed out" |

**Logging.** Logger name: `osu_renderer`. Standard stdlib `logging`; no
`structlog` coupling. Levels:

- `INFO` — stage transitions ("parsing beatmap", "starting ffmpeg", "render
  complete").
- `WARNING` — soft-degrades (missing audio → silent, missing bg, unknown mod
  ignored, RD/KC → NM fallback).
- `ERROR` — the exceptions above (before they propagate).
- No `DEBUG` spam.

**Progress callback.** Throttled to at most once per ~0.5 s of rendered video
*and* ~0.5 s wall-clock, so a fast render doesn't blow the consumer's edit
budget.

**Testing.**

| Layer | Test | Slow? |
|---|---|---|
| `beatmap.py` | Parse 5 real `.osu` files (4K / 5K / 6K / 7K / hold-heavy). Assert note counts, key counts, audio filenames. | no |
| `replay.py` | Parse the AO-INFINITY mania replay + a std replay. Assert mania-only check rejects std. | no |
| `mods.py` | DT: `audio_rate == 1.5`. HT: 0.75. MR: column flip. RD: silent NM fallback + warning. KC: same. | no |
| `scene.py` | Hand-built tiny beatmap; assert which notes are visible and their Y positions at known `t`. | no |
| `gpu/` | One smoke test: open context, draw known pattern, read back, compare pixels. Works on AMD/Intel/llvmpipe. | yes |
| `encode.py` | Mock ffmpeg, assert command-line shape. One real test pipes 60 frames to real ffmpeg, confirms non-empty MP4. | yes |
| End-to-end | CLI renders a fixture replay → MP4 → `ffprobe` confirms duration, codec, resolution, dual streams. | yes |

Slow tests are gated `RUN_SLOW=1`. Unit tests run in <2 s.

## Bot integration

The existing Mania ORDR bot's `mania_ordr/renderer.py` is rewritten as a
~50-line shim around `render_mania`. The bot's `Renderer.render` method
signature is preserved so nothing else changes.

**Deltas in the bot during this pivot:**

- Drop `DANSER_BINARY` env var from `Settings`.
- Drop the `_default-source.osk` staging step from the install docs.
- Drop the renderer's call to `skin_resolver._ensure_default()` (the new
  renderer ships its own assets).
- Keep `skin_resolver`'s register/get/clear flow — dead code today but lines
  up with the "custom skin layer" stretch goal.
- Worker pipeline adds a `"🛠️ Processing…"` placeholder edit between
  rendering finish and the final embed (per user request).
- Worker pipeline updates the placeholder more richly:
  `"🎵 Mania replay detected: <player> · <artist> - <title> [<diff>]"`
  immediately after parse, then `"🎬 Rendering {pct}%"` during, then
  `"🛠️ Processing…"`, then the final embed.

Roughly 50 LOC of renderer rewrite, 5 LOC of config trim, 10 LOC of new
placeholder formatting in the worker.

The integration plan for the bot is a follow-up document after this renderer
ships.

## Open questions

None. All earlier open items have been resolved:

- Polish tier: mid ("looks like osu! mania")
- Tech stack: ModernGL + ffmpeg/VAAPI
- Mod scope: all relevant mods except RD and KC, which warn and fall back
- Audio: beatmap audio mixed in, no hit-sound synthesis
- Background: bundled image, ~50 % dim
- HUD: full (judgments + score + combo + accuracy + max combo)
- Banner: artist - title [diff] · player
- Default skin: Night05 v1.1 by Corne2Plum3 (license check pending public
  release)
- Bot integration: ~50-line shim, no other consumer-facing changes
