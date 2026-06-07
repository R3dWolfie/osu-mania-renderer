# Old GPU Renderer Reference

A comprehensive reference for the hand-coded GPU renderer pipeline (the
"old" path, dispatched when `OSU_USE_WIKI_RENDERER=0`). The wiki-driven
replacement must reproduce every visual element this pipeline produces.

---

## Architecture Overview

```
CLI (cli.py)
  │
  ▼
__init__.py (public API — render_mania)
  │
  ▼
render.py (orchestrator — parse → mod → render → encode)
  │
  ├── beatmap.py       ── .osu file parser (mania-native + std→mania conversion)
  ├── replay.py        ── .osr parser via osrparse
  ├── judgments.py     ── map keypresses → hit judgments (geki/300/katu/100/50/miss)
  ├── mods.py          ── apply DT/HT/MR/HD/FI/FL/V2 mods
  ├── scene.py         ── per-frame visible-state computation
  ├── hitsounds.py     ── pre-mix per-note hitsound WAV + nightcore overlay
  ├── pp.py            ── rosu-pp-py wrapper for live PP
  ├── encode.py        ── ffmpeg subprocess: probe, build cmd, pipe frames
  │
  └── gpu/
      ├── context.py   ── HeadlessGl: EGL context + offscreen FBO
      ├── atlas.py     ── SpriteAtlas: pack PNGs into GL texture array
      ├── renderer.py  ── FrameRenderer: all draw calls (playfield, notes, HUD, bg)
      ├── shaders.py   ── load GLSL programs
      ├── readback.py  ── PBO-ring GPU→CPU frame readback
      └── text.py      ── PIL→GL texture for HUD strings
```

---

## Pipeline Flow (render.py)

### 1. Parse inputs
```
replay = parse_replay(osr_path)          # read .osr via osrparse
osu_file = _find_osu(beatmap_dir, md5)   # find matching .osu by md5
beatmap = parse_beatmap(osu_file, ...)    # parse .osu → BeatmapInfo
```

### 2. Apply mods
```
mod_res = apply_mods(beatmap, replay)     # DT/HT rescales times, MR mirrors, etc.
modded = mod_res.beatmap                  # modded BeatmapInfo
```

### 3. Compute judgments
```
judgments = compute_judgments(notes, key_events, key_count, OD)
  → JudgmentTimeline with per-note JudgmentEvent (time, column, judgment, offset_ms)
```

### 4. Pre-compute data
```
judged_hits     — (col, note_time) → press_time (for hiding attempted notes)
kiai_ranges     — (start_ms, end_ms) windows from timing-point effects
per_column_ur   — final UR per column for results card
sv_table        — cumulative SV distance table (beatmap.build_sv_distance_table)
player_pp, max_pp — rosu-pp values
judgment_timeline — sorted by effective press time for combo/score/acc tracking
```

### 5. GPU render + ffmpeg encode loop
```
with HeadlessGl(w, h) as gl:
    rc = RenderContext(ctx=gl.ctx, fbo=gl.fbo, ...)
    fr = FrameRenderer(rc, options, skin_dir, beatmap_dir, first_note_ms)
    fr.set_background(bg_path)      # load & upload bg image
    fr.set_banner_text(...)         # artist - title [diff] | player
    reader = FrameReader(gl.ctx, gl.fbo, components=3)  # PBO readback

    for frame_n in range(total_frames):
        t_ms = frame_n * 1000 / fps
        scene = snapshot(notes, key_events, t_ms, ...)   # SceneState
        fr.draw(scene)                                      # GPU draw
        frame = reader.read()                               # → CPU bytes
        pipe.write_frame(frame)                             # → ffmpeg stdin
```

### 6. Post-process
```
pipe.close(output_path)  # close stdin, wait for ffmpeg, verify output
```

---

## Module Details

### `models.py` — Data Models

Immutable dataclasses shared across the pipeline:

- **`Note`**: `column`, `time_ms`, `hit_sound`, `hit_sample`
- **`HoldNote`**: `column`, `time_ms`, `end_time_ms`, `hit_sound`, `hit_sample` (+ `duration_ms`)
- **`TimingPoint`**: `time_ms`, `sample_set`, `custom_index`, `volume`, `sv_multiplier`, `uninherited`, `beat_length_ms`
- **`KeyEvent`**: `time_ms`, `keys_held` (bitmask)
- **`BeatmapInfo`**: `key_count`, `notes`, `audio_filename`, `background_filename`, `total_duration_ms`, `audio_lead_in_ms`, `artist`, `title`, `difficulty`, `creator`, `beatmap_id`, `beatmapset_id`, `default_sample_set`, `timing_points`, `overall_difficulty`
- **`ReplayInfo`**: `mode`, `beatmap_md5`, `player_name`, `replay_md5`, `mods`, `key_events`, `score`, `accuracy`, `max_combo`, `count_geki/300/katu/100/50/miss`, `grade`
- **`RenderOptions`**: `resolution`, `fps`, `encoder`, `encoder_device`, `timeout_seconds`, `audio_required`, `video_bitrate`, `audio_bitrate`, plus ~30 visual/audio toggle bools
- **`VisualMods`**: `hidden`, `fade_in`, `flashlight`, `score_v2`
- **`HitSample`**: `normal_set`, `addition_set`, `index`, `volume`, `filename`

### `errors.py` — Exception Hierarchy

All inherit `RendererError`:
- `BeatmapParseError`, `ReplayParseError`, `NotAManiaError`, `MissingAudioError`
- `GpuUnavailableError`, `EncoderError`, `RenderTimeoutError`

### `replay.py` — .osr Parser

Wraps `osrparse.Replay.from_path`. Decodes mania key events from `replay_data`
as `KeyEvent(time_ms, keys_held)` with per-column bitmask. Computes accuracy
using the 305-weighted formula. Raises `NotAManiaError` if mode != 3.

### `beatmap.py` — .osu Parser

Splits the .osu file into sections (`[General]`, `[Metadata]`, `[Difficulty]`,
`[HitObjects]`, `[TimingPoints]`). Parses hit objects with `HoldNote` detection
via bit 7. Timing points parse SV multipliers from negative beatLength values.
When `allow_converted=True` and mode != 3, routes through the std→mania converter.

Also provides:
- **`build_sv_distance_table(tps)`**: Pre-computes cumulative SV-integrated
  distances at each timing-point boundary. O(N) once per render.
- **`sv_distance_at(t_ms, tps, table)`**: Returns cumulative SV-integrated
  distance at any time via binary search. O(log N) per call.

### `judgments.py` — Hit Judgment Engine

Pairs each scoring event (tap head, hold head, hold tail) with the closest
matching key-press or key-release. Uses **lazer-style OD-scaled windows**:
```
320 = 16ms constant
300 = 64 - 3*OD
200 = 97 - 3*OD
100 = 127 - 3*OD
50  = 151 - 3*OD
```
Hold tails get 1.5× wider windows. Falls back to stable fixed windows
(±16.5/40/73/103/127 ms) when OD is unavailable.

Returns `JudgmentTimeline` with `JudgmentEvent` per note, including signed
`hit_offset_ms` for UR computation.

Also exports `compute_consumed_times()`: maps each note to its press time
within ±250ms for hiding attempted notes that scrolled past.

### `mods.py` — Mod Application

`Mod` IntFlag enum with bitmask values. `apply_mods()` returns `ModResult`:
- DT/NC/HT rescales note times by 1.5× / 0.75×
- MR mirrors column layout
- RD/KC emit warnings (not visually applied)
- HD/FI/FL/V2 populate `VisualMods`

`mod_acronyms()` produces display-ordered pill labels like `("4K", "HD", "DT")`.

### `converter.py` — std→mania Conversion

Two-phase operation:
1. **Heuristic**: expand std hit objects into mania notes. Sliders become
   ¼-beat streams or single holds (gated by OD). Position-based column
   assignment with light RNG + stair-walk.
2. **Replay-driven**: match key-press events (±150ms) to the full expanded
   note list to recover the player's actual column choices. This is the
   key fix for slider-heavy converted maps.

### `hit sounds.py` — Hitsound Track Builder

Resolves per-note WAV files by priority:
1. `hit_sample.filename` in beatmap dir
2. `{set}-hit{type}{index}.wav|.ogg` in beatmap dir
3. `{set}-hit{type}.wav|.ogg` (index 0 fallback)
4. Silent

Honours per-timing-point sample set/volume and per-note hit-sound bits
(normal/whistle/clap/finish). Builds a stereo float32 WAV at 44.1 kHz,
mixes combo-break sounds on large combo breaks, and optionally layers
nightcore claps + finishes on each beat.

### `pp.py` — Performance Points

Wraps `rosu_pp_py`. Called once per render with the `.osu` path and replay
judgment counts. Returns `(player_pp, max_fc_pp)` or `(0.0, 0.0)` if
rosu-pp is not installed or fails.

### `encode.py` — ffmpeg Subprocess

- **`probe_encoder()`**: Auto-selects h264_vaapi > libx264 > libopenh264.
- **`build_ffmpeg_cmd()`**: Constructs ffmpeg argv with rawvideo input,
  optional audio (song + hitsound mix via filter_complex), vflip, YUV
  conversion, BT.709 colour tags, +faststart moov atom.
- **`FfmpegPipe`**: Async subprocess manager. Supports FIFO mode for
  flatpak-spawn host-ffmpeg path. Uses `asyncio.create_subprocess_exec`.

### `scene.py` — Per-Frame State

`snapshot()` returns a `SceneState` containing everything visible at time `t_ms`:
- `visible_notes`: list of `VisibleNote` (column, y_fraction, is_hold, head/tail positions)
- `keys_held`: per-column bool tuple
- `active_judgments`: recent `JudgmentPopup` entries
- HUD values: `score`, `combo`, `max_combo`, `accuracy`, `grade`, `mod_acronyms`
- `judgment_counts`: `(geki, 300, katu, 100, 50, miss)`
- `results_opacity`: 0→1 fade-in for results card
- `fade_to_black`: 0→1 for intro/outro fades
- Per-column: `hit_light_age_ms`, `hit_light_judgment`, `hit_offset_per_col`, `key_press_age_ms`
- `hp`, `song_progress`, `is_kiai`, `unstable_rate`, `avg_hit_offset_ms`
- `pp`, `max_pp`, `per_column_ur`, `miss_break_age_ms`

SV positioning uses **cumulative-distance integration** (`_y_integrated()`)
for lazer-faithful note positions across SV section boundaries. Old
point-sample code (`_y()`) kept as legacy fallback.

Key constants (render.py):
```
APPROACH_MS = 600           # ms for note to travel receptor→top (at SV baseline=17)
SCROLL_SPEED_BASELINE = 17  # scales approach_ms: actual = 600 * 17 / scroll_speed
RESULTS_DURATION_MS = 6000  # results card display time
RESULTS_GAP_MS = 800        # silence between last note and results
START_FADE_MS = 1600        # fade-in from black at song start
END_FADE_MS = 600           # gameplay→results transition fade
HIT_LIGHT_DURATION_MS = 320 # receptor flash linger
COMBO_POP_DURATION_MS = 180 # combo number scale animation
```

---

## GPU Rendering System

### `gpu/context.py` — HeadlessGl

Creates a standalone EGL + OpenGL 3.3 context without a window manager.
Supports multi-GPU pinning via `R3D_EGL_DEVICE_INDEX` env var using
`eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT)`. Allocates an offscreen
FBO at render resolution. Context manager: `with HeadlessGl(w, h) as gl:`.

### `gpu/shaders.py` — GLSL Programs

Loads from `assets/shaders/`:
- `sprite.vert` + `sprite.frag` — basic textured quad
- `sprite_instanced.vert` + `sprite.frag` — instanced variant for batching
- `flashlight.frag` — HD flashlight post-process

### `gpu/atlas.py` — SpriteAtlas

Packs all sprites into a single `Texture2DArray`. Two regions:
1. **Global slots**: stage frames, judgement popups, lighting, `hit_strip`,
   `note_circle`, `bg_vignette`. Indexed by name via `atlas.index_of()`.
2. **Per-column slots**: tap/head/body/tail/receptor_off/receptor_on × K columns.
   Indexed by `atlas.column_slot_index(kind, col)`.

**Sprite resolution priority** (4 tiers):
1. Beatmap dir (per-map overrides)
2. Skin dir (via skin.ini `NoteImage{N}` / `KeyImage{N}` overrides)
3. Skin dir (conventional filenames like `mania-note1.png`)
4. Bundled fallback PNGs (shipped in `assets/sprites/`)
5. 4×4 transparent placeholder

Supports **animation frames** (`<base>-0.png`, `<base>-1.png`, …) for
judgement popups, stage-light, and lighting effects. `note_hold_body` sprites
use `_fit_stretch` (non-uniform stretch) while all others use `_fit_letterbox`.

**Per-column layout**: wiki-correct table for 1K–18K defining "outer" (1),
"inner" (2), and "centre" (S) column kinds. `SpecialStyle` relocates the S
lane to column 0 or K-1 for even keycounts ≥ 6.

### `gpu/renderer.py` — FrameRenderer

All draw calls, in order:

#### Playfield geometry
- Resolved from skin.ini `[Mania]` block or lazer defaults (480-ref coords → render px)
- Column widths, spacing, line widths, receptor Y position, stage-left/right positions
- Auto-centres the playfield within the 4:3 region of the 16:9 frame

#### Draw order per frame (in `draw()`)
1. **Background** — beatmap bg image (scaled to fill, dimmed)
2. **Background vignette** — radial dark overlay (global slot `bg_vignette`)
3. **Stage decorations** — `stage_left`, `stage_right`, `playfield_frame`, `hit_light`
4. **Column backgrounds** — per-column tinted strips
5. **Hit strip** — rainbow strip at the receptor line
6. **Note bodies** — hold body sprites (stretched to fill time interval)
7. **Note heads/tails/taps** — per-column note sprites
8. **Receptors** — off (idle) per column, on (pressed) per column
9. **Stage light** — per-column flash strip on key press (animated at `LightFramePerSecond`)
10. **Hit lighting** — per-column impact flash (`lighting_n` / `lighting_l`)
11. **Judgment popups** — floating `geki/300/katu/100/50/miss` sprites (animated)
12. **Combo break shake** — playfield offset on big combo misses
13. **Score/combo/accuracy** — rasterized via PIL → GL texture
14. **HP bar** — thin bar at bottom-left
15. **Progress bar** — thin bar at top of screen
16. **UR bar** — hit-offset distribution display
17. **Key overlay** — per-column key flash at receptor
18. **Grade letter** — large grade on results card
19. **Mod pills** — "4K HD DT" labels
20. **Hit error popup** — floating "+8ms" per-column
21. **Banner text** — "artist - title [diff] | player"
22. **Watermark** — bottom-right text
23. **PP counter** — live PP display
24. **Results card** — full-screen overlay with judgment counts, grade, UR
25. **Flashlight pass** — radial mask over everything (when HD mod active)
26. **Black fade** — full-screen fade for intro/outro transitions

#### Instanced rendering
All batched sprites use `glDrawArraysInstanced` with a single VBO of 4 unit-quad
corners + a per-instance buffer of `(x, y, w, h, atlas_layer, r, g, b, a)`.
Batches of up to 4096 instances per draw call.

#### Key draw-time features
- **HD (Hidden)**: notes fade as they approach the receptor; receptors dim
- **FI (Fade In)**: notes appear at full opacity at spawn, remain visible
- **FL (Flashlight)**: narrow bright circle around a cursor position, rest is dark
- **Kiai**: stage lights pulse; hit strip glows
- **Miss shake**: entire playfield offsets briefly after big combo breaks
- **Stage-light flash**: per-column bright strip on key press, fades over time
- **Combo pop**: combo number scales up briefly on each increment
- **Smooth SV**: cumulative-distance integration for note positioning
- **Note trails**: optional ghost notes trailing each visible note (off by default)

### `gpu/readback.py` — FrameReader

PBO-ring asynchronous readback. Maintains a ring of 3 Pixel Buffer Objects:
- Issues `glReadPixels` into PBO[N] (async GPU→pinned-host copy)
- Maps PBO[N-2] (settled data) → returns bytes
- Provides `drain()` to flush the last 2 frames after the render loop

Falls back to synchronous `fbo.read()` when PyOpenGL is unavailable.

### `gpu/text.py` — Text → GL Texture

Renders short strings via PIL (`ImageFont.truetype`) → uploads to a
ModernGL texture with mipmaps. Font lookup order: DejaVuSans-Bold →
LiberationSans-Bold → NotoSans-Bold → Arial Bold → PIL bitmap default.
Caches font handles via `lru_cache`.

---

## Skin System (`skin_ini.py`)

Parses `skin.ini` with full spec coverage:
- `[Colours]` → `Combo1..N` note colour palette
- `[Mania]` per-keycount blocks → per-column colours, sprite paths, stage frames,
  judgement popup overrides, playfield geometry (`ColumnWidth`, `ColumnSpacing`,
  `HitPosition`, `SpecialStyle`, etc.)
- `0`-indexed and `1`-indexed column numbers both accepted (osu-stable compat)
- Missing keys fall back to spec defaults

Output is `SkinIni` → `ManiaSection` (per keycount) with all parsed fields.

---

## Constants Reference

| Constant | Value | Purpose |
|---|---|---|
| `APPROACH_MS` | 600 | Base time for note travel (at SV=1, scroll=17) |
| `SCROLL_SPEED_BASELINE` | 17 | Reference scroll speed (1-40 scale) |
| `RESULTS_DURATION_MS` | 6000 | Results card display time |
| `RESULTS_GAP_MS` | 800 | Gap between last note and results |
| `START_FADE_MS` | 1600 | Black fade-in from song start |
| `END_FADE_MS` | 600 | Gameplay→results fade |
| `HIT_LIGHT_DURATION_MS` | 320 | Receptor flash duration |
| `COMBO_POP_DURATION_MS` | 180 | Combo number scale duration |
| `MISS_GRACE_MS` | 250 | How long true misses stay visible past receptor |
| `WINDOW_320` | 16.5 | Stable mania 320 window (ms) |
| `WINDOW_300` | 40 | Stable mania 300 window |
| `WINDOW_200` | 73 | Stable mania 200 window |
| `WINDOW_100` | 103 | Stable mania 100 window |
| `WINDOW_50` | 127 | Stable mania 50 window |
| `_TAIL_MULTIPLIER` | 1.5 | Hold tail release window multiplier |
| `PLAYFIELD_X_FRAC` | 0.36 | Legacy playfield X position (fraction of screen) |
| `PLAYFIELD_W_FRAC` | 0.28 | Legacy playfield width (fraction of screen) |
| `LAZER_DEFAULT_COLUMN_SIZE_REF` | 30 | Default column width in 480-ref pixels |
| `NOTE_HEIGHT_REL_COL` | 0.95 | Note height as fraction of column width |
| `RECEPTOR_HEIGHT_REL_COL` | 1.0 | Receptor height as fraction of column width |
| `RECEPTOR_BOTTOM_OFFSET_FRAC` | 0.05 | Receptor Y offset from bottom |

---

## CLI Reference (`cli.py`)

```
osu-renderer <play.osr> <beatmap_dir/> -o out.mp4 [options]
```

Key flags:
- `--resolution WxH` (default 1920x1080)
- `--fps N` (default 60)
- `--encoder auto|h264_vaapi|h264_nvenc|libx264`
- `--skin-dir <path>` — path to extracted .osk directory
- `--bg-dim 0.0-1.0` / `--bg-dim-intro/game/breaks 0-100`
- `--scroll-speed 1-40`
- `--allow-converted` — render std/taiko/ctb via converter
- `--convert-to-keys 4-10`
- Numerous `--no-*` toggles for HUD elements
- `--show-pp` — enable live PP counter
