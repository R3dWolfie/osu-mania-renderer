# AIREADME — Project Map for AI Agents

A consolidated reference for anyone (human or AI) coming fresh to the
**R3DRenderer** ecosystem. This document is the source of truth for
*what each piece is, how the pieces talk to each other, and where each
known issue lives*. Skim the top, jump to the section that matches the
work in front of you.

> **For users:** the public README.md is the friendlier intro.
> **For contributors:** `docs/NAVIGATION.md` is the renderer-only code tour.
> **For AI agents in mid-conversation:** start here.

## What this is

The R3DRenderer ecosystem turns an osu! `.osr` replay (plus the matching
beatmap) into an MP4 video, and serves those videos on
[renderer.r3dwolfie.com](https://renderer.r3dwolfie.com) — basically an
o!rdr clone with native osu!mania support, custom skin support, and a
Discord-bot upload front-end.

## Two repos

This project lives across **two separate git repos** on Red's local
machine:

| Repo | Path on Red's box | Role | Pushed to GitHub |
|---|---|---|---|
| `OsuManiaRenderer` | `/var/home/red/Projects/Reddie/OsuManiaRenderer/` | The actual renderer — Python + ModernGL + EGL + ffmpeg. Stateless library + CLI. | `R3dWolfie/OsuManiaRenderer` (private). |
| `Mania ORDR` | `/var/home/red/Projects/Mania ORDR/` | The Discord bot + web frontend + queue + DB. Wraps the renderer subprocess. | No public remote. Local only. |

They communicate via subprocess: `mania_ordr.renderer` shells out to
`python -m osu_renderer.cli` (mania) or to `danser-cli` (std).

## Mode support matrix

| osu! mode | Renderer | Status |
|---|---|---|
| 0 — osu!standard | `danser-go` (single binary, std-only) | ✅ Works. Custom skins may have visual quirks per skin author. |
| 1 — osu!taiko | None today | ❌ Bounced at upload with BETA message. Scaffold at `osu_renderer/taiko_renderer.py`. |
| 2 — osu!catch | None today | ❌ Bounced at upload with BETA message. Scaffold at `osu_renderer/catch_renderer.py`. |
| 3 — osu!mania | `osu_renderer` (this repo's Python+GPU pipeline) | ✅ Works. Custom skins in BETA. Std→mania converted renders also in BETA. |

`danser-go` is *fundamentally* std-only — `libdanser-core.so` panics on
`-replay` with mode ≠ 0 (`Modes other than osu!standard are not
supported` at `app/app.go:212`). Quoted from the danser README:
"danser-go is a GUI/CLI visualisation tool for osu!standard maps."

## High-level dispatch

```
user uploads .osr
      │
      ▼
mania_ordr/embed_server.py  (web /upload) ─┐
mania_ordr/bot.py (Discord on_message)  ───┤── builds RenderJob
mania_ordr/embed_server.py (POST /api/v1/render) ─┘
      │
      ▼  WorkerPool.submit(job)
mania_ordr/worker.py
      │
      ▼  determines mode from .osr byte 0
      │
      ├── mode=3 (mania) ──► mania_ordr/renderer.py
      │                       │
      │                       ▼  subprocess
      │                       python -m osu_renderer.cli
      │                           [or .wiki_renderer for admin opt-in]
      │
      └── mode=0 (std)   ──► mania_ordr/danser_renderer.py
                              │
                              ▼  subprocess
                              danser-cli -replay ... -record ...

      │
      ▼  on success
   .mp4 + .jpg thumb + metadata.json land in web/videos/{id}.* on NAS
   embed page /v/{id} serves it
```

## OsuManiaRenderer — file map

Beyond what `docs/NAVIGATION.md` covers. Recent work highlighted (★ = new this session).

```
osu_renderer/
├── __init__.py                     public API: render_mania()
├── cli.py                          argparse entry — `python -m osu_renderer.cli`
├── render.py                       orchestrator — beatmap + replay → GPU → ffmpeg pipe.
│                                   Contains the lazer skip-intro fix (drops pre-skip
│                                   notes so judgement loop doesn't cascade to FAILED).
├── beatmap.py                      .osu parser, timing-point tables, SV integration
├── replay.py                       .osr parser via osrparse; mania KeyEvent decoding
├── converter.py             ★      std → mania converter. RECENTLY REWORKED to
│                                   press-match the FULL expanded mania-note list
│                                   (slider streams included), not just the top-level
│                                   std-object list. See memory note
│                                   `std-to-mania-slider-columns`.
├── mods.py                         mod bitfield → behaviour dispatch (DT, MR, etc.)
├── judgments.py                    hit-window math, OD-derived timing windows
├── scene.py                        per-frame SceneState
├── skin_ini.py                     [Mania] section parser
├── hitsounds.py                    pre-mixed hitsound .wav generation
├── pp.py                           rosu-pp-py wrapper (optional dep)
├── errors.py                       exception types — NotAManiaError etc.
├── models.py                       BeatmapInfo, ReplayInfo, RenderOptions dataclasses
├── encode.py                       ffmpeg pipe management
├── wiki_renderer.py         ★      Wiki-driven scaffold (admin A/B path). Empty
│                                   ELEMENTS + RENDER_ORDER registries; fails
│                                   closed with `RenderError` when called before
│                                   populating from the wiki.
├── taiko_renderer.py        ★      Same scaffold for taiko. Empty registries.
├── catch_renderer.py        ★      Same scaffold for catch. Empty registries.
├── assets/                  ★      gitignored. Bootstrap layout:
│   ├── default_skin/                  ↑ extracted stable default skin (268 files).
│   └── wiki_cache/                    ↑ sparse clone of ppy/osu-wiki + symlinks.
│                                   Regenerate via `scripts/wiki_renderer_bootstrap.sh`.
└── gpu/
    ├── context.py                  HeadlessGl — EGL context creation (device-pinned)
    ├── atlas.py                    SpriteAtlas — per-skin texture upload
    ├── renderer.py                 FrameRenderer — actual draw calls
    ├── readback.py                 FrameReader — GPU→CPU pixel readback
    └── shaders/                    GLSL fragment/vertex shaders
```

Tests live under `tests/`. Notable: `test_converter.py` (new, locks down
the slider-stream press-match regression).

## Mania ORDR — file map (the bot)

```
mania_ordr/
├── app.py                          aiohttp + discord.py wiring; top-level _run() entry
├── bot.py                          Discord ManiaOrdrBot — on_message handler;
│                                   _RetryView (now PERSISTENT — state JSON keyed by
│                                   message.id under tmp_root/_retry_state/)
├── embed_server.py                 aiohttp web frontend: /upload, /v/{id}, /me, /admin,
│                                   /settings, /api/v1/*, OAuth, Ko-fi webhook
├── worker.py                       WorkerPool — serial render dispatch, mode routing
├── renderer.py                     subprocess wrapper around osu_renderer.cli
├── danser_renderer.py              subprocess wrapper around danser-cli (std only)
├── database.py                     SQLite schema + V1..V17 migrations
│                                   (V17 added users.use_wiki_renderer admin opt-in)
├── models.py                       RenderJob, ReplayInfo, RenderRow dataclasses
├── presets.py                      User-render-preset validation + downgrade-for-free
├── beatmap_resolver.py             osu! API + .osz mirror download, MD5-strict matching
├── skin_resolver.py                Per-user skin library, default fallback
├── ratelimit.py                    Cooldown / 4K limiter
├── queue_hooks.py                  NAS queue dispatch (FoofPC + RedPC overflow)
├── nas_queue.py                    NAS-backed job queue (post-NAS-cutover)
├── osr_parser.py                   thin wrapper over osrparse for the bot's needs
├── cleanup.py                      Render row + file purge helpers
├── watermark.py                    ffmpeg-based watermarker for std renders
├── templates/
│   ├── _base.html.j2               ★ site-wide BETA banner lives here (mentions
│   │                                  custom skins + std→mania converted)
│   ├── upload.html.j2              upload form
│   ├── embed.html.j2               /v/{id} embed page
│   ├── settings.html.j2            ★ admin wiki-renderer toggle ships here
│   ├── gallery.html.j2             recent-renders grid
│   └── ... (admin, home, login, etc.)
├── static/                         CSS, favicon, logo
└── cli/r3d_render.py               headless render CLI (used by overflow worker)
```

## RenderJob shape (what flows from upload → worker)

`mania_ordr/models.py:RenderJob` — fields added this session marked ★:

| Field | Meaning |
|---|---|
| `job_id` | ULID |
| `discord_user_id` | Discord ID (web uploads use `"osu_<id>"`) |
| `discord_msg_id` | Originating message; `"web:<id>"` / `"api:<id>"` for non-Discord |
| `osr_path` | Path on FoofPC tmp_root |
| `osk_path` | Optional per-render skin .osk path |
| `osr_sha256` | Dedup key |
| `resolution` / `fps` | Per-job overrides |
| `preset` | Validated preset dict |
| `visibility` | "public" / "unlisted" |
| `priority` | 1 supporter, 10 free |
| `skin_override_id` | Library row to force |
| `use_wiki_renderer` ★ | Admin A/B route to the experimental wiki-driven renderer |

## Mode-byte gating at upload

All three upload sites read `osr_bytes[0]` and bounce:

| Byte | Mode | Action |
|---|---|---|
| 0 | std | Routed to danser_renderer.py |
| 1 | taiko | `mode_beta` error: "osu!taiko rendering is in BETA and not available yet" |
| 2 | catch | `mode_beta` error: same wording, mode=catch |
| 3 | mania | Routed to osu_renderer |
| other | unknown | `bad_mode` error |

Gate locations:
- `mania_ordr/embed_server.py` `_handle_upload_post` (web /upload, ~L800)
- `mania_ordr/embed_server.py` `_handle_apirender_submit` (API, ~L2370)
- `mania_ordr/bot.py` Discord on_message loop (~L920)

## Recent in-flight work (this session)

1. **Skip-intro fix in `render.py`** — lazer's "Skip" produces a giant
   first replay frame `time_delta` (e.g. 77559 ms) representing the
   intro skip. Drops pre-skip notes from the chart so the judgement
   loop doesn't cascade them to FAILED. Memory:
   `lazer-skip-intro-format`.
2. **Std→mania converter rewrite in `converter.py`** ★ committed `7304e8d`.
   Press-matching now runs against the FULL expanded mania-note list
   (slider streams included), not the top-level std-object list. Three
   regression tests in `tests/test_converter.py` cover the slider
   stream, plain circle, partial-press fallback. Memory:
   `std-to-mania-slider-columns`.
3. **Wiki-driven renderer scaffold** ★ on `wiki-renderer` branch
   (currently same SHA as main). Generic pipeline lookup
   (user_skin → default_skin → wiki_default → RenderError). Sibling
   scaffolds for taiko + catch. Empty registries; fills as the wiki
   gets cracked open. Bootstrap script:
   `scripts/wiki_renderer_bootstrap.sh`.
4. **Admin A/B routing in mania_ordr** — `users.use_wiki_renderer` DB
   column (V17 migration), `RenderJob.use_wiki_renderer` field, toggle
   in `/settings` (admin-only), gate in three upload sites. Cleanup
   plan when wiki-renderer merges: memory note
   `wiki-renderer-admin-routing-cleanup`.
5. **Persistent _RetryView in `bot.py`** — button now has static
   `custom_id="r3d:retry"`, state JSON keyed by `interaction.message.id`
   under `tmp_root/_retry_state/`. Survives bot restarts. Class-level
   `configure()` wires bot + reporter + state_dir refs.
6. **BETA banner update** in `templates/_base.html.j2` — now reads
   *"⚠ BETA — custom mania skins and std→mania converted renders may
   have visual bugs while we iron things out."*
7. **Taiko / catch beta gates** — see "Mode-byte gating" above.

## Deploy mechanism

This is a **manual** deploy chain, not auto. AI agents and humans both
need to understand the topology before touching prod.

```
Red's box                  NAS                        FoofPC
─────────                  ───                        ──────
OsuManiaRenderer/   ── post-commit hook ──►   /var/mnt/Synology-Reddie/OsuManiaRenderer/
   (committed code,
    rsync'd to NAS)

Mania ORDR/         ── (no auto-deploy hook yet) ──►  manual rsync needed

FoofPC                                       /home/foof/r3drender/{OsuManiaRenderer,mania_ordr}/
                                             ↑ runs the live bot
                                             ↑ DOES NOT auto-pull from NAS

renderer.r3dwolfie.com  ── Cloudflare tunnel ──►  FoofPC's aiohttp embed_server on :something
```

**Deploy procedure** (current, manual):

```bash
# 1. Verify no in-flight renders
ssh foof@192.168.1.31 'pgrep -af "danser|osu_renderer"'   # should be empty

# 2. Rsync renderer (mirrors post-commit exclude list)
rsync -a --delete \
    --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' \
    --exclude='*.mp4' --exclude='assets/' --exclude='.git/' \
    /var/home/red/Projects/Reddie/OsuManiaRenderer/ \
    foof@192.168.1.31:/home/foof/r3drender/OsuManiaRenderer/

# 3. Rsync bot
rsync -a --delete \
    --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' \
    --exclude='*.mp4' --exclude='.git/' --exclude='.env' \
    "/var/home/red/Projects/Mania ORDR/" \
    foof@192.168.1.31:/home/foof/r3drender/mania_ordr/

# 4. Restart
ssh foof@192.168.1.31 'systemctl --user restart mania-ordr.service'

# 5. Verify
ssh foof@192.168.1.31 'systemctl --user is-active mania-ordr.service'
ssh foof@192.168.1.31 'journalctl --user -u mania-ordr.service --since "5 seconds ago" --no-pager | tail -15'
```

FoofPC SSH alias: `foof@192.168.1.31`. Bot service: `mania-ordr.service`
(user-scoped systemd unit). Workdir: `/home/foof/r3drender/`.

## Local testing (renderer repo only)

```bash
cd /var/home/red/Projects/Reddie/OsuManiaRenderer
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_placeholder_sprites.py

# Run unit tests
.venv/bin/python -m pytest -q

# Run a render against a known good .osr (.osr lives under tests/fixtures/)
.venv/bin/python -m osu_renderer.cli \
    tests/fixtures/sample.osr \
    tests/fixtures/sample_beatmap/ \
    -o /tmp/out.mp4 \
    --resolution 1280x720 --fps 60

# Regenerate the wiki-renderer assets (default skin + wiki cache)
DEFAULT_SKIN_OSK=/path/to/default.osk ./scripts/wiki_renderer_bootstrap.sh
```

The renderer needs an EGL-capable GPU at runtime (NVIDIA or AMD;
software fallback is too slow for any real chart). VAAPI encoding is
preferred (`--encoder h264_vaapi`); NVENC on FoofPC's NVIDIA pools.

## Known issues + memory pointers

The persistent memory store at
`~/.claude/projects/-var-home-red-Projects-MC-servers-Me-and-Vio---/memory/`
carries operational facts that survive across AI sessions. Notable
entries:

- `feedback_no_restart_during_renders.md` — verify no danser/render
  subprocess before restarting mania-ordr.
- `feedback_skin_must_be_skin_agnostic.md` — fixes must work for any
  uploaded skin; no per-skin heuristics.
- `project_lazer_skip_intro_format.md` — explains the skip-intro bug
  + the render.py fix.
- `project_std_to_mania_slider_columns.md` — the converter drift
  pattern; the rewrite that fixes it.
- `project_wiki_renderer_admin_routing_cleanup.md` — what to tear out
  when wiki-renderer eventually merges and the A/B is no longer needed.
- `project_skin_renderer_anchoring_pending.md` — receptor anchoring
  geometry needs top-aligned (not centered) to match real osu!.
- `reference_testing_data_path.md` — R3DRenderer testing artifacts
  go to `Synology-Reddie/R3DRenderer testing/`.

## Environment + secrets

Production env lives on FoofPC at `/home/foof/r3drender/state/env`.
Contains (do not log or write to files):
- `DISCORD_BOT_TOKEN`
- `OSU_API_CLIENT_ID` / `OSU_API_CLIENT_SECRET`
- `KOFI_VERIFICATION_TOKEN`

If any of these leak in a conversation, treat as compromised and rotate.

Render pools:
- **Pool A** = FoofPC, RTX 2070 SUPER, NVENC, std/mania
- **Pool B** = FoofPC, GTX 1070, NVENC, std/mania (overflow)
- **Pool C** = RedPC, RX 7900 XTX, VAAPI, mania-only (overflow)

## Glossary

- **.osr** — replay file. Mode byte at offset 0.
- **.osu** — beatmap file (plain text, ASCII-with-utf8). Mode declared inside.
- **.osz** — beatmapset (zip of multiple .osu + audio + bg).
- **.osk** — skin package (zip).
- **mania convert** — when a player loads a non-mania map in lazer with the
  mania ruleset, lazer's `ManiaBeatmapConverter` produces a synthetic
  mania chart at play time. The replay's `beatmap_md5` still references
  the *source* (non-mania) .osu. Our renderer must reproduce the same
  conversion to render the replay correctly.
- **danser** — std-only replay-rendering tool (`wieku/danser-go`). The
  std path's only renderer today.
- **lazer** — modern osu! client (`ppy/osu`); replays it produces have
  `replay_id == -1`, `rng_seed == 0`, no life-bar graph, and use
  large-time-delta skip-intro markers.
- **stable** — legacy osu! client; positive `replay_id`, non-zero
  `rng_seed`, has life-bar graph.

## Where to look first when something breaks

| Symptom | First place to look |
|---|---|
| Render shows FAILED + 0% across the whole intro | Lazer skip-intro; `render.py` `skip_intro_detected` log |
| Std→mania looks wrong on slider-heavy maps | Converter drift; check `converter.py` `_assign_columns_to_notes` |
| Black rectangles on a custom-skin std render | Skin author saved sprite without alpha; `atlas_loaded` log line + skin dir |
| Retry button → "This interaction failed" | Pre-restart _RetryView; check `tmp_root/_retry_state/{message_id}.json` exists |
| Wrong .osu picked | `beatmap_resolver.py` md5-strict path; should NEVER fall back to "first .osu in dir" |
| Bot won't start, AttributeError on Settings | Don't reference `settings.X` for vars defined locally in `_run()` (e.g. `tmp_root`) |
| Custom mania skin looks wrong | Skin-pipeline work in progress; user-facing BETA banner is intentional |

## Open scaffolds / planned work

- **CTB renderer prototype** — design discussed; ~1 week for v0
  (catcher + tap fruits + HUD), +1 week for v1 (sliders + hyperdash),
  +1 week for v2 (polish). Scaffold at `osu_renderer/catch_renderer.py`.
- **Taiko renderer prototype** — simpler than CTB. No date set.
  Scaffold at `osu_renderer/taiko_renderer.py`.
- **Wiki-driven mania renderer** — replace the current hand-coded
  mania pipeline with the wiki-driven scaffold once `ELEMENTS` +
  `RENDER_ORDER` are populated from
  `assets/wiki_cache/mania/skinning.md`.
- **Mania ORDR post-commit auto-deploy hook** — parity with
  OsuManiaRenderer's hook so the manual rsync step becomes unnecessary.

## Repo conventions

- Commit messages: subject line ≤72 chars, imperative ("converter: …",
  not "Updated converter…"), body explains the *why*.
- No emojis in commit messages or code unless the user explicitly asks.
- Default to writing zero comments; the WHY belongs in commits, the
  WHAT belongs in well-named identifiers.
- Tests live under `tests/`. Regression tests must include enough
  context in the docstring for a future reader to understand the bug
  they're locking down without reading the conversation that
  produced them.
