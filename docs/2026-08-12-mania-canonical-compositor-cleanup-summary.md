# Mania canonical compositor cleanup

## Starting state

- Branch: `TheAussie/mania-canonical-compositor-cleanup`
- HEAD: `c563ada mania: restore the B1-B4 corner key counter (revert #14) (#16)`
- Initial `git status --short`: clean
- The task brief named `main`, but the clean worktree was already on the dedicated cleanup branch at the expected commit. No branch, remote, commit, or push operation was performed.

## Call-site inventory

### `FrameRenderer.draw()`

Before cleanup, the monolithic method was defined in `gpu/renderer.py` and had two caller classes:

- Legacy production/public route: `render/render.py::render_mania()` called `fr.draw(scene_full)` once per output frame.
- Slow tests: `test_gpu_renderer_background.py`, `test_gpu_renderer_hud.py` (two calls), `test_gpu_renderer_playfield.py`, and `test_gpu_renderer_text.py` called `fr.draw()` directly.

After cleanup, `FrameRenderer.draw()` and every `fr.draw(...)` call are gone. The GPU tests now construct a `FrameContext` and call `render.compositor.compose_frame()` with the populated registry.

### Monolithic `render_mania()`

Before cleanup:

- `osu_mania_renderer_v2.__init__::_gpu_render_mania()` lazily imported `render.render.render_mania()`.
- Package `render_mania()` selected `_gpu_render_mania()` whenever `USE_WIKI_RENDERER` was false.
- The package API is used by `cli.py`, `tests/test_render_orchestrator.py`, and the documented library example.

After cleanup, `render.render.render_mania()` remains only as a compatibility-shaped forwarding coroutine. It imports and awaits `render.compositor.render_mania()` and contains no parsing, GL, frame, readback, or encoder loop.

### Canonical compositor

- `render.compositor.render()` remains the real render loop.
- `render.compositor.render_mania()` is the public-signature adapter used by the package API and the old direct import path.
- `compose_frame()` is now the single per-frame dispatch function: begin frame, iterate `RENDER_ORDER`, invoke registered render functions, flush.
- `resolve_elements()` caches registry asset/variable resolution once per render.
- `render.pipeline._ORDER` is the only painter-order authority.
- Direct compositor callers are the worker compatibility CLI, the canonical smoke test, and the public adapter.

### Environment selector

Before cleanup, `OSU_USE_WIKI_RENDERER` populated `USE_WIKI_RENDERER` in `__init__.py` and selected between two composition loops. The environment variable is no longer read. `USE_WIKI_RENDERER = True` remains as an explicitly documented compatibility marker for library code that imported the old symbol; it does not branch rendering.

### Compatibility shim

`wiki_renderer.py` remains an external compatibility surface only. It re-exports compositor registry/types/functions and forwards `python -m osu_mania_renderer_v2.wiki_renderer` to `render.compositor._cli()`. No rendering logic was moved into the shim.

### Retained lower-level GPU primitives

These `FrameRenderer._draw_*` methods remain because canonical elements or their results fallback still call them:

- `_draw_background` — `render.stage.background`
- `_draw_watermark` — `hud.elements.watermark`
- `_draw_combo_and_judgment` — the no-score-font fallback in `render.notes`
- `_draw_progress_bar` — the non-Argon progress element
- `_draw_external_texture` — `FrameContext`, notes/HUD effects, break overlay, logo, and results delivery
- `_draw_direct` — `FrameContext` full-resolution sprite path
- `_draw_results_overlay` — canonical results element
- `_draw_ur_histogram` — fail-soft results-card fallback
- `_draw_sprite` / `_draw_sprite_idx` — batched sprite primitives used throughout canonical elements

`draw_logo_splash()`, `apply_note_cover()`, and `_flush_sprite_batch()` are also retained non-`_draw_*` primitives used by registered elements and frame lifecycle code.

## Changes

- Made the package v2 API route unconditionally through the compositor.
- Kept `render.render.render_mania()` as a thin forwarding compatibility name.
- Added `compose_frame()` and `resolve_elements()` so production and GPU tests exercise the same dispatch mechanism.
- Removed `FrameRenderer.draw()` and the second complete frame/GL/encoder loop from `render/render.py`.
- Removed GPU composition methods proven unreachable after the test migration: `_draw_stage_decorations`, `_draw_banner`, `_draw_hud`, `_draw_mode_pills`, `_draw_judgments`, `_draw_hit_strip`, `_draw_hp_bar`, `_draw_hit_error_popups`, `_draw_top_chrome`, `_draw_ur_summary`, `_draw_flashlight_pass`, `_draw_stage_lights`, `_draw_columns`, `_draw_notes`, and `_draw_receptors`.
- Removed their dead helper methods/constants, including `_note_anim_fps`, `_stage_light_fps`, `_stage_light_tint`, obsolete banner setup, and retired draw-only geometry constants.
- Removed the no-op `fail_overlay` registry element and updated the exact canonical-order test.
- Updated architecture comments and README wording from an A/B wiki/legacy model to the canonical compositor model.
- Did not change `build_render_plan()` or `build_frame_state()` gameplay semantics.

## Compatibility

- Package/library entrypoint: `from osu_mania_renderer_v2 import render_mania` keeps the same async signature and now forwards to `render.compositor.render_mania()`.
- Old direct import: `osu_mania_renderer_v2.render.render.render_mania` still exists and forwards to the same adapter.
- Worker command: `python -m osu_mania_renderer_v2.wiki_renderer ...` still forwards through the thin shim. `--help` exits successfully, and the progress-line implementation in `_cli()` is unchanged.
- Canonical compositor: callers that already use `render.compositor.render()` retain its explicit `skin_dir` / `default_skin_dir` contract.
- Skin fallback: package/ordinary CLI calls with `skin_dir=None` now use a temporary empty user-skin directory for the duration of the render, matching the existing worker compatibility behavior instead of raising a new hard error. Explicit skin paths are passed through unchanged.
- `log_path` remains accepted for ABI compatibility. Existing package-route `render_start` / `render_done` structured log events are retained.

## Dead code removed

### Results stage 2

- `_bake_stage2`
- `_draw_stats_panels`
- `_draw_bar_rows`
- `_draw_perf`
- `_draw_combo`
- `_draw_judgements`
- `_panel_unfold`
- `_bake_area_chart`
- `STAGE1_MS`, `OPEN_MS`, and `STAGGER_MS`
- `_stage2` flags and all open/slide/unfold branches and pose-signature fields
- Stats-panel images/geometry and performance/judgment row bake state
- Combo timeline, map-max-combo, UR, and average-offset fields/calculation that existed only for stage 2
- The leaderboard's now-constant stage-2 translation argument

The centered results card, score/grade animation, mod/star/grid rows, date, black wash, settled timing, and optional leaderboard flank remain.

### Fail overlay

- `FrameRenderer._draw_fail_overlay`
- `hud.elements.fail_overlay`
- The `fail_overlay` canonical registry stage

Fail/quit behavior still comes from the existing death-time render truncation in `build_render_plan()`; that calculation was not changed.

### Other retired composition code

- The all-in-one `FrameRenderer.draw()` painter sequence
- The duplicated render loop in `render/render.py`
- GPU helper implementations used only by that painter, listed under Changes
- The `_gpu_render_mania` lazy fallback and the internal environment-controlled A/B branch

## Tests

Validation used an isolated temporary Python 3.14.6 virtual environment created from `.[dev]` because the checkout had no repository venv.

### Focused canonical/results tests

Command:

```bash
PYTHONHASHSEED=0 python -m pytest tests/test_wiki_renderer.py tests/test_results_composition_regression.py
```

Result: **4 passed, 1 failed, 1 skipped**.

The failure is the pinned results-state digest: this environment produces `4a3798e9eeaba840f62b9e50ff4b940c007c31800725fff3ff0a80cdde219a5a`, while the test expects `da731ad...`. Running the untouched `c563ada` tree under the same interpreter produces the same `4a3798e9...` failure, so it is environment/pre-existing rather than cleanup-caused. The pinned expected digest was not changed.

### Canonical GPU tests

Command:

```bash
RUN_SLOW=1 PYTHONHASHSEED=0 python -m pytest \
  tests/test_gpu_renderer_background.py \
  tests/test_gpu_renderer_hud.py \
  tests/test_gpu_renderer_playfield.py \
  tests/test_gpu_renderer_text.py
```

Result: **6 passed, 0 failed, 0 skipped**. These tests now compose through `FrameContext + RENDER_ORDER` rather than `FrameRenderer.draw()`.

The package/orchestrator and the same GPU files without `RUN_SLOW` also produced **1 passed, 7 skipped**.

### Non-slow suite

Command:

```bash
PYTHONHASHSEED=0 python -m pytest -m 'not slow'
```

Result: **81 passed, 6 failed, 13 deselected**.

The five task-brief failures remain exactly as expected:

- `test_slider_stream_notes_inherit_player_press_columns`
- `test_circle_press_match_unchanged`
- `test_unmatched_stream_notes_fall_back_to_heuristic`
- `test_build_cmd_software`
- `test_no_notes_visible_before_any_appear`

The sixth failure is the Python-3.14 results digest described above and is reproduced by untouched HEAD. **No new cleanup-caused non-slow failure was introduced.**

### Canonical MP4 smoke

Command:

```bash
RUN_SLOW=1 PYTHONHASHSEED=0 python -m pytest tests/test_wiki_renderer.py -k wiki_path_renders_argon
```

Result: **1 failed, 4 deselected** with an ffmpeg broken pipe. Untouched `c563ada` fails identically in this environment. GL composition itself initializes and emits frames, but end-to-end MP4 smoke remains unverified here because the local ffmpeg path exits early.

### Lint and repository checks

- `python -m compileall -q osu_mania_renderer_v2 tests`: passed.
- `python -m ruff check osu_mania_renderer_v2 tests`: non-zero with **92 existing errors**. Untouched HEAD reports **143 errors** under the same Ruff version. Changed canonical routing/test files pass a focused Ruff invocation; unrelated/pre-existing repository lint debt was not rewritten.
- `git diff --check`: passed.
- `python -m osu_mania_renderer_v2.wiki_renderer --help`: passed.

## Fidelity

- Inspected and preserved the existing `scripts/baseline_capture.py` state-hash approach.
- The results-period state digest produced by cleanup and untouched HEAD is identical (`4a3798e9...`) under the available interpreter.
- Performed a direct HEAD-vs-cleanup pixel hash of `ManiaLazerResults.render_overlay()` at ages 0, 320, 1000, 2000, and 3800 ms with a fixed timestamp. All five SHA-256 hashes match exactly; representative settled hash: `3e21890fc1aa8c407200d407746842b0e9753710d28ea6eef23b287e9c63dc7b`.
- Six canonical EGL GPU composition tests pass.
- `git diff -U0` contains no B1/B2/B3/B4 label or key-counter behavior change. The key-counter element and its ordering remain intact.
- Full encoded MP4 byte/pixel parity remains human/render-box gated because the local canonical smoke hits the same pre-existing ffmpeg broken pipe on HEAD and cleanup.

## Remaining cleanup

This renderer-local change intentionally does not alter cross-repository routing or persisted compatibility fields. A later coordinated cleanup must handle:

- `users.use_legacy_renderer`
- stale `users.use_wiki_renderer`
- `RenderJob.use_wiki_renderer`
- bot/worker module routing away from the `wiki_renderer` compatibility name
- NAS/persisted job compatibility and queue serialization

No Bot repository, Website repository, database schema, or queue payload was edited here.
