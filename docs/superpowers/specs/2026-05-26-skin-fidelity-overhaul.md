# Skin Fidelity Overhaul — Spec

**Goal:** Make `osu_renderer` render custom osu! mania skins like real osu! / lazer does — drop in any `.osk`, render is correct. **No per-skin code paths.** One algorithm reading skin.ini + sprite metrics, applied uniformly.

**Why this spec exists:** Previous incremental fixes (square-crush → aspect-corrected receptors → bottom-anchored receptors) each made the receptor math less wrong but left the broader rendering visibly off-spec compared to real osu!. User feedback after the v2 receptor fix flagged three coupled problems: receptors still wrong-looking, hold bodies and notes wrong, playfield way too narrow. They're coupled because a wider playfield is the prerequisite for any of the sprite-fidelity work to look right — narrow column makes correctly-anchored tall sprites look weird, makes hold bodies look chunky, makes notes look like dots.

**Reference impl:** ppy/osu Mania ruleset (Legacy* classes), confirmed by Quaver. danser-go provides only the skin-loader pattern (it doesn't render mania). See `references/skin-pipeline-research.md` (this dir, sibling).

**Non-goals:** UpsideDown=1 polish (FNF works "well enough" as a fallback); per-beatmap skin overrides (already implemented); animated frame timing changes; encode/codec changes; result screen overhaul.

## Architecture

The render is currently structured as one big `FrameRenderer` with hardcoded fractional playfield placement (`PLAYFIELD_X_FRAC=0.36`, `PLAYFIELD_W_FRAC=0.28`) and square-crush sprite drawing. The overhaul keeps `FrameRenderer` as the host but moves geometry into a `PlayfieldLayout` resolver and rewrites three sprite draw paths (receptors, taps, hold bodies) to follow lazer's anchoring rules.

**Module boundaries:**
- `gpu/playfield.py` (new) — `PlayfieldLayout` resolves column widths, X positions, hit-position Y, stage chrome rects from skin.ini + bundle defaults. Pure data, no GL calls.
- `gpu/renderer.py` — consumes `PlayfieldLayout` instead of computing fractional placement. Receptor / note / hold-body draws use the lazer model.
- `gpu/atlas.py` — already exposes `column_aspect`; add `column_native_size(kind, col) -> (w, h)` for hold-body tile math.
- `skin_ini.py` — already parses most fields. Verify `WidthForNoteHeightScale`, `NoteBodyStyle`, `BarlineHeight` make it through.

## Phases

### Phase 1: Playfield geometry (no sprite changes yet)
**Files:** `gpu/playfield.py` (new), `gpu/renderer.py:_compute_playfield_geometry` (rewrite)

Resolve column widths from skin.ini's `[Mania] Keys: N` section's `ColumnWidth[]` (one width per column). When missing, fall back to lazer's default — `48 × 1.6 ≈ 76.8` logical-px per column, scaled to our render resolution. The playfield total width = sum(col_widths) + sum(col_spacings) + 2 × side_padding. X position = centred horizontally OR derived from skin.ini's `ColumnStart` value.

For 4K at 1280×720: total column width = 4 × 77 + 3 × spacing ≈ 320-360px (vs the current ~360px from 0.28×1280) — actually similar overall. But for 7K: lazer would render 7 × 77 ≈ 540px (vs current 360px squeezed across 7 cols). **The big visual difference is per-column width grows for higher-K maps.**

Decision: also widen the default for 4K. Real osu! at 720p with default-skin column widths actually renders columns ~80px wide too, but the playfield sits within a wider stage frame with chrome. Either widen the default per-column or leave defaults and lean on stage chrome to fill side gaps.

**Acceptance:** 4K render keeps current per-column width. 7K render has each column slightly wider than now. No sprite changes yet — receptors stay square-crush. Compare a smoke render to baseline.

### Phase 2: Stage chrome
**Files:** `gpu/renderer.py:_draw_stage_chrome` (new method)

Draw `mania-stage-left` to the left of the playfield, vertically full-height (or scaled to native aspect from native width), bottom-anchored. Same for `mania-stage-right` on right. Draw `mania-stage-hint` as a thin horizontal strip at HitPosition Y inside the playfield. Use atlas slots already loaded.

Per lazer: stage panels' X position = directly adjacent to the column edges. Native sprite width scaled to a sensible value (lazer scales by `POSITION_SCALE_FACTOR = 1.6`, but our render is at 720p so a different scale applies — TBD by smoke test). Height = scaled by sprite aspect, anchored to the bottom of the screen (or top of HitTarget).

**Acceptance:** 4K Pii AR11 render shows left/right stage chrome filling the BG-bleed gaps. Skins that ship no stage sprites get the existing dimmed-BG look (no regression).

### Phase 3: Receptors (re-apply v2, now that the column context is right)
**Files:** `gpu/renderer.py:_draw_receptors`

Same change as my v2 attempt: `rec_h = col_w / aspect`, anchor `rec_y = centre_y - cw // 2` for downscroll. Lighting (`lighting_l`, `lighting_n`) anchors to the hit centre (`centre_y`), not the receptor rect. UpsideDown branch keeps old centred behaviour.

Pii AR11's tall key sprites now extend upward into a wider column with stage chrome on the sides — looks right.

**Acceptance:** Pii AR11 receptors look like Pii AR11 in real osu!. Night05 (square sprites) unchanged. SC arrows proportional.

### Phase 4: Tap notes + hold heads/tails
**Files:** `gpu/renderer.py:_draw_notes` (the note loop)

For each note:
- X position: `col_x[c]`, width: `col_w[c]` (unchanged)
- Height: if skin.ini has `WidthForNoteHeightScale`, use that; else `cw × (tex.h / tex.w)`
- Anchor: bottom of the sprite at the note's hit-time Y (`to_screen_y(1.0) - 0` for the rendered hit moment)

For falling notes mid-track, the note's bottom is at `to_screen_y(y_fraction)` and the top extends up by note_h.

**Acceptance:** Tap notes of skins that ship wide-flat note sprites (e.g. FFR-style chevrons) render at the right aspect, not squashed.

### Phase 5: Hold body NoteBodyStyle
**Files:** `gpu/renderer.py:_draw_notes` (hold branch), `gpu/atlas.py:column_native_size`

Read `NoteBodyStyle` from skin.ini section:
- `0 / Stretch` (default for most): scale the L-sprite to fill head-bottom→tail-top, with ClampToEdge wrap so the body texture doesn't tile
- `2 / RepeatTop`, `3 / RepeatBottom`, `4 / RepeatTopAndBottom`: tile the L-sprite vertically with `Repeat` wrap; the cap (top/bottom) variant controls whether tiling scrolls

Implementation: the existing `_draw_sprite_idx` always uses ClampToEdge. Add a `wrap_repeat=False` parameter that swaps the texture sampler's wrap-Y mode. For tile-Y, the V coord goes 0..tiles_count instead of 0..1; `tiles_count = body_h_px / (cw * tex.h / tex.w)`.

**Acceptance:** Kori's pick hold bodies tile cleanly (no rectangular crush). Default-skin holds (no L sprite) still get the existing capsule fallback.

### Phase 6: Background dimming
**Files:** `gpu/renderer.py:set_background` (existing) — adjust default dim level

Per skin convention, beatmap BG should be visible but heavily dimmed during gameplay. Bump the default dim to ~50-60% (currently lower). Skins that ship `mania-stage-bg` should have that override instead.

**Acceptance:** Beatmap BG is faint, doesn't compete with gameplay.

## Test Plan

Three reference skins, three reference replays. Compare each phase end-state side-by-side against:
- Real osu! screenshot of the same skin (if available — user has osu! installed)
- The pre-overhaul baseline render
- The previous phase's render

Reference set:
1. **Pii AR11 + 4K replay** (`/var/mnt/ASUStor-Samsung/R3DManiaORDRBot/web/replays/01KS0H27N0F89Z2BN6FHX6BXYD.osr`)
2. **Kori's pick + same 4K replay**
3. **SC arrows + Harumachi 7K replay** (`/var/mnt/ASUStor-Samsung/R3DManiaORDRBot/web/replays/01KS0AB30194H1WS7G0VXTVH02.osr`)

Output dir: `/var/mnt/Synology-Reddie/R3DRenderer testing/skin-overhaul-2026-05-26/`. Frame + mp4 per skin per phase. Stop at each phase for user sign-off before moving to the next.

## Rollback Plan

Renderer changes can be reverted by restoring `/tmp/renderer.py.pre-aspect-fix.bak` on foof (which IS the pre-Phase-1 main state, since we reverted v2). For a multi-phase change, work on a feature branch in the OsuManiaRenderer git repo and only merge to main after the user signs off on the whole overhaul.

## Status

- v2 receptor change: **reverted** on foof + RedPC at 2026-05-26T~11:30 UTC
- Beta banner: **active** site-wide (`_base.html.j2`)
- Spec written: 2026-05-26
- Implementation: pending user go-ahead

## What NOT to do (lessons learned)

- Don't alpha-trim sprites — transparent padding is intentional spacing per skin authors.
- Don't ship Phase 3 (receptors) without Phase 1+2 first — narrow column + no chrome makes correctly-anchored receptors look as wrong as crushed ones.
- Don't pick a different default per-keys-count by closest-matching another section (lazer doesn't do this; fabricate fresh defaults).
- Don't add per-skin branches (`if skin.name == "Pii AR11":`). Ever.
