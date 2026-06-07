# osu!mania Custom Skinning — Implementation Planning Doc

**Target:** `/home/red/Projects/Reddie/OsuManiaRenderer/osu_renderer/`
**Goal:** feature-parity with danser-go's mania skin handling — but
danser doesn't actually render mania (the only "mania" code in the
repo is for replay-frame compat). So "parity with danser" reduces to
"parity with osu-stable mania skinning behavior as documented in the
osu! wiki," using danser's texture-resolution philosophy (4-tier
fallback, `@2x`-first, same-source animation frames, INI parsing for
global keys).

---

## 1. Mania skin asset taxonomy

All filenames support `@2x.png` (HD) and `.png` (SD); osu-stable and
danser try `@2x` first (`loadTexture` in `danser-go/app/skin/skin.go`).
Animations: `<base>-0.png`, `<base>-1.png`, …

### 1.1 Notes (default key layout)

| File | Purpose | Per-keymode? | Fallback |
| --- | --- | --- | --- |
| `mania-note1.png` | Outer-lane tap | No | bundled default |
| `mania-note2.png` | Inner-lane tap | No | falls back to `note1` |
| `mania-noteS.png` | Special (centre) tap | No | per layout |
| `mania-note1H.png` | Outer hold head | No | reuses `note1` |
| `mania-note2H.png` | Inner hold head | No | reuses `note2` |
| `mania-noteSH.png` | Special hold head | No | reuses `noteS` |
| `mania-note1L.png` | Outer hold body (animatable strip) | No | required for `NoteBodyStyle=1` |
| `mania-note2L.png` | Inner hold body | No | as above |
| `mania-noteSL.png` | Special hold body | No | as above |
| `mania-note1T.png` | Outer hold tail | No | flipped head (v2.5+) |
| `mania-note2T.png` | Inner hold tail | No | flipped head |
| `mania-noteST.png` | Special hold tail | No | flipped head |

**Default key layout** (osu-wiki `osu!mania/en.md`):

| Keys | Columns (1=outer / 2=inner / S=centre) |
| --- | --- |
| 1 | S |
| 2 | 1 1 |
| 3 | 1 S 1 |
| 4 | 1 2 2 1 |
| 5 | 1 2 S 2 1 |
| 6 | 1 2 1 1 2 1 |
| 7 | 1 2 1 S 1 2 1 |
| 8 | 1 2 1 2 2 1 2 1 |
| 9 | 1 2 1 2 S 2 1 2 1 |

**The "OINSNIO" mnemonic for 7K is wrong** — correct mapping is
`1 2 1 S 1 2 1`. Drop the OINIO mental model; use this table.

### 1.2 Receptors / keys

| File | Purpose | Fallback |
| --- | --- | --- |
| `mania-key1.png` | Outer idle | bundled default |
| `mania-key1D.png` | Outer pressed | `mania-key1.png` |
| `mania-key2.png`, `mania-key2D.png` | Inner idle/pressed | mirror |
| `mania-keyS.png`, `mania-keySD.png` | Centre idle/pressed | falls to `key1` |

**Only `keyND` convention** — no `key{N}-D.png` form. Animations not
supported on keys.

### 1.3 Stage frames

| File | Purpose |
| --- | --- |
| `mania-stage-left.png` | Left border, stretched to playfield height |
| `mania-stage-right.png` | Right border, stretched |
| `mania-stage-bottom.png` | Bottom plate; animatable; 0.625× stage width |
| `mania-stage-hint.png` | Judgement line; centred at `HitPosition` |
| `mania-stage-light.png` | Per-column press lighting; multiplicative; tinted by `ColourLightN` |
| `mania-warningarrow.png` | Pre-start countdown |

### 1.4 Impact lighting (additive)

| File | Purpose |
| --- | --- |
| `lightingN.png` (anim) | Flash on tap-hit + hold-tail; at judgement line |
| `lightingL.png` (anim) | Sustained for hold duration |

Width overridable per column via `LightingNWidth` / `LightingLWidth`.

### 1.5 HUD / judgments

| File | Animatable | Notes |
| --- | --- | --- |
| `mania-hit0.png` | Yes (60fps) | Miss; can be overridden per `Hit0:` in block |
| `mania-hit50.png` | Yes | 50 |
| `mania-hit100.png` | Yes | "Good" |
| `mania-hit200.png` | Yes | "Great" (mania-specific tier) |
| `mania-hit300.png` | Yes | "Perfect" |
| `mania-hit300g.png` | Yes | "Marvelous" / rainbow 300 |
| `comboburst-mania-N.png` | No (random-of-set) | Side decoration |

Global (mania consumes): `score-0..9.png`, `score-comma.png`,
`score-dot.png`, `score-percent.png`, `score-x.png`, `scorebar-bg.png`
+ `scorebar-colour-{n}.png` (rotated 90° CCW in mania).

### 1.6 Animation conventions

- Frame series `<base>-N.png` starts at N=0.
- Default fps: 60 for hitbursts/lightings; `LightFramePerSecond` for
  stage-light; `[General] AnimationFramerate` global override; `-1` =
  derive from frame count.
- Hitbursts play once, hold last frame, fade out.
- Bodies (`L`) loop only while held.

---

## 2. `skin.ini [Mania]` block reference

| Key | Type | Default | Per-keymode? | Effect |
| --- | --- | --- | --- | --- |
| `Keys` | int | (required) | (block scope) | Keycount this block configures (1-10, 12, 14, 16, 18) |
| `ColumnStart` | int @ 480-ref | 136 | Y | Left edge of column 1 on 640×480 ref |
| `ColumnRight` | int | 19 | Y | Right reserve |
| `ColumnSpacing` | csv ints | 0 | Y | Gap between cols (N−1 entries) |
| `ColumnWidth` | csv ints | 30 | Y | Width per column (N entries) |
| `ColumnLineWidth` | csv ints | 2 | Y | Per-divider thickness (N+1 entries) |
| `BarlineHeight` | float | 1.2 | Y | Measure-bar pixel thickness |
| `HitPosition` | int | 402 | Y | Y of judgement line on 480-ref |
| `LightPosition` | int | 413 | Y | Y of StageLight bottom |
| `ScorePosition` | int | — | Y | Y of hitburst popups |
| `ComboPosition` | int | — | Y | Y of combo counter |
| `JudgementLine` | 0/1 | 0 | Y | Extra line above StageHint |
| `LightFramePerSecond` | int | — | Y | StageLight animation fps |
| `SpecialStyle` | 0/1/2 | 0 | Y | 0=none, 1=outer/left-special, 2=inner/right-special |
| `ComboBurstStyle` | 0/1/2 \| L/R/Both | 1 | Y | Side for `comboburst-mania` |
| `SplitStages` | 0/1 | impl | Y | Force half-split (10K+) |
| `StageSeparation` | float | 40 | Y | Gap between split halves |
| `SeparateScore` | 0/1 | 1 | Y | Per-half hitbursts |
| `KeysUnderNotes` | 0/1 | 0 | Y | Draw key sprites below notes |
| `UpsideDown` | 0/1 | 0 | Y | Top-scroll like DDR |
| `KeyFlipWhenUpsideDown[N][D]` | 0/1 | 1 | Y per col/part | Per-column key flip overrides |
| `NoteFlipWhenUpsideDown[N][HLT]` | 0/1 | 1 | Y per col/part | Per-column note flip overrides |
| `NoteBodyStyle` (`[N]`) | 0/1/2 | 1 | Y (per col) | 0=stretch head, 1=cascade L strip, 2=stretch L |
| `WidthForNoteHeightScale` | float | min col width | Y | Canonical width for height scaling |
| `LightingNWidth` | csv floats | empty | Y | Per-col `lightingN` width |
| `LightingLWidth` | csv floats | empty | Y | Per-col `lightingL` width |
| `Colour#` | RGB(a) | 0,0,0,255 | Y per col | Lane background tint, 1-indexed |
| `ColourLight#` | RGB | 55,255,255 | Y per col | StageLight tint when pressed |
| `ColourColumnLine` | RGB(a) | white | Y | Divider colour (global to block) |
| `ColourBarline` | RGB(a) | white | Y | Measure-bar tint |
| `ColourJudgementLine` | RGB | white | Y | Tints the extra `JudgementLine=1` line |
| `ColourKeyWarning` | RGB | 0,0,0 | Y | Pre-start key-hint tint |
| `ColourHold` | RGB(a) | 255,191,51 | Y | Combo-counter tint during hold |
| `ColourBreak` | RGB | red | Y | Combo-counter tint on break |
| `KeyImage[N][D]` | path | per layout | Y per col | Receptor idle/pressed override |
| `NoteImage[N][HLT]` | path | per layout | Y per col, per part | Tap/head/body/tail override |
| `StageLeft`/`Right`/`Bottom`/`Hint`/`Light` | path | `mania-stage-*` | Y | Stage frame overrides |
| `LightingN` / `LightingL` | path | `lightingN.png` etc. | Y | Impact-flash overrides |
| `WarningArrow` | path | `mania-warningarrow.png` | Y | |
| `Hit0` … `Hit300g` | path | `mania-hit{N}.png` | Y | Judgement-popup overrides |

Quirks:

- `Colour#` is 1-indexed per spec. `KeyImage#` / `NoteImage#` are
  **ambiguous** — wiki text says 1-indexed but real skins use 0-indexed.
  osu-stable accepts both. **Recommendation:** accept both, prefer
  0-indexed if both present.
- `Colour` (no number) and `SpecialColour` are deprecated v1.x; ignore.

---

## 3. Per-keymode rules

- Multiple `[Mania]` blocks distinguished by `Keys: N`. Processed in
  order; each applies only to charts with exactly N keys.
- Supported `Keys` values: **1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18**.
- **No-match fallback:** osu-stable uses hard-coded defaults (column
  widths, positions per the wiki defaults). Mirror this — maintain
  per-keycount fallback tables.
- 1K/2K/3K are real (rare but supported).
- `SpecialStyle`:
  - `0` = no special.
  - `1` = "outer"/left-special: relocates `S` to the leftmost lane.
    For DP (`SplitStages`): special is outer lane of each half.
  - `2` = "inner"/right-special: mirror of 1.
  - Effective only for even keycounts ≥ 6 (others already have an
    `S` from the default layout).
- No inheritance between blocks. Each `Keys:N` block specifies its
  own `Colour1..N`, etc.

---

## 4. Colour / tinting rules

- `Colour#` (1-indexed): solid rect drawn behind each column. Default
  opaque black. RGBA allowed.
- `ColourLight#`: multiplied into `mania-stage-light` on press. Default
  `55,255,255`.
- `ColourColumnLine`: **one colour for all dividers** (not per-line),
  but dividers can have varying widths via `ColumnLineWidth`. Default
  white.
- `ColourBarline`: measure-bar tint. Default white.
- `ColourJudgementLine`: tints the extra line drawn when
  `JudgementLine=1`. The stage-hint sprite itself is not tinted.
- `ColourKeyWarning`: pre-start key-binding hint. P3, not worth it.
- `ColourHold`: combo-counter font tint while any hold is being pressed.
- `ColourBreak`: combo-counter font tint on combo break (flash to red).
- `Colour` (no number) and `SpecialColour`: deprecated, ignore.

Tinting semantics: multiplied as `sprite.rgb * colour.rgb`, alpha as
`sprite.a * colour.a`. Our shader already has per-instance colour
on the atlas pipeline.

---

## 5. Animation rules

- **Hitbursts** (`mania-hit{N}-0..N`): **60 fps, plays once, hold last
  frame during fade-out**. Wiki contradicts itself ("looped" in §mania,
  "does not loop" in §Interface); osu-framework's
  `LegacyManiaJudgementPiece` is one-shot-hold — that's the truth.
- **`mania-stage-bottom-{n}`**: rate = global `AnimationFramerate`.
- **`mania-stage-light-{n}`**: rate = `LightFramePerSecond`
  (per-keymode), loops while column pressed.
- **`mania-note{1,2,S}L-{n}`**: loops while held; freezes (last frame)
  on release.
- **Tap notes / heads / tails**: animatable; rate = global
  `AnimationFramerate`. Loops, phase = note-spawn time.
- **`lightingN-{n}`**: 60 fps, plays once. **`lightingL-{n}`**: 60 fps,
  loops for hold duration.
- **`comboburst-mania-{n}`**: NOT an animation — one-of-a-set, picked
  at random per combo milestone.
- **Frame-timing precedence:** explicit `LightFramePerSecond` >
  fixed (60fps for hitbursts/lightings) > `[General] AnimationFramerate`
  > derived (frames÷cycle).

---

## 6. Sound rules

Mania uses the same hitsound infrastructure as std; nothing
mania-specific.

- Per-beatmap sampleset: `{normal|soft|drum}-hit{normal|whistle|finish|clap}.{wav,ogg,mp3}`.
- Custom sampleset indexing: `{set}-hitnormal{N}.wav` for custom
  index N (N=0/1 → unsuffixed).
- `combobreak.wav`: played on miss when combo > threshold.
- Sample priority (highest first):
  1. Per-note custom sample (the note line's `customSampleFilename`)
  2. Beatmap-folder skin samples
  3. User-skin samples
  4. Default skin samples
  5. Silence
- `LayeredHitSounds` (default `true`, `[General]`): if true, additions
  (`finish`/`whistle`/`clap`) play **on top of** base hitnormal;
  if false, base is suppressed when an addition is set.

`hitsounds.py` already handles this for std-style; mania has no
different rules.

---

## 7. What danser-go does

danser **does not ship a mania renderer**. The only mania references
are for replay-frame and beatmap-DB compatibility. The
`assets/default-skin/skin.ini` includes full `[Mania]` blocks but
they're pure data, never consumed.

What danser **does** generically (`app/skin/skin.go`):

- **Asset resolution chain (`GetTextureSource`):**
  `BEATMAP > SKIN > FALLBACK > LOCAL(default)`. Tries `<name>@2x.png`
  first, then `<name>.png`. Misses cache as nil.
- **`@2x` precedence**: always tries `@2x.png` first; on success
  stores width/height as `image.W/2, image.H/2` so coordinate math
  stays in SD units.
- **Animation frames (`GetFrames`):** fetches `<name>-0`, `<name>-1`,
  … until miss. **All frames must come from the same source tier**
  (`GetMostSpecific`). Mixing a skin's `note-0` with the default's
  `note-1` is forbidden.
- **`skin.ini` parsing (`info.go`):** parses `[General]`, `[Colours]`,
  `[Fonts]`, some `[CatchTheBeat]`. **Does not parse `[Mania]` keys.**
- **Animation timing:** `info.GetFrameTime(frames) = 1000/AnimationFramerate`
  if set, else `1000/frame_count`.

So "parity with danser" = implement osu-stable mania behaviour using
danser's *texture resolution philosophy* (4-tier fallback, `@2x`-first,
same-source animation frames, INI parsing for global keys).

---

## 8. Current implementation status

### 8.1 Slots we support (`SPRITE_NAMES`, 32 atlas layers)

Tap notes (outer/inner/centre), hold heads, hold bodies, hold tails,
receptors (idle + pressed), stage (left/right/light/bottom/hint),
HUD (column_bg, bg_vignette, note_circle, hit_strip), judgments
(geki, 300, katu, 100, 50, miss).

### 8.2 Slots we don't support

- `mania-warningarrow.png`
- `lightingN.png` / `lightingL.png` (impact flashes — distinct from
  stage-light)
- `comboburst-mania.png`
- Per-column overrides (`NoteImage0..N`, `KeyImage0..N`)
- Animation frames for any slot (`-0`, `-1`, …)
- Score-font bitmap glyphs (`score-0..9` etc.) for combo

### 8.3 `skin.ini` we currently read

- `[Colours] Combo1..N` (combo colours)
- `[Mania] Keys: N` (block scoping)
- `[Mania] ColumnColour: r,g,b|r,g,b|...` — **non-standard, doesn't
  exist in the spec**. Real skins use `Colour1`, `Colour2`, … which
  we currently ignore. **This is the smoking gun for "skin doesn't
  change in output."**

We do NOT read:

- `ColumnStart`, `ColumnWidth`, `ColumnSpacing`, `ColumnLineWidth`
- `HitPosition`, `LightPosition`, `ScorePosition`, `ComboPosition`
- `Colour1..N`, `ColourLight1..N` (correct keys)
- `ColourHold`, `ColourBreak`, `ColourColumnLine`, `ColourBarline`
- `KeyImage*`, `NoteImage*`, `StageLeft/Right/Bottom/Hint/Light`
- `Hit0..Hit300g` (judgement path overrides)
- `SpecialStyle`, `UpsideDown`, `KeysUnderNotes`, `SplitStages`
- `NoteBodyStyle`, `WidthForNoteHeightScale`
- `[General] AnimationFramerate`, `[Fonts]` prefixes

### 8.4 Tinting / animations / per-keymode variation

- **Tinting from skin:** zero. Instance buffer has colour, hardcoded
  to white. Combo-colour tint only from `[Colours] Combo*` for note
  bodies.
- **Animations:** zero. No multi-frame loading anywhere.
- **Per-keymode variation:** zero in skinning. `column_variant()`
  returns `outer/inner/center` via a synthetic rule that **matches
  4K/5K/7K only**. **6K, 8K, 9K, 10K+ are wrong** vs the wiki table.

---

## 9. Gap-analysis matrix

| Feature | Spec | Our current | Priority | Hours |
| --- | --- | --- | --- | --- |
| Parse all `[Mania]` blocks per `Keys: N` | Required | Partial (Keys + bogus ColumnColour) | P0 | 4 |
| `Colour1..N` lane background tint | Required | No (wrong key) | P0 | 3 |
| `ColourLight1..N` per-col press tint | Required | No | P1 | 3 |
| `NoteImage{N,NH,NL,NT}` per-col overrides | Required | No (3-bucket only) | P0 | 8 |
| `KeyImage{N,ND}` per-col overrides | Required | No | P1 | 4 |
| Correct default key layout for 6/8/9K | Required | Wrong | P0 | 2 |
| `mania-hit{0,50,100,200,300,300g}` + frames | Required | Static only | P1 | 6 |
| `ColumnWidth/Start/Spacing` layout | Required | Hardcoded | P1 | 6 |
| `HitPosition/ScorePosition/ComboPosition` | Required | Hardcoded | P1 | 3 |
| `KeysUnderNotes` toggle | Required | No (always under) | P2 | 1 |
| Stage frame asset overrides | Required | Yes | — | 0 (done) |
| `mania-stage-light-{n}` animation | Required | No | P2 | 4 |
| `mania-warningarrow.png` | Optional | No | P3 | 2 |
| `lightingN/L` impact flashes | Required | No | P1 | 6 |
| `LightingN/LWidth` per-col widths | Optional | No | P2 | 2 |
| `comboburst-mania.png` | Optional | No | P3 | 4 |
| `NoteBodyStyle` (0/1/2) | Required (v2.5+) | Fixed | P2 | 6 |
| `SpecialStyle` (1/2) | Required | No | P2 | 4 |
| `UpsideDown` + flips | Required | No | P3 | 6 |
| `SplitStages` (10K+) | Required (10K+) | No | P2 | 8 |
| `@2x` precedence everywhere | Required | Yes | — | 0 (done) |
| Animation-frame discovery for notes | Required | No | P2 | 6 |
| `score-{0..9,comma,…}` bitmap font | Required | No (uses TTF) | P3 | 6 |
| `comboburst-mania-{n}` random-of-set | Optional | No | P3 | 3 |
| `[General] AnimationFramerate` | Required | No | P2 | 2 |
| `ColourHold`/`Break` combo tint | Optional | No | P3 | 2 |
| `ColourColumnLine/Barline` | Required | No | P1 | 2 |
| 4-tier resolution chain | Required | 2-tier | P1 | 4 |

**Totals:** P0 ~17 h · P0+P1 ~50 h · P0+P1+P2 ~75 h.

---

## 10. Recommended implementation order

### Phase A — "skin actually changes the output" (P0, ~1 week)

1. **Fix `skin_ini.py`** to parse the real keys: `Colour{N}`,
   `ColourLight{N}`, `ColourColumnLine`, `ColourBarline`, `ColourHold`,
   `ColourBreak`, `KeyImage{N}[D]`, `NoteImage{N}[HLT]`,
   `StageLeft/Right/Bottom/Hint/Light`, `Hit{0,50,100,200,300,300g}`.
   Drop the bogus `ColumnColour`. Output `ManiaSection` with explicit
   per-column dicts.
2. **Replace 3-bucket atlas keying** (`outer/inner/center`) with
   **per-column slots**. Either extend `SPRITE_NAMES` per-column
   (atlas grows with keycount), or keep 3-bucket + add per-column
   "override texture" array. Latter scales better for 18K.
3. **Fix default key layout** to match wiki table (§1.1). Drop
   `column_variant`'s synthetic rule; use per-keycount lookup.
4. **`Colour{N}` as lane background** via new instanced bg draw.
5. **`KeyImage{N}[D]`** for receptors instead of hardcoded.
6. **`Hit{0..300g}` overrides** for judgment popups (static first,
   animation in Phase C).

**Acceptance:** load a representative skin (Hide Note / RetroSkin),
render a 4K + 7K clip, see distinct per-column tints and per-column
note sprites.

### Phase B — stage, key sprites, judgment popups (P1, ~1 week)

1. `HitPosition`, `LightPosition`, `ScorePosition`, `ComboPosition`
   read + applied. Convert 480-height ref → render resolution.
2. `ColumnStart`, `ColumnWidth`, `ColumnSpacing` read. Replace
   `PLAYFIELD_X_FRAC` / `PLAYFIELD_W_FRAC` constants when skin specifies.
3. `ColourColumnLine` / `ColourBarline` applied to dividers/barlines.
4. `lightingN.png` / `lightingL.png` as new atlas slots; render at
   `HitPosition`.
5. **4-tier resolution chain** (BEATMAP > SKIN > FALLBACK > DEFAULT)
   — wire BEATMAP source by passing `beatmap_dir` to atlas loader.

### Phase C — column tints, hit-light, animations (P2, ~1 week)

1. `ColourLight{N}` multiplicative tint on stage-light press.
2. **Animation-frame discovery**: extend resolver to load
   `<base>-0.png`, `-1.png`, … as frame stack; atlas dimension =
   max frames; shader input = current frame index.
3. `[General] AnimationFramerate` + `LightFramePerSecond` wired in.
4. `NoteBodyStyle` (0/1/2): style 0 = stretch head; 1 = repeating
   `L` strip; 2 = stretch `L`.

### Phase D — per-keymode positions, animations, edges (P2/P3, ~1 week)

1. `SpecialStyle` (1/2): relocate `S` to outer/inner; reorder layout
   array.
2. `KeysUnderNotes`: draw-order toggle.
3. `UpsideDown` + flip toggles: vertical flip + per-sprite overrides.
4. `SplitStages` + `StageSeparation` + `SeparateScore`: render two
   playfields for 10K+.
5. `mania-warningarrow`, `comboburst-mania`, `ColourHold`/`Break`.

### Phase Z — won't do

- Pixel-exact peppy reproduction (sub-pixel divider alignment, exact
  font-metric overlap).
- `ColourKeyWarning` pre-start hint (3-sec pre-roll only).
- `HitCircleOverlayAboveNumber` (std-only).
- BEATMAP-source skin overrides for non-image samples (already done
  in `hitsounds.py`).

---

## 11. Open questions for the implementer

1. **`KeyImage#` / `NoteImage#` — 0 or 1-indexed?** Real skins commonly
   use 0-indexed; wiki text says 1-indexed; osu-stable accepts both.
   **Resolution:** accept both, prefer 0-indexed if both present.
2. **`@2x` precedence for skin.ini-named paths.** If `NoteImage3:
   my-note.png` and folder has both `my-note.png` + `my-note@2x.png`,
   `@2x` wins per danser's `loadTexture`. Mirror danser.
3. **Hitburst animation — looped or one-shot?** Wiki contradicts itself;
   osu-framework's `LegacyManiaJudgementPiece` is one-shot-hold.
   **Resolution:** one-shot, hold-last, 60fps.
4. **`SpecialStyle` for SP vs DP.** 6K/8K originally have no `S`;
   `SpecialStyle=1` adds one in outer position. Confirm by inspecting
   peppy's 6K-with-SpecialStyle output before shipping.
5. **`SplitStages` for keycounts 2–9.** Wiki: "Each keycount > 1 can
   be split." Treat as P3 except auto-split 10K+.
6. **Per-column `Colour{N}` alpha.** With alpha < 255, lane bg shows
   playfield bg through. Our renderer has no playfield bg → render
   opaque-equivalent against our scene bg.
7. **`ColumnLineWidth` count.** Examples show N+1 entries (outer-L,
   between, outer-R). Parse 1..N+1, pad with defaults if short.
8. **`comboburst-mania-{n}` numbering origin.** 0 or 1? Probe both.
9. **Animation source-tier strictness.** danser's `GetFrames` requires
   all frames from same tier. If skin has `note-0` but not `note-1`,
   drop to static. **Recommendation:** match danser.
10. **`AnimationFramerate = -1` semantics.** danser: `1000/frames` ms
    per frame. osu-stable: cycle ≈ frames × (1/60). Differ on long
    animations. **Resolution:** match danser for parity.
11. **What is `katu` (mania-hit100k)?** "Good"-tier judgement with extra
    ring (Japanese 'katsu'/200-tier). Sprite is `mania-hit100k.png`
    (lowercase). Our current fallback chain
    (`mania-hit100k.png, mania-hit100K.png, mania-hit200.png`) is
    correct ordering. **Note:** `katu` is awarded for the **200** tier,
    NOT 100 — verify scoring code matches the spec.

---

## Source references

- osu-wiki skin.ini: https://github.com/ppy/osu-wiki/blob/master/wiki/Skinning/skin.ini/en.md
- osu-wiki mania: https://github.com/ppy/osu-wiki/blob/master/wiki/Skinning/osu!mania/en.md
- osu-wiki interface: https://github.com/ppy/osu-wiki/blob/master/wiki/Skinning/Interface/en.md
- danser-go skin code: https://github.com/Wieku/danser-go/tree/master/app/skin
- danser-go default-skin: https://github.com/Wieku/danser-go/blob/master/assets/default-skin/skin.ini
- Our code: `osu_renderer/gpu/atlas.py`, `osu_renderer/skin_ini.py`
