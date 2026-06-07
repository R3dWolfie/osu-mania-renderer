"""Bundled sprite PNGs → ModernGL texture array.

The atlas has two regions:

  Global slots
      Stage frames, judgement popups, particle/HUD assets. Looked up by
      name (`atlas.index_of("stage_left")`).

  Per-column slots
      Note (tap/head/body/tail) and receptor (off/on) sprites. There is
      one layer per column per kind so each column can have its own
      bundled appearance OR its own skin override. Looked up by kind +
      column (`atlas.column_slot_index("note_tap", col)`).

Per-column sprite resolution priority:

  1. Skin's explicit override from `skin.ini` `NoteImage{N}` /
     `KeyImage{N}[D]` (per-column path).
  2. Skin's conventional file for the column's default "kind" — picked
     from the osu! wiki's per-keycount default layout table (1=outer,
     2=inner, S=centre).
  3. Bundled role-named PNG that ships with the renderer.
  4. 4×4 transparent placeholder.
"""
from __future__ import annotations

import logging
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from osu_mania_renderer_v2.skin_ini import ManiaSection

SPRITES_DIR = Path(__file__).resolve().parent.parent / "assets" / "sprites"
_LOG = logging.getLogger("osu_mania_renderer_v2")


# ===== Default per-keycount column layout =====
#
# Each entry maps column index → bundled-sprite kind:
#   "1" = outer  (mania-note1.png / mania-key1.png)
#   "2" = inner  (mania-note2.png / mania-key2.png)
#   "S" = centre (mania-noteS.png / mania-keyS.png)
#
# Source: osu! wiki, Skinning/osu!mania. For odd K the centre column
# is S. For even K the centre pair doubles up the same kind (K mod 4
# == 2 → 1-1, K mod 4 == 0 → 2-2) and everything else alternates
# outer/inner from the edges in.
DEFAULT_LAYOUT: dict[int, tuple[str, ...]] = {
    1:  ("S",),
    2:  ("1", "1"),
    3:  ("1", "S", "1"),
    4:  ("1", "2", "2", "1"),
    5:  ("1", "2", "S", "2", "1"),
    6:  ("1", "2", "1", "1", "2", "1"),
    7:  ("1", "2", "1", "S", "1", "2", "1"),
    8:  ("1", "2", "1", "2", "2", "1", "2", "1"),
    9:  ("1", "2", "1", "2", "S", "2", "1", "2", "1"),
    10: ("1", "2", "1", "2", "1", "1", "2", "1", "2", "1"),
    12: ("1", "2", "1", "2", "1", "2", "2", "1", "2", "1", "2", "1"),
    14: ("1", "2", "1", "2", "1", "2", "1", "1", "2", "1", "2", "1", "2", "1"),
    16: ("1", "2", "1", "2", "1", "2", "1", "2", "2", "1", "2", "1", "2", "1", "2", "1"),
    18: ("1", "2", "1", "2", "1", "2", "1", "2", "1", "1", "2", "1", "2", "1", "2", "1", "2", "1"),
}


def default_column_kind(column: int, key_count: int) -> str:
    """Bundled-sprite kind ("1"/"2"/"S") for a column in K-key mode."""
    if key_count in DEFAULT_LAYOUT:
        return DEFAULT_LAYOUT[key_count][column]
    # Unusual K (11/13/etc): edges = outer, centre on odd = S, alternate
    # everywhere else. Spec doesn't define these formally — we follow
    # the spirit of the convention.
    if column == 0 or column == key_count - 1:
        return "1"
    if key_count % 2 == 1 and column == key_count // 2:
        return "S"
    return "2" if column % 2 == 1 else "1"


def effective_column_kind(
    column: int, key_count: int, special_style: int | None = None,
) -> str:
    """Column kind after applying SpecialStyle relocation.

    SpecialStyle is meaningful only for even keycounts ≥ 6 (odd K
    already has S in the centre; smaller K has no inner columns to
    swap with). Values:

      0  → default (no relocation)
      1  → "outer" / left-special: col 0 becomes the S (special) lane
      2  → "inner" / right-special: col K-1 becomes the S lane

    The rest of the layout is unchanged. Per peppy's reference docs,
    skins that set SpecialStyle expect S-themed assets at the outer
    column instead of the central one.
    """
    base = default_column_kind(column, key_count)
    if special_style not in (1, 2):
        return base
    if key_count < 6 or key_count % 2 != 0:
        return base
    if special_style == 1 and column == 0:
        return "S"
    if special_style == 2 and column == key_count - 1:
        return "S"
    return base


def column_variant(column: int, key_count: int) -> str:
    """Legacy classification ("outer"/"inner"/"center") for callers that
    haven't yet migrated to `default_column_kind`. Now routes through
    the wiki-correct layout table."""
    kind = default_column_kind(column, key_count)
    return {"1": "outer", "2": "inner", "S": "center"}[kind]


# ===== Global slot file map =====

_GLOBAL_SKIN_FILE_MAP: dict[str, tuple[str, ...]] = {
    "stage_left":      ("mania-stage-left.png",),
    "stage_right":     ("mania-stage-right.png",),
    "stage_light":     ("mania-stage-light.png",),
    "playfield_frame": ("mania-stage-bottom.png",),
    # `hit_light` = mania-stage-hint, the persistent judgement-line
    # decoration. Distinct from the per-press lighting_n flash below.
    "hit_light":       ("mania-stage-hint.png",),
    # Animated impact flash at hit moment (and at hold tail). Skin
    # convention varies — try the mania-prefixed name first, then the
    # bare `lightingN`, then the older `lighting` fallback.
    "lighting_n":      ("mania-lightingN.png", "lightingN.png", "lighting.png"),
    # Sustained flash during a hold. Loops at AnimationFramerate.
    "lighting_l":      ("mania-lightingL.png", "lightingL.png"),
    "judgment_geki":   ("mania-hit300g.png", "mania-hit300G.png"),
    "judgment_300":    ("mania-hit300.png",),
    "judgment_katu":   ("mania-hit100k.png", "mania-hit100K.png", "mania-hit200.png"),
    "judgment_100":    ("mania-hit100.png",),
    "judgment_50":     ("mania-hit50.png",),
    "judgment_miss":   ("mania-hit0.png",),
    # Score font (score/combo/accuracy digits). osu! draws mania score and
    # combo with the `score-*` glyphs; accuracy reuses them with the percent
    # glyph. Each is an independent global slot resolved user→bundle.
    "score_0":         ("score-0.png",),
    "score_1":         ("score-1.png",),
    "score_2":         ("score-2.png",),
    "score_3":         ("score-3.png",),
    "score_4":         ("score-4.png",),
    "score_5":         ("score-5.png",),
    "score_6":         ("score-6.png",),
    "score_7":         ("score-7.png",),
    "score_8":         ("score-8.png",),
    "score_9":         ("score-9.png",),
    "score_comma":     ("score-comma.png",),
    "score_dot":       ("score-dot.png",),
    "score_percent":   ("score-percent.png",),
    "score_x":         ("score-x.png",),
    # Combo font — filenames built from [Fonts] ComboPrefix at load time (the
    # map entries are placeholders; _resolve_global rewrites combo_* per the
    # prefix, default "score" so combo falls back to the score font).
    "combo_0":         ("combo-0.png",),
    "combo_1":         ("combo-1.png",),
    "combo_2":         ("combo-2.png",),
    "combo_3":         ("combo-3.png",),
    "combo_4":         ("combo-4.png",),
    "combo_5":         ("combo-5.png",),
    "combo_6":         ("combo-6.png",),
    "combo_7":         ("combo-7.png",),
    "combo_8":         ("combo-8.png",),
    "combo_9":         ("combo-9.png",),
    "combo_x":         ("combo-x.png",),
    # Health bar (scorebar). New style = scorebar-marker present; old style
    # uses scorebar-ki / kidanger / kidanger2 for the marker. scorebar-colour
    # animates as scorebar-colour-0..N.
    "scorebar_bg":         ("scorebar-bg.png",),
    "scorebar_colour":     ("scorebar-colour.png",),
    "scorebar_marker":     ("scorebar-marker.png",),
    "scorebar_ki":         ("scorebar-ki.png",),
    "scorebar_kidanger":   ("scorebar-kidanger.png",),
    "scorebar_kidanger2":  ("scorebar-kidanger2.png",),
}

# Canonical order — indexes are stable, callers reference by name via
# `atlas.index_of`.
GLOBAL_SPRITE_NAMES: tuple[str, ...] = (
    "stage_left",
    "stage_right",
    "stage_light",
    "hit_light",
    "column_bg",
    "playfield_frame",
    "bg_vignette",
    "note_circle",
    "hit_strip",
    "lighting_n",
    "lighting_l",
    "judgment_geki",
    "judgment_300",
    "judgment_katu",
    "judgment_100",
    "judgment_50",
    "judgment_miss",
    # Argon default-skin pieces (white shapes, tinted per-column at draw).
    # No user-skin filename map → always resolve to the bundle PNG.
    "argon_note_body",
    "argon_note_glyph",
    "argon_col_glow",
    "argon_key_pill",
    "argon_key_dots",
    # Argon counter font (lazer's "argon-counter" — individual glyph textures
    # at Gameplay/Fonts/argon-counter-{lookup}, fetched from ppy/osu-resources).
    # Bundle-only (no skin file map) → always resolve to the bundled PNG; used
    # for the Argon default HUD score / accuracy / combo digits.
    "argon_0", "argon_1", "argon_2", "argon_3", "argon_4",
    "argon_5", "argon_6", "argon_7", "argon_8", "argon_9",
    "argon_dot", "argon_percent", "argon_x", "argon_wireframes",
    # Argon score banner (wedge) + HP tube + leaderboard card — drawn direct.
    "argon_wedge",
    "argon_hp",
    "argon_card",
    # Score font glyphs (user-skin score-*.png; no bundle fallback — the
    # Argon/PIL HUD path is used when the skin omits them).
    "score_0",
    "score_1",
    "score_2",
    "score_3",
    "score_4",
    "score_5",
    "score_6",
    "score_7",
    "score_8",
    "score_9",
    "score_comma",
    "score_dot",
    "score_percent",
    "score_x",
    # Combo font glyphs (separate from score when ComboPrefix is set).
    "combo_0", "combo_1", "combo_2", "combo_3", "combo_4",
    "combo_5", "combo_6", "combo_7", "combo_8", "combo_9", "combo_x",
    # Health bar (scorebar) glyphs — user-skin only (no bundle); the Argon
    # default health bar is drawn procedurally when these are absent.
    "scorebar_bg",
    "scorebar_colour",
    "scorebar_marker",
    "scorebar_ki",
    "scorebar_kidanger",
    "scorebar_kidanger2",
    # Mod-icon hexagon (bundle PNG, tinted per mod type at draw time).
    "mod_hex",
)


# ===== Per-column slot file map =====

PER_COLUMN_KINDS: tuple[str, ...] = (
    "note_tap",
    "note_hold_head",
    "note_hold_body",
    "note_hold_tail",
    "receptor_off",
    "receptor_on",
)

# Per-column slots that support multi-frame animation discovery.
# Receptors are NOT animated per peppy's reference (the wiki explicitly
# notes "Animations not supported on keys").
_ANIMATABLE_PER_COLUMN_KINDS: frozenset[str] = frozenset({
    "note_tap",
    "note_hold_head",
    "note_hold_body",
    "note_hold_tail",
})

# Default conventional filename(s) for (kind, col_kind). Tried in order
# inside the skin dir; first hit wins.
_PER_COLUMN_DEFAULT_FILES: dict[tuple[str, str], tuple[str, ...]] = {
    ("note_tap",       "1"): ("mania-note1.png",),
    ("note_tap",       "2"): ("mania-note2.png", "mania-note1.png"),
    ("note_tap",       "S"): ("mania-noteS.png", "mania-note1.png"),
    ("note_hold_head", "1"): ("mania-note1H.png", "mania-note1.png"),
    ("note_hold_head", "2"): ("mania-note2H.png", "mania-note2.png", "mania-note1.png"),
    ("note_hold_head", "S"): ("mania-noteSH.png", "mania-noteS.png", "mania-note1.png"),
    ("note_hold_body", "1"): ("mania-note1L.png",),
    ("note_hold_body", "2"): ("mania-note2L.png", "mania-note1L.png"),
    ("note_hold_body", "S"): ("mania-noteSL.png", "mania-note1L.png"),
    ("note_hold_tail", "1"): ("mania-note1T.png", "mania-note1H.png", "mania-note1.png"),
    ("note_hold_tail", "2"): ("mania-note2T.png", "mania-note2H.png", "mania-note2.png"),
    ("note_hold_tail", "S"): ("mania-noteST.png", "mania-noteSH.png", "mania-noteS.png"),
    ("receptor_off",   "1"): ("mania-key1.png",),
    ("receptor_off",   "2"): ("mania-key2.png", "mania-key1.png"),
    ("receptor_off",   "S"): ("mania-keyS.png", "mania-key1.png"),
    ("receptor_on",    "1"): ("mania-key1D.png", "mania-key1.png"),
    ("receptor_on",    "2"): ("mania-key2D.png", "mania-key2.png", "mania-key1D.png"),
    ("receptor_on",    "S"): ("mania-keySD.png", "mania-keyS.png", "mania-key1D.png"),
}

# Bundled fallback PNG stems for (kind, col_kind). These ship with the
# renderer, used when neither a skin override nor a conventional file
# is available.
_BUNDLED_FALLBACK_STEM: dict[tuple[str, str], str] = {
    ("note_tap",       "1"): "note_tap_outer",
    ("note_tap",       "2"): "note_tap_inner",
    ("note_tap",       "S"): "note_tap_center",
    ("note_hold_head", "1"): "note_hold_head_outer",
    ("note_hold_head", "2"): "note_hold_head_inner",
    ("note_hold_head", "S"): "note_hold_head_center",
    ("note_hold_body", "1"): "note_hold_body_outer",
    ("note_hold_body", "2"): "note_hold_body_inner",
    ("note_hold_body", "S"): "note_hold_body_center",
    ("note_hold_tail", "1"): "note_hold_tail_outer",
    ("note_hold_tail", "2"): "note_hold_tail_inner",
    ("note_hold_tail", "S"): "note_hold_tail_center",
    ("receptor_off",   "1"): "receptor_off_outer",
    ("receptor_off",   "2"): "receptor_off_inner",
    ("receptor_off",   "S"): "receptor_off_center",
    ("receptor_on",    "1"): "receptor_on_outer",
    ("receptor_on",    "2"): "receptor_on_inner",
    ("receptor_on",    "S"): "receptor_on_center",
}


def _per_column_override(
    section: ManiaSection | None, kind: str, col: int,
) -> str | None:
    """Return the skin-author's per-column path for (kind, col), if any."""
    if section is None:
        return None
    table_map = {
        "note_tap":       section.note_image,
        "note_hold_head": section.note_image_h,
        "note_hold_body": section.note_image_l,
        "note_hold_tail": section.note_image_t,
        "receptor_off":   section.key_image,
        "receptor_on":    section.key_image_d,
    }
    return table_map[kind].get(col)


# Atlas layer dimensions — sprites are letterboxed to fit. Square so
# circular notes and square receptors don't get squashed when drawn into
# a col_w × col_w screen rect.
LAYER_W = 256
LAYER_H = 256


class SpriteAtlas:
    """Packs sprites into a single Texture2DArray.

    Layout:
        [0 .. len(GLOBAL_SPRITE_NAMES))        named global slots
        [G .. G + N_KINDS * key_count)         per-column slots,
                                                 indexed by kind_idx * K + col
    """

    def __init__(self, key_count: int) -> None:
        self.key_count = key_count
        self._global_indices: dict[str, int] = {}
        self._global_count = 0
        # (kind, col) → "beatmap" | "user" | "bundle" | "missing". Lets
        # the renderer decide whether to draw real skin note sprites or
        # fall back to the synthesized circle look.
        self._column_sources: dict[tuple[str, int], str] = {}
        # Per-column slot base layer indices. Filled during load.
        # column_slot_index() now reads from this dict instead of
        # computing kind_idx * K + col, because animated columns may
        # occupy variable layer ranges.
        self._column_indices: dict[tuple[str, int], int] = {}
        self._column_frames: dict[tuple[str, int], int] = {}
        # Per-column source aspect ratio (width / height) of the
        # original sprite, before letterboxing. Used by the renderer's
        # cascade-body tiling so the L sprite repeats at its natural
        # aspect instead of being stretched.
        self._column_aspects: dict[tuple[str, int], float] = {}
        self._column_native_sizes: dict[tuple[str, int], tuple[int, int]] = {}
        # Full-resolution RGBA images kept OUTSIDE the layered atlas for wide
        # sprites (scorebar, stage panels) — the 256² atlas tile would crush
        # a 1366-wide health bar. These are drawn as direct textures at native
        # resolution so they stay crisp regardless of skin. Stored in DESIGN
        # orientation (PIL top-row-first); native px (÷2 if @2x baked here too).
        self._direct_images: dict[str, Image.Image] = {}
        # Global slot sources, parallel to _column_sources.
        self._global_sources: dict[str, str] = {}
        # Per-global source aspect ratio (width / height) of the
        # original sprite, before letterboxing into the atlas layer.
        # Used by `_draw_stage_decorations` so wide / tall stage chrome
        # (e.g. Night05's 1200×770 starfield, Aristia's 800×770 edge
        # strip) renders at native aspect against the playfield instead
        # of being non-uniformly stretched into the layer square.
        self._global_aspects: dict[str, float] = {}
        # Per-global native pixel dimensions (w, h) before letterboxing.
        # Distinct from aspect because the renderer also needs to
        # distinguish "meaningfully sized sprite" from "1x1 transparent
        # placeholder" — many skins ship a placeholder pixel so the
        # file resolves to "user" source without actually contributing
        # visible chrome.
        self._global_native_sizes: dict[str, tuple[int, int]] = {}
        # Animation frame counts. Slots without an entry → single-frame.
        # global_frames[slot_name] = N; the slot's index_of returns
        # frame-0's layer index, frames 1..N-1 follow consecutively.
        self._global_frames: dict[str, int] = {}
        self.texture_array: moderngl.TextureArray | None = None

    @classmethod
    def load(
        cls,
        ctx: moderngl.Context,
        *,
        key_count: int,
        skin_dir: Path | None = None,
        beatmap_dir: Path | None = None,
        mania_section: ManiaSection | None = None,
        combo_prefix: str = "score",
    ) -> SpriteAtlas:
        """Build the atlas. Sprite resolution tries beatmap_dir first
        (per-map overrides), then skin_dir, then bundled fallback. This
        matches danser's BEATMAP > SKIN > FALLBACK > LOCAL chain.

        `combo_prefix` ([Fonts] ComboPrefix, default "score") selects the
        combo number font files (`<prefix>-N.png`) so skins with a separate
        combo font (Night05's `combo-N`) render combo with the right glyphs."""
        atlas = cls(key_count)
        layers: list[np.ndarray] = []
        from_beatmap = 0
        from_user = 0
        from_bundle = 0
        from_missing = 0
        from_anim = 0

        # Global slots (indexed by name). Animated slots occupy multiple
        # contiguous layers; `index_of(slot)` returns frame 0's layer
        # and frame counts live in `_global_frames`.
        layer_idx = 0
        for name in GLOBAL_SPRITE_NAMES:
            frames, src = cls._resolve_global(
                name, skin_dir=skin_dir, beatmap_dir=beatmap_dir,
                section=mania_section, combo_prefix=combo_prefix,
            )
            atlas._global_indices[name] = layer_idx
            atlas._global_sources[name] = src
            # Capture native aspect + native size from the FIRST frame
            # before letterboxing — same convention as per-column slots.
            # Skipped on the missing/placeholder case where frames is
            # empty (the loop below wouldn't push any layers either).
            if frames:
                fw, fh = frames[0].size
                atlas._global_aspects[name] = (
                    fw / fh if fh > 0 else 1.0
                )
                # Native size in DESIGN units = pixels / ScaleAdjust (@2x → /2),
                # matching lazer's Texture.DisplaySize. All on-screen sizing
                # uses design units, so @2x and @1x skins render identically.
                sa = frames[0].info.get("scale_adjust", 1)
                atlas._global_native_sizes[name] = (fw / sa, fh / sa)
                # Keep full-res image for wide sprites drawn directly (crisp).
                if name in _DIRECT_DRAW_SLOTS:
                    atlas._direct_images[name] = frames[0]
            if len(frames) > 1:
                atlas._global_frames[name] = len(frames)
                from_anim += 1
            # Wide global sprites drawn to a quad of their NATIVE aspect must
            # stretch-fill the atlas tile — letterboxing a wide sprite (e.g.
            # mania-stage-bottom 5:1, scorebar-bg 15:1, scorebar-colour 10:1)
            # into the square tile then drawing to a wide quad shrinks it to a
            # thin sliver. Stretch → draw-to-native-aspect un-distorts it.
            # (stage_left/right stay letterboxed: they use the square-quad
            # trick to stay full-height at the edges.)
            g_fit = (_fit_stretch
                     if name in ("playfield_frame", "scorebar_bg", "scorebar_colour",
                                 "stage_left", "stage_right",
                                 "judgment_geki", "judgment_300", "judgment_katu",
                                 "judgment_100", "judgment_50", "judgment_miss",
                                 # Argon note body/glyph: non-square (1.43:1);
                                 # stretch so the note renders at lazer's
                                 # 60:42 aspect, not letterboxed-squished.
                                 "argon_note_body", "argon_note_glyph",
                                 # Argon key pill (22:14) + dots (22:17) are
                                 # baked to fill their canvas at the lazer
                                 # aspect — stretch so they render full-size.
                                 "argon_key_pill", "argon_key_dots")
                     else _fit_letterbox)
            for frame in frames:
                layers.append(np.asarray(
                    g_fit(frame, LAYER_W, LAYER_H), dtype=np.uint8,
                ))
                layer_idx += 1
            if src == "beatmap":
                from_beatmap += 1
            elif src == "user":
                from_user += 1
            elif src == "bundle":
                from_bundle += 1
            else:
                from_missing += 1
        atlas._global_count = layer_idx

        # Per-column slots. Each (kind, col) occupies M consecutive
        # layers where M is the resolved frame count. Atlas tracks the
        # base layer index per (kind, col); animated slots also live in
        # _column_frames.
        for kind in PER_COLUMN_KINDS:
            for col in range(key_count):
                frames, src = cls._resolve_column(
                    kind=kind, col=col, key_count=key_count,
                    skin_dir=skin_dir, beatmap_dir=beatmap_dir,
                    section=mania_section,
                )
                atlas._column_sources[(kind, col)] = src
                atlas._column_indices[(kind, col)] = layer_idx
                # Source aspect — use the first frame's dimensions
                # (animation frames are typically the same size).
                first_w, first_h = frames[0].size
                atlas._column_aspects[(kind, col)] = (
                    first_w / first_h if first_h > 0 else 1.0
                )
                # Design units = pixels / ScaleAdjust (@2x → /2), per lazer.
                sa = frames[0].info.get("scale_adjust", 1)
                atlas._column_native_sizes[(kind, col)] = (first_w / sa, first_h / sa)
                if len(frames) > 1:
                    atlas._column_frames[(kind, col)] = len(frames)
                    from_anim += 1
                if src == "beatmap":
                    from_beatmap += 1
                elif src == "user":
                    from_user += 1
                elif src == "bundle":
                    from_bundle += 1
                else:
                    from_missing += 1
                # ALL per-column sprites stretch-fill their atlas tile. The
                # square atlas would otherwise letterbox a non-square sprite
                # (Vio notes 50×30, keys 50×107) into the tile, and drawing
                # that to a sized quad shrinks the sprite (short notes, narrow
                # keys). The consumer draws each to a quad of the intended
                # shape — notes/heads/tails to (col_w, col_w × native_h/native_w)
                # = lazer's uniform scale to column width; keys to (col_w,
                # native_h × height/768) = lazer's X-stretch-to-column; bodies
                # to the hold extent — so stretch-fill reproduces all of them.
                fit_fn = _fit_stretch
                for frame in frames:
                    layers.append(np.asarray(
                        fit_fn(frame, LAYER_W, LAYER_H), dtype=np.uint8,
                    ))
                    layer_idx += 1

        total = len(layers)
        _LOG.info(
            "atlas_loaded slots=%d from_beatmap=%d from_user_skin=%d "
            "from_bundle=%d from_missing=%d animated_slots=%d "
            "key_count=%d skin_dir=%s beatmap_dir=%s",
            total, from_beatmap, from_user, from_bundle, from_missing,
            from_anim,
            key_count, str(skin_dir) if skin_dir else "(none)",
            str(beatmap_dir) if beatmap_dir else "(none)",
        )
        # Surface the same diagnostic via stdout NDJSON so the worker's
        # log-streamer picks it up and the bot's queue watcher forwards
        # it. Without this, `from_user` counts are invisible outside the
        # process — making it impossible to diagnose "custom skin
        # rendered as bundled" complaints from job artifacts alone. The
        # `per_column_sources` payload itemises every (kind, col) slot
        # so we can see EXACTLY which sprites the user's skin contributed.
        # Import datetime/json locally to keep the atlas module's
        # top-level imports unchanged.
        try:
            import datetime as _dt
            import json as _json
            import sys as _sys
            per_col = {}
            for (kind, col), src in atlas._column_sources.items():
                per_col[f"{kind}/{col}"] = src
            event = {
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(
                    timespec="seconds") + "Z",
                "stage": "atlas_loaded",
                "level": "info",
                "msg": (
                    f"slots={total} from_beatmap={from_beatmap} "
                    f"from_user={from_user} from_bundle={from_bundle} "
                    f"from_missing={from_missing} key_count={key_count}"
                ),
                "slots":          total,
                "from_beatmap":   from_beatmap,
                "from_user":      from_user,
                "from_bundle":    from_bundle,
                "from_missing":   from_missing,
                "animated_slots": from_anim,
                "key_count":      key_count,
                "skin_dir":       str(skin_dir) if skin_dir else None,
                "beatmap_dir":    str(beatmap_dir) if beatmap_dir else None,
                "global_sources": dict(atlas._global_sources),
                "per_column_sources": per_col,
            }
            print(_json.dumps(event), flush=True, file=_sys.stdout)
        except Exception:  # noqa: BLE001
            # Diagnostics are best-effort; the render itself must not
            # fail because the JSON emit broke.
            pass

        arr = np.stack(layers, axis=0)
        atlas.texture_array = ctx.texture_array(
            size=(LAYER_W, LAYER_H, total),
            components=4,
            data=arr.tobytes(),
        )
        atlas.texture_array.build_mipmaps()
        atlas.texture_array.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        return atlas

    # Backwards-compat constructors. New code should call load() directly.
    @classmethod
    def load_default(cls, ctx: moderngl.Context, key_count: int = 4) -> SpriteAtlas:
        return cls.load(ctx, key_count=key_count)

    @classmethod
    def load_from_skin_dir(
        cls,
        ctx: moderngl.Context,
        skin_dir: Path,
        key_count: int = 4,
        mania_section: ManiaSection | None = None,
        beatmap_dir: Path | None = None,
    ) -> SpriteAtlas:
        return cls.load(
            ctx, skin_dir=skin_dir, key_count=key_count,
            mania_section=mania_section, beatmap_dir=beatmap_dir,
        )

    def index_of(self, name: str) -> int:
        """Layer index for a named global slot (frame 0 for animated)."""
        return self._global_indices[name]

    def frame_count(self, name: str) -> int:
        """Number of animation frames for a global slot. Returns 1 for
        single-frame slots."""
        return self._global_frames.get(name, 1)

    def global_source(self, name: str) -> str:
        """`"beatmap"` / `"user"` / `"bundle"` / `"missing"` for a global
        slot."""
        return self._global_sources.get(name, "missing")

    def direct_image(self, name: str):
        """Full-resolution PIL image for a wide direct-draw slot (scorebar /
        stage panels), or None. Drawn outside the layered atlas to stay crisp."""
        return self._direct_images.get(name)

    def global_aspect(self, name: str) -> float:
        """Source-image aspect ratio (width / height) for a global slot,
        before letterboxing into the atlas layer. Returns 1.0 if the
        slot wasn't loaded. Used by stage-chrome positioning so
        wide/tall sprites render at native aspect."""
        return self._global_aspects.get(name, 1.0)

    def global_native_size(self, name: str) -> tuple[int, int]:
        """Native (width, height) in pixels of the source sprite
        before letterboxing. Returns (0, 0) if the slot wasn't loaded.
        Used to distinguish meaningful skin chrome from 1×1 transparent
        placeholders that many skins ship to satisfy file-existence
        checks without contributing visible art."""
        return self._global_native_sizes.get(name, (0, 0))

    def column_slot_index(self, kind: str, col: int) -> int:
        """Layer index of frame 0 for a per-column slot."""
        if kind not in PER_COLUMN_KINDS:
            raise KeyError(f"unknown per-column kind: {kind!r}")
        if not 0 <= col < self.key_count:
            raise IndexError(
                f"column {col} out of range for key_count={self.key_count}"
            )
        try:
            return self._column_indices[(kind, col)]
        except KeyError:
            raise IndexError(
                f"column slot ({kind!r}, {col}) not loaded — "
                "atlas may not have been built yet"
            ) from None

    def column_frame_count(self, kind: str, col: int) -> int:
        """Number of animation frames for a per-column slot. Returns 1
        for static slots."""
        return self._column_frames.get((kind, col), 1)

    def column_aspect(self, kind: str, col: int) -> float:
        """Source-image aspect ratio (width / height) for a per-column
        slot, before letterboxing into the atlas. Returns 1.0 if the
        slot wasn't loaded. Used by cascade-body tiling so the L sprite
        repeats at its natural aspect instead of being stretched."""
        return self._column_aspects.get((kind, col), 1.0)

    def column_native_size(self, kind: str, col: int) -> tuple[int, int]:
        """Native (width, height) in pixels of a per-column slot's source
        image, before letterboxing. Returns (0, 0) if not loaded. Used by
        the legacy key area, whose height = the texture's native pixel
        height (osu-pixels) rather than an aspect-scale of column width."""
        return self._column_native_sizes.get((kind, col), (0, 0))

    def column_source(self, kind: str, col: int) -> str:
        """`"user"` if the column's sprite came from the skin, `"bundle"`
        if from the renderer's bundled fallback, `"missing"` otherwise."""
        return self._column_sources.get((kind, col), "missing")

    def has_skin_notes(self) -> bool:
        """True if any per-column note sprite is registered — user
        skin, beatmap override, OR bundled default. The synthesized
        `note_circle` capsule is only the fallback when NO sprite is
        registered for any column. Including "bundle" matches the
        intent of `has_skin_note(col)` and ensures the bundled note
        art ships through instead of being shadowed by the capsule."""
        return any(
            self._column_sources.get(("note_tap", c)) in (
                "user", "beatmap", "bundle",
            )
            for c in range(self.key_count)
        )

    def has_skin_note(self, col: int) -> bool:
        """True when this column has a usable tap-note sprite — from
        the user's skin, the beatmap-tier override, OR the bundled
        default. The capsule fallback (`note_circle`) is only for
        cases where NO sprite is registered at all. Treating the
        bundled sprites as second-class produced a worse look than
        the capsule even though the bundled assets are dedicated
        hold/tap art."""
        return self._column_sources.get(("note_tap", col)) in (
            "user", "beatmap", "bundle",
        )

    def has_skin_hold(self, col: int) -> bool:
        """True when the skin (or the bundled default) authored ALL
        three hold parts — head + body + tail — for this column.
        Mixed/partial skins still fall through to the renderer's
        capsule fallback so a head-only skin doesn't render a
        Frankensteined rectangular body."""
        return all(
            self._column_sources.get((kind, col)) in ("user", "beatmap", "bundle")
            for kind in ("note_hold_head", "note_hold_body", "note_hold_tail")
        )

    @staticmethod
    def _resolve_global(
        slot: str,
        *,
        skin_dir: Path | None,
        beatmap_dir: Path | None,
        section: ManiaSection | None,
        combo_prefix: str = "score",
    ) -> tuple[list[Image.Image], str]:
        """Pick the right PNG(s) for a global atlas slot.

        Returns a frame list (length 1 for static slots, ≥ 1 for
        animated). Priority chain: beatmap_dir (per-map overrides) →
        skin's explicit section override → skin's conventional file →
        bundled fallback → 4×4 transparent placeholder.

        Animation discovery (`<base>-0.png`, …) runs per tier for
        animatable slots so per-skin animations are honoured. All
        frames must come from the same tier (danser's rule)."""
        candidates = _GLOBAL_SKIN_FILE_MAP.get(slot, ())
        # Combo glyphs use [Fonts] ComboPrefix (default "score" → fall back to
        # the score font). e.g. slot "combo_5" → "<prefix>-5.png".
        if slot.startswith("combo_"):
            glyph = slot[len("combo_"):]
            candidates = (f"{combo_prefix}-{glyph}.png",)
        animatable = slot in _ANIMATABLE_GLOBAL_SLOTS

        # Per-map (BEATMAP tier).
        if beatmap_dir is not None:
            if animatable:
                frames = _try_animation_frames(beatmap_dir, candidates)
                if frames:
                    return frames, "beatmap"
            for candidate in candidates:
                img = _try_skin_file(beatmap_dir, candidate)
                if img is not None:
                    return [img], "beatmap"
        # Skin section override (explicit path; static only — skin.ini
        # named paths don't have a -0 convention).
        if skin_dir is not None and section is not None:
            override = _global_section_override(section, slot)
            if override is not None:
                img = _try_skin_override(skin_dir, override)
                if img is not None:
                    return [img], "user"
        # Skin conventional filename(s).
        if skin_dir is not None:
            if animatable:
                frames = _try_animation_frames(skin_dir, candidates)
                if frames:
                    return frames, "user"
            for candidate in candidates:
                img = _try_skin_file(skin_dir, candidate)
                if img is not None:
                    return [img], "user"
        # Bundled role-named PNG (always single-frame).
        bundled = SPRITES_DIR / f"{slot}.png"
        if bundled.exists():
            return [Image.open(bundled).convert("RGBA")], "bundle"
        return [Image.new("RGBA", (4, 4), (0, 0, 0, 0))], "missing"

    @staticmethod
    def _resolve_column(
        *,
        kind: str,
        col: int,
        key_count: int,
        skin_dir: Path | None,
        beatmap_dir: Path | None,
        section: ManiaSection | None,
    ) -> tuple[list[Image.Image], str]:
        """Pick the right PNG(s) for a per-column slot. Returns a frame
        list (length 1 for static, ≥ 1 for animated). See
        `_resolve_global` for the priority chain — additionally, the
        per-column skin.ini override (`NoteImage{N}` / `KeyImage{N}`)
        takes precedence over both beatmap and skin conventional files
        (the author explicitly named the file, so honour it).

        Hold-tail orientation: stable mania renders any non-T file used
        as a tail (i.e. a head/tap fallback when the skin doesn't ship
        `mania-note*T.png`) vertically flipped. We replicate that
        convention so partial skins still produce a proper-looking
        tail cap instead of two head-shaped caps."""
        special_style = section.special_style if section is not None else None
        col_kind = effective_column_kind(col, key_count, special_style)
        candidates = _PER_COLUMN_DEFAULT_FILES.get((kind, col_kind), ())
        animatable = kind in _ANIMATABLE_PER_COLUMN_KINDS
        is_tail = kind == "note_hold_tail"

        def _maybe_flip_tail(frames: list[Image.Image], src_filename: str) -> list[Image.Image]:
            # lazer's LegacyHoldNoteTailPiece INVERTS the scroll direction, so
            # the tail is always flipped vertically relative to the head for a
            # downward-scrolling stage — both for real `*T` tails (e.g. Night05's
            # flat-top mania-note1T → rounded-top cap) and head-as-tail
            # fallbacks. (UpsideDown stages, handled separately, would not flip.)
            if not is_tail:
                return frames
            return [f.transpose(Image.FLIP_TOP_BOTTOM) for f in frames]

        # Skin's explicit per-column override (named path).
        if skin_dir is not None:
            override = _per_column_override(section, kind, col)
            if override is not None:
                if animatable:
                    frames = _try_animation_frames(skin_dir, (f"{override}.png",))
                    if frames:
                        # Tail override paths name the actual file the
                        # author wants used — `note_image_t` is the T
                        # slot so no flip. If somehow `note_image` (tap)
                        # was used as tail (shouldn't happen since the
                        # consumer table picks the right dict per kind)
                        # we'd flip — but the consumer doesn't, so this
                        # is moot.
                        return frames, "user"
                img = _try_skin_override(skin_dir, override)
                if img is not None:
                    return [img], "user"

            # Per the osu! spec, when NoteImageNH / NoteImageNT is
            # *specified* but the file is missing, the head/tail
            # falls back to the tap sprite (NoteImageN). Some skins
            # rely on this (e.g. minimaly ships mania/upH.png for 3 of
            # 4 columns but accidentally omitted col 2's `upH.png` —
            # without this fallback the renderer drops all the way to
            # the bundled circle, producing a stray blue blob mid-frame
            # where col 2's hold head should be).
            if kind in ("note_hold_head", "note_hold_tail"):
                tap_override = _per_column_override(section, "note_tap", col)
                if tap_override is not None:
                    if animatable:
                        frames = _try_animation_frames(
                            skin_dir, (f"{tap_override}.png",),
                        )
                        if frames:
                            return _maybe_flip_tail(frames, tap_override), "user"
                    img = _try_skin_override(skin_dir, tap_override)
                    if img is not None:
                        return (
                            _maybe_flip_tail([img], tap_override), "user",
                        )

        # Per-map override (BEATMAP tier). Per candidate, try animation
        # then static — exhausting one candidate's variants before
        # moving to the next preserves the skin author's per-column
        # intent (e.g. `mania-note2.png` static beats `mania-note1-N.png`
        # animated as a fallback for col 1).
        if beatmap_dir is not None:
            for candidate in candidates:
                if animatable:
                    frames = _try_animation_frames(beatmap_dir, (candidate,))
                    if frames:
                        return _maybe_flip_tail(frames, candidate), "beatmap"
                img = _try_skin_file(beatmap_dir, candidate)
                if img is not None:
                    return _maybe_flip_tail([img], candidate), "beatmap"

        # Skin's conventional file — same per-candidate precedence.
        if skin_dir is not None:
            for candidate in candidates:
                if animatable:
                    frames = _try_animation_frames(skin_dir, (candidate,))
                    if frames:
                        return _maybe_flip_tail(frames, candidate), "user"
                img = _try_skin_file(skin_dir, candidate)
                if img is not None:
                    return _maybe_flip_tail([img], candidate), "user"

        # Bundled role PNG (single-frame, already in correct orientation).
        stem = _BUNDLED_FALLBACK_STEM.get((kind, col_kind))
        if stem is not None:
            bundled = SPRITES_DIR / f"{stem}.png"
            if bundled.exists():
                return [Image.open(bundled).convert("RGBA")], "bundle"

        return [Image.new("RGBA", (4, 4), (0, 0, 0, 0))], "missing"


def _filename_is_T_variant(name: str) -> bool:
    """True when `name` looks like a tail file (`mania-noteNT.png`).
    Used to skip the auto-flip on real T-variant assets."""
    if "." not in name:
        return False
    stem, _ext = name.rsplit(".", 1)
    # Match common T-variants: `mania-note1T`, `mania-noteST`, etc.
    return stem.endswith("T") and any(c.isdigit() or c in "SLR" for c in stem[-3:-1])


def _global_section_override(section: ManiaSection, slot: str) -> str | None:
    """Map an atlas slot name to the ManiaSection field that overrides it."""
    return {
        "stage_left":       section.stage_left,
        "stage_right":      section.stage_right,
        "stage_light":      section.stage_light,
        "playfield_frame":  section.stage_bottom,
        "hit_light":        section.stage_hint,
        "judgment_geki":    section.hit_300g,
        "judgment_300":     section.hit_300,
        "judgment_katu":    section.hit_200,
        "judgment_100":     section.hit_100,
        "judgment_50":      section.hit_50,
        "judgment_miss":    section.hit_0,
    }.get(slot)


# ===== Animation-aware resolution =====

# Slots that support multi-frame animation discovery. Skin authors
# author frames as `<base>-0.png`, `<base>-1.png`, ... etc. We
# currently animate only judgement popups; stage-light and other
# animatable slots are single-frame for now (Phase C+).
# Wide sprites drawn as full-resolution direct textures (bypass the layered
# atlas, which would crush them). scorebar-bg/colour + the stage panels.
_DIRECT_DRAW_SLOTS: frozenset[str] = frozenset({
    "scorebar_bg", "scorebar_colour", "stage_left", "stage_right",
    "playfield_frame",
    "argon_wedge",   # wide score banner — crisp at native resolution.
    "argon_hp",      # glossy HP tube — crisp + stretches to fill.
    "argon_card",    # rounded leaderboard / avatar card — tinted at draw.
})


_ANIMATABLE_GLOBAL_SLOTS: frozenset[str] = frozenset({
    "judgment_geki",
    "judgment_300",
    "judgment_katu",
    "judgment_100",
    "judgment_50",
    "judgment_miss",
    "stage_light",     # looped per press, at LightFramePerSecond (default 60).
    "lighting_n",      # one-shot per hit, at 60fps.
    "lighting_l",      # looped during hold, at AnimationFramerate.
    "scorebar_colour", # HP fill; skins ship scorebar-colour-0..N.
})


def _try_animation_frames(
    skin_dir: Path, candidates: tuple[str, ...],
) -> list[Image.Image]:
    """Look for `<base>-0.png`, `<base>-1.png`, … in `skin_dir`.

    Per danser's rule, all frames must come from the same source tier.
    We pick the first candidate that has at least a `-0` frame and load
    the contiguous run from there. Returns the frame list (empty if
    none found)."""
    for cand in candidates:
        if "." in cand:
            stem, ext = cand.rsplit(".", 1)
        else:
            stem, ext = cand, "png"
        first = _try_skin_file(skin_dir, f"{stem}-0.{ext}")
        if first is None:
            continue
        frames: list[Image.Image] = [first]
        for n in range(1, 256):    # hard cap; no skin ships >256 frames
            nxt = _try_skin_file(skin_dir, f"{stem}-{n}.{ext}")
            if nxt is None:
                break
            frames.append(nxt)
        return frames
    return []


def _ci_lookup(base: Path, rel: str) -> Path | None:
    """Resolve a relative path under `base` case-INSENSITIVELY. osu! skins are
    authored on Windows (case-insensitive FS), so a skin.ini `ComboPrefix:
    Combo` legitimately points at `combo-0.png`; on Linux that mismatch makes
    the file vanish. Exact match is the fast path; otherwise each path
    component is matched ignoring case."""
    p = base / rel
    if p.is_file():
        return p
    cur = base
    for part in rel.replace("\\", "/").split("/"):
        nxt = cur / part
        if nxt.exists():
            cur = nxt
            continue
        match = None
        try:
            for entry in cur.iterdir():
                if entry.name.lower() == part.lower():
                    match = entry
                    break
        except OSError:
            return None
        if match is None:
            return None
        cur = match
    return cur if cur.is_file() else None


def _try_skin_file(skin_dir: Path, filename: str) -> Image.Image | None:
    """Try `<skin>/<stem>@2x.<ext>` then `<skin>/<filename>` (case-insensitive)."""
    if "." in filename:
        stem, ext = filename.rsplit(".", 1)
    else:
        stem, ext = filename, "png"
    for candidate in (f"{stem}@2x.{ext}", filename):
        path = _ci_lookup(skin_dir, candidate)
        if path is not None:
            try:
                img = Image.open(path).convert("RGBA")
                # lazer ScaleAdjust: an @2x asset's design size is pixels/2.
                img.info["scale_adjust"] = 2 if "@2x." in candidate else 1
                return img
            except (OSError, ValueError):
                continue
    return None


def _try_skin_override(skin_dir: Path, ref: str) -> Image.Image | None:
    """Resolve a skin.ini path reference (`NoteImage1: mania/note-left`).

    osu! uses backslashes and conventionally omits the extension. We
    normalise separators, accept an explicit extension if present,
    otherwise try .png + @2x.png.
    """
    ref = ref.replace("\\", "/").strip()
    if not ref:
        return None
    leaf = ref.rsplit("/", 1)[-1]
    if "." in leaf:
        stem = ref.rsplit(".", 1)[0]
    else:
        stem = ref
    for candidate in (f"{stem}@2x.png", f"{stem}.png"):
        path = _ci_lookup(skin_dir, candidate)
        if path is not None:
            try:
                img = Image.open(path).convert("RGBA")
                img.info["scale_adjust"] = 2 if "@2x." in candidate else 1
                return img
            except (OSError, ValueError):
                continue
    return None


def _fit_letterbox(img: Image.Image, w: int, h: int) -> Image.Image:
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    src_w, src_h = img.size
    scale = min(w / src_w, h / src_h)
    new = img.resize(
        (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
        Image.LANCZOS,
    )
    canvas.paste(new, ((w - new.width) // 2, (h - new.height) // 2), new)
    return canvas


def _fit_stretch(img: Image.Image, w: int, h: int) -> Image.Image:
    """Resize `img` to exactly (w, h), NOT preserving aspect ratio.

    Used for hold-body L sprites which osu!mania designs explicitly to
    be stretched non-uniformly to fill a hold's vertical extent. Wide,
    short L sprites (FNF's 158x43 hold-arrow strips, Aristia/Cinia/NRW's
    256x82 hold capsules) need the entire atlas tile to BE the sprite —
    not centered with transparent padding above and below — so that when
    the renderer later stretches the tile to the per-frame body rect
    the whole rect fills with sprite content. Letterbox padding turned
    those rects into 30%-filled stubs surrounded by transparency, which
    looked like the hold body was a tiny vertical sliver."""
    return img.resize((max(1, w), max(1, h)), Image.LANCZOS)
