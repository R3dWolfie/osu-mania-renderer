"""FrameContext — the `ctx` handed to every wiki render_fn.

It wraps the proven `FrameRenderer` (GL engine + per-column `SpriteAtlas`)
and carries the current frame's `SceneState`. Element render functions draw
through it; painter order is the registry's `RENDER_ORDER`.

Built once per render, mutated per frame (scene/t_ms/frame_n are re-bound —
no per-frame allocation). `persistent` holds cross-frame element state
(hold-body tiling phase, popup pools, light ages) keyed by element name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import moderngl


@dataclass
class FrameContext:
    fr: Any                       # FrameRenderer (the reused GL engine)
    skin: Any                     # SkinPair (variable tier; atlas owns images)
    gl: moderngl.Context
    fbo: moderngl.Framebuffer
    width: int
    height: int
    key_count: int
    # per-frame (re-bound each frame)
    scene: Any = None
    t_ms: int = 0
    frame_n: int = 0
    # cross-frame element state
    persistent: dict[str, dict] = field(default_factory=dict)

    # ---- atlas (image truth) ----
    @property
    def atlas(self):
        return self.fr.atlas

    @property
    def mania_section(self):
        return self.fr.mania_section

    @property
    def skin_ini(self):
        return self.fr.skin_ini

    @property
    def options(self):
        return self.fr.options

    # ---- geometry passthrough (computed by FrameRenderer) ----
    @property
    def col_x(self):
        return self.fr.col_x

    @property
    def col_w(self):
        return self.fr.col_w

    @property
    def pf_x(self):
        return self.fr.pf_x

    @property
    def pf_w(self):
        return self.fr.pf_w

    @property
    def col_w_uniform(self):
        return self.fr.col_w_uniform

    @property
    def receptor_centre_y_gl(self):
        return self.fr.receptor_centre_y_gl

    @property
    def upside_down(self):
        return self.fr.upside_down

    @property
    def combo_baseline_y_gl(self):
        return self.fr.combo_baseline_y_gl

    @property
    def score_popup_y_gl(self):
        return self.fr.score_popup_y_gl

    def persist(self, element: str) -> dict:
        """Get-or-create the persistent bag for an element."""
        bag = self.persistent.get(element)
        if bag is None:
            bag = {}
            self.persistent[element] = bag
        return bag

    # ---- frame lifecycle ----
    def begin_frame(self) -> None:
        """Clear the FBO and set the standard alpha blend — mirrors the head
        of FrameRenderer.draw()."""
        self.fr._stage_clock_ms = self.scene.t_ms
        self.fbo.use()
        self.fbo.clear(0.03, 0.03, 0.05, 1.0)
        self.gl.enable(moderngl.BLEND)
        self.gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    # ---- draw API (delegates into the engine's instanced batch) ----
    def draw_sprite(self, name: str, x, y, w, h, tint) -> None:
        self.fr._draw_sprite(name, x, y, w, h, tint)

    def draw_sprite_idx(self, idx: int, x, y, w, h, tint) -> None:
        self.fr._draw_sprite_idx(idx, x, y, w, h, tint)

    def draw_column_sprite(self, kind: str, col: int, x, y, w, h, tint) -> None:
        idx = self.atlas.column_slot_index(kind, col)
        self.fr._draw_sprite_idx(idx, x, y, w, h, tint)

    def draw_external(self, tex, x, y, w, h, alpha) -> None:
        self.fr._draw_external_texture(tex, x, y, w, h, alpha)

    def draw_direct(
        self,
        name,
        x,
        y,
        w,
        h,
        tint=(1.0, 1.0, 1.0, 1.0),
        *,
        frame_index: int | None = None,
    ) -> None:
        """Full-resolution direct draw for wide sprites (scorebar / stage
        panels) — bypasses the layered atlas so they stay crisp."""
        self.fr._draw_direct(
            name, int(x), int(y), int(w), int(h), tint,
            frame_index=frame_index,
        )

    def text(self, s, size, color):
        """Rasterize text → (GL texture, w, h), cached. Text rasterization
        stays a FrameRenderer primitive; elements own layout."""
        return self.fr._cached_text(s, size, color)

    def set_note_fx(self, hidden: bool, fade_in: bool, combo: int = 0) -> None:
        # Delegate to the shared cover computation so Hidden/FadeIn scale
        # with combo here exactly as in the monolithic draw path.
        self.fr.apply_note_cover(hidden, fade_in, combo)

    def flush(self) -> None:
        self.fr._flush_sprite_batch()

    # ---- skin number fonts (score-*.png / combo-*.png glyph composition) ----
    # char → glyph suffix. Prefixed by font ("score" / "combo") into the slot.
    _GLYPH_SUFFIX = {
        "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
        "6": "6", "7": "7", "8": "8", "9": "9", ",": "comma", ".": "dot",
        "%": "percent", "x": "x",
    }

    def _glyph_slot(self, ch: str, font: str) -> str | None:
        """Atlas slot for char `ch` in `font` ("score"/"combo"/"argon"). Combo
        glyphs fall back to the score font when the combo font lacks that glyph
        (e.g. combo has no comma/percent). The "argon" font is lazer's
        bundled "argon-counter" glyph set, used for the Argon default HUD."""
        suf = self._GLYPH_SUFFIX.get(ch)
        if suf is None:
            return None
        if font == "argon":
            # argon-counter has no comma glyph; the Argon score counter draws
            # plain digits (no grouping) so this never arises in practice.
            return f"argon_{suf}" if suf != "comma" else None
        if font == "combo":
            slot = f"combo_{suf}"
            if self.atlas.global_source(slot) in ("user", "beatmap", "bundle"):
                return slot
            # fall through to score for glyphs the combo font doesn't define
        return f"score_{suf}"

    def has_score_font(self) -> bool:
        """True when the user skin supplies the digit glyphs (score-0..9).
        The HUD uses skin digits when present and falls back to the
        Argon/PIL readout otherwise — 'if it's in the .osk, use it'."""
        a = self.atlas
        return all(
            a.global_source(f"score_{d}") == "user" for d in range(10)
        )

    def has_combo_font(self) -> bool:
        """True when the skin supplies a real combo font (combo-0..9 as user).
        When ComboPrefix isn't set, combo_* resolve to the score font, so this
        also returns True whenever the score font is present."""
        a = self.atlas
        return all(
            a.global_source(f"combo_{d}") == "user" for d in range(10)
        )

    def has_argon_font(self) -> bool:
        """True when the bundled argon-counter glyphs are available (they ship
        with the renderer, so this is normally always True)."""
        a = self.atlas
        return all(
            a.global_source(f"argon_{d}") in ("user", "beatmap", "bundle")
            for d in range(10)
        )

    @staticmethod
    def _eff_overlap(overlap_px: int, gw: float) -> float:
        """Per-glyph overlap clamped so a narrow glyph (e.g. the `.` at 52px
        vs the 240px digit boxes) isn't pulled past its own width into a
        negative advance. Caps at 45% of the glyph's native width."""
        return min(overlap_px, gw * 0.45)

    def _number_glyphs(self, glyph_h: float, overlap_px: int, font: str,
                       wireframe: bool) -> dict:
        """Per-character layout of a number font at one size: everything
        `number_width` and `draw_number` need for a character that does not
        depend on where the number is drawn. A number is drawn ~8 times a
        frame and each character used to cost ~8 lookups; here it is worked
        out once per (size, overlap, font, wireframe) and character, with the
        same arithmetic in the same order, so every position is the same
        float it was. Kept per atlas: another atlas starts a new table."""
        atlas = self.atlas
        tables = getattr(self, "_number_tables", None)
        if tables is None or tables[0] is not atlas:
            tables = self._number_tables = (atlas, {})
        key = (glyph_h, overlap_px, font, wireframe)
        table = tables[1].get(key)
        if table is None:
            # an animated number (the combo's pop) has a new size every frame:
            # its tables are used once, so do not let them pile up
            if len(tables[1]) >= 64:
                tables[1].clear()
            table = tables[1][key] = _NumberGlyphs(self, glyph_h, overlap_px, font, wireframe)
        return table

    def number_width(self, text: str, glyph_h: float, overlap_px: int,
                     font: str = "score") -> float:
        """On-screen width of `text` drawn with the given number font at
        `glyph_h` digit height — for right/centre alignment."""
        glyphs = self._number_glyphs(glyph_h, overlap_px, font, False)
        total = 0.0
        prev_gap = 0.0
        for ch in text:
            g = glyphs[ch]
            if g is None:
                continue
            total += g[0] - prev_gap
            prev_gap = g[1]
        return total

    def draw_number(
        self, text: str, *, x: float, center_y: float, glyph_h: float,
        overlap_px: int = 0, align: str = "left", alpha: float = 1.0,
        font: str = "score", wireframe: bool = False,
        tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> float:
        """Compose `text` from the skin's score-font glyphs.

        All glyphs share one scale (digit height → `glyph_h`); score/combo
        digits use the authored ``5`` width while punctuation keeps its own
        width. Legacy skin glyphs draw from native direct textures; Argon's
        bundled counter continues through the shared atlas.
        `x` is the left/right/centre anchor per `align`. Returns total width.

        `wireframe=True` (argon font only) substitutes lazer's "wireframes"
        backing glyph for every digit (the dim segmented template behind the
        live counter), keeping the same advance so it aligns under the real
        digits. The dot keeps its own glyph (lazer's wireframesLookup)."""
        total_w = self.number_width(text, glyph_h, overlap_px, font)
        if align == "right":
            pen_x = x - total_w
        elif align == "center":
            pen_x = x - total_w / 2
        else:
            pen_x = x
        rgba = (tint[0], tint[1], tint[2], alpha)
        glyphs = self._number_glyphs(glyph_h, overlap_px, font, wireframe)
        fr = self.fr
        for ch in text:
            g = glyphs[ch]
            if g is None:
                continue
            # (cell_width, gap, advance, half_cell, direct, what, half_w, half_h, w, h)
            cx = pen_x + g[3]
            if g[4]:
                fr._draw_direct(
                    g[5],
                    int(round(cx - g[6])),
                    int(round(center_y - g[7])),
                    g[8],
                    g[9],
                    rgba,
                )
            else:
                fr._draw_sprite_idx(
                    g[5], int(round(cx - g[6])), int(round(center_y - g[7])),
                    g[8], g[9], rgba,
                )
            pen_x += g[2]
        return total_w



class _NumberGlyphs(dict):
    """character -> layout tuple for one number font at one size (see
    FrameContext._number_glyphs), or None for a character the font does not
    draw. Filled on first use of each character.

    The tuple is (cell_width, gap, advance, half_cell, direct, what, half_w,
    half_h, w, h): `what` is the direct-texture slot name when `direct`, the
    atlas layer index otherwise. Every value is the expression the per-glyph
    code computed, so sums and rounded positions come out the same."""

    def __init__(self, ctx, glyph_h, overlap_px, font, wireframe):
        super().__init__()
        self._ctx = ctx
        self._overlap_px, self._font, self._wireframe = overlap_px, font, wireframe
        atlas = ctx.atlas
        digit_h = atlas.global_native_size(f"{font}_0")[1] or 1
        self._scale = glyph_h / digit_h
        self._fixed_digit_width = (
            atlas.global_native_size(f"{font}_5")[0]
            if font in ("score", "combo")
            else 0.0
        )

    def __missing__(self, ch):
        ctx, font, scale = self._ctx, self._font, self._scale
        atlas = ctx.atlas
        slot = ctx._glyph_slot(ch, font)
        if slot is None:
            self[ch] = None
            return None
        gw, _gh = atlas.global_native_size(slot)   # advance from real glyph
        cell_width = (
            self._fixed_digit_width * scale
            if ch.isdigit() and self._fixed_digit_width > 0
            else gw * scale
        )
        gap = ctx._eff_overlap(self._overlap_px, gw) * scale
        draw_slot = slot
        if self._wireframe and font == "argon":
            draw_slot = "argon_dot" if ch == "." else "argon_wireframes"
        dw, dh = atlas.global_native_size(draw_slot)
        if font in ("score", "combo") and not self._wireframe:
            draw_width = dw * scale
            draw_height = dh * scale
            g = (cell_width, gap, cell_width - gap, cell_width / 2, True, draw_slot,
                 draw_width / 2, draw_height / 2,
                 max(1, int(round(draw_width))), max(1, int(round(draw_height))))
        else:
            q = max(dw, dh) * scale
            g = (cell_width, gap, cell_width - gap, cell_width / 2, False,
                 atlas.index_of(draw_slot), q / 2, q / 2, int(round(q)), int(round(q)))
        self[ch] = g
        return g
