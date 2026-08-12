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
        """Clear the FBO and set the compositor's standard alpha blend."""
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

    def draw_direct(self, name, x, y, w, h, tint=(1.0, 1.0, 1.0, 1.0)) -> None:
        """Full-resolution direct draw for wide sprites (scorebar / stage
        panels) — bypasses the layered atlas so they stay crisp."""
        self.fr._draw_direct(name, int(x), int(y), int(w), int(h), tint)

    def text(self, s, size, color):
        """Rasterize text → (GL texture, w, h), cached. Text rasterization
        stays a FrameRenderer primitive; elements own layout."""
        return self.fr._cached_text(s, size, color)

    def set_note_fx(self, hidden: bool, fade_in: bool, combo: int = 0) -> None:
        # Delegate to the shared cover computation so Hidden/FadeIn scale
        # with combo consistently across note elements.
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

    def number_width(self, text: str, glyph_h: float, overlap_px: int,
                     font: str = "score") -> float:
        """On-screen width of `text` drawn with the given number font at
        `glyph_h` digit height — for right/centre alignment."""
        digit_h = self.atlas.global_native_size(f"{font}_0")[1] or 1
        scale = glyph_h / digit_h
        total = 0.0
        prev_gap = 0.0
        for ch in text:
            slot = self._glyph_slot(ch, font)
            if slot is None:
                continue
            gw, _gh = self.atlas.global_native_size(slot)
            total += gw * scale - prev_gap
            prev_gap = self._eff_overlap(overlap_px, gw) * scale
        return total

    def draw_number(
        self, text: str, *, x: float, center_y: float, glyph_h: float,
        overlap_px: int = 0, align: str = "left", alpha: float = 1.0,
        font: str = "score", wireframe: bool = False,
        tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> float:
        """Compose `text` from the skin's score-font glyphs.

        All glyphs share one scale (digit height → `glyph_h`); comma/dot/
        percent keep their native proportions, exactly like osu!. Glyphs
        are letterboxed into square atlas layers, so each is drawn into a
        square quad sized to its long edge and centred on its slot — the
        transparent padding makes the visible glyph land at native aspect.
        `x` is the left/right/centre anchor per `align`. Returns total width.

        `wireframe=True` (argon font only) substitutes lazer's "wireframes"
        backing glyph for every digit (the dim segmented template behind the
        live counter), keeping the same advance so it aligns under the real
        digits. The dot keeps its own glyph (lazer's wireframesLookup)."""
        digit_h = self.atlas.global_native_size(f"{font}_0")[1] or 1
        scale = glyph_h / digit_h
        total_w = self.number_width(text, glyph_h, overlap_px, font)
        if align == "right":
            pen_x = x - total_w
        elif align == "center":
            pen_x = x - total_w / 2
        else:
            pen_x = x
        rgba = (tint[0], tint[1], tint[2], alpha)
        for ch in text:
            slot = self._glyph_slot(ch, font)
            if slot is None:
                continue
            gw, gh = self.atlas.global_native_size(slot)   # advance from real glyph
            vis_w = gw * scale
            draw_slot = slot
            if wireframe and font == "argon":
                draw_slot = "argon_dot" if ch == "." else "argon_wireframes"
            dw, dh = self.atlas.global_native_size(draw_slot)
            q = max(dw, dh) * scale            # square quad = glyph long edge
            cx = pen_x + vis_w / 2
            idx = self.atlas.index_of(draw_slot)
            self.fr._draw_sprite_idx(
                idx, int(round(cx - q / 2)), int(round(center_y - q / 2)),
                int(round(q)), int(round(q)), rgba,
            )
            pen_x += vis_w - self._eff_overlap(overlap_px, gw) * scale
        return total_w
