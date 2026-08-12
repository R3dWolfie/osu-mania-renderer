"""Lower-level ModernGL primitives used by the canonical compositor."""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from osu_mania_renderer_v2.beatmap.models import RenderOptions
from osu_mania_renderer_v2.beatmap.skin_ini import parse_skin_ini
from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas, column_variant
from osu_mania_renderer_v2.gpu.shaders import load_programs
from osu_mania_renderer_v2.gpu.text import text_to_texture
from osu_mania_renderer_v2.render.dim import build_dim_envelope
from osu_mania_renderer_v2.render.scene import SceneState

log = logging.getLogger("osu_mania_renderer_v2")

# 6 vertices x 9 float32 attrs for the ad-hoc external-texture quads.
# Little-endian f32 == the np.array(dtype="f4").tobytes() it replaces.
_EXT_QUAD_PACK = struct.Struct("<54f").pack

# Lazer-style defaults — match ppy/osu LegacyManiaSkinConfiguration.cs.
# Values are in stable osu! reference coords (480-tall, 4:3 letterbox),
# converted to render pixels via `px_per_ref = render_height / 480`.
# Per the spec, when a skin doesn't define `[Mania] Keys: K` for our
# chart's key count, we fabricate fresh defaults rather than
# closest-matching another section — lazer does the same (LegacySkin.cs
# Mania branch instantiates a new LegacyManiaSkinConfiguration(K)).
LAZER_DEFAULT_COLUMN_SIZE_REF = 42      # widened to match Argon's playfield proportions
                                        # (lazer centres columns in 16:9; our 4:3 letterbox
                                        # made the field read too narrow at the 30-ref value)
LAZER_DEFAULT_COLUMN_SPACING_REF = 0    # legacy default (Argon spacing set in argon branch)
LAZER_DEFAULT_COLUMN_LINE_WIDTH_REF = 2 # Hairline divider
# Receptor strip lives at the bottom; ≈ column-width tall so receptors are
# rendered close to circular.
RECEPTOR_HEIGHT_REL_COL = 1.0
RECEPTOR_BOTTOM_OFFSET_FRAC = 0.05
@dataclass
class RenderContext:
    ctx: moderngl.Context
    fbo: moderngl.Framebuffer
    width: int
    height: int
    key_count: int


class FrameRenderer:
    # Per-instance: x, y, w, h, atlas_idx, r, g, b, a = 9 floats.
    _FLOATS_PER_INSTANCE = 9
    _INSTANCE_CAP = 4096     # safety cap; auto-flushes if exceeded.

    def __init__(
        self,
        rc: RenderContext,
        options: RenderOptions | None = None,
        skin_dir: Path | None = None,
        beatmap_dir: Path | None = None,
        first_note_ms: int = 0,
        note_starts: tuple = (),
        breaks: tuple = (),
        approach_ms: int | None = None,
        rate: float = 1.0,
    ) -> None:
        self.rc = rc
        # Settings-page toggles. Pass options to gate optional HUD draws.
        self.options = options or RenderOptions(resolution=(rc.width, rc.height), fps=60)
        self.first_note_ms = first_note_ms
        # Background dim envelope (std's DimEnvelope, ported in dim.py): the
        # dim GLIDES intro→game as the first note begins its scroll-in
        # (approach_ms = RenderPlan.effective_approach_ms), brightens into
        # [Events] breaks and re-dims at the resume anchor — smoothstep over
        # the same 900 ms std/catch use. Built only when the orchestrator
        # provides note_starts (render.py / compositor.py); constructions
        # without them (unit tests, tools) keep the legacy 600 ms linear
        # intro ramp in _draw_background.
        self._dim_env = None
        if note_starts:
            _game = (self.options.bg_dim_game
                     if self.options.bg_dim_game is not None
                     else self.options.background_dim)
            _intro = (self.options.bg_dim_intro
                      if self.options.bg_dim_intro is not None else _game)
            _breaks = (self.options.bg_dim_breaks
                       if self.options.bg_dim_breaks is not None else _game)
            self._dim_env = build_dim_envelope(
                _intro, _game, _breaks, note_starts,
                float(approach_ms if approach_ms is not None else 600),
                breaks or ())
        # lazer's BreakOverlay (countdown + progress bar + CURRENT PROGRESS
        # info + slide-in chevrons) — gpu/break_overlay.py, a 1:1 port of
        # osu.Game/Screens/Play/BreakOverlay.cs on this engine's GL quad
        # primitives (the catch d8ccb60 rollout). Fed the SAME real-time
        # break periods the dim envelope gets; `rate` (plan.audio_rate)
        # converts them — and each frame's clock — back to the map-time
        # axis lazer's transforms run on. None on no-break maps; all GL
        # objects bake lazily on the first visible break frame, so
        # no-break renders never touch GL state here (byte-identical).
        self._break_overlay = None
        if breaks:
            from osu_mania_renderer_v2.gpu.break_overlay import LazerBreakOverlay
            self._break_overlay = LazerBreakOverlay(self, breaks, rate)
        # R3D intro splash (show_logo) textures — baked lazily on the first
        # frame the splash is visible, so flag-off renders never touch them.
        self._logo_tex: moderngl.Texture | None = None
        self._logo_glow_tex: moderngl.Texture | None = None
        self.programs = load_programs(rc.ctx)
        # 4-tier sprite resolution: BEATMAP > SKIN > FALLBACK > DEFAULT.
        # Each map's folder can ship per-map skin overrides (rare but
        # supported — e.g. boss maps with custom note art). Atlas tries
        # `beatmap_dir/<file>` first, then `skin_dir/<file>`, then the
        # bundled fallback. skin.ini parsing only ever looks at skin_dir
        # (per-map skin.ini doesn't exist in stable).
        self.skin_ini = None
        self.mania_section = None
        bm_dir = beatmap_dir if (beatmap_dir is not None and beatmap_dir.is_dir()) else None
        sk_dir = skin_dir if (skin_dir is not None and skin_dir.is_dir()) else None
        if sk_dir is not None:
            self.skin_ini = parse_skin_ini(sk_dir)
            self.mania_section = self.skin_ini.mania_for_keycount(rc.key_count)
        self.atlas = SpriteAtlas.load(
            rc.ctx,
            key_count=rc.key_count,
            skin_dir=sk_dir,
            beatmap_dir=bm_dir,
            mania_section=self.mania_section,
            combo_prefix=(self.skin_ini.combo_prefix
                          if self.skin_ini is not None else "score"),
        )
        self._make_quad_geometry()

        # Instanced sprite pipeline. Per-vertex buffer is the 4-corner
        # unit quad (built once); per-instance buffer is filled on the
        # fly each frame with up to _INSTANCE_CAP rectangles, then one
        # `glDrawArraysInstanced` fires per `_flush_sprite_batch`. This
        # replaces ~30 per-sprite draw calls per frame with 1-3 total.
        ctx = rc.ctx
        unit_corners = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype="f4",
        )
        self._unit_quad_vbo = ctx.buffer(unit_corners.tobytes())
        self._instance_vbo = ctx.buffer(
            reserve=self._INSTANCE_CAP * self._FLOATS_PER_INSTANCE * 4,
            dynamic=True,
        )
        self._instance_vao = ctx.vertex_array(
            self.programs["sprite_instanced"],
            [
                (self._unit_quad_vbo, "2f", "in_corner"),
                # `/i` = instance divisor 1 (advance one entry per instance).
                (self._instance_vbo, "4f 1f 4f /i",
                 "in_rect", "in_atlas", "in_color"),
            ],
        )
        # Pre-allocated CPU-side instance buffer. Slice-assigning a tuple
        # of 9 floats into a row of a typed numpy array is one C call
        # (much faster than building a Python list and `np.asarray`'ing
        # it, which is the regression that killed the previous attempt).
        self._instance_arr = np.zeros(
            (self._INSTANCE_CAP, self._FLOATS_PER_INSTANCE), dtype="f4",
        )
        self._instance_count: int = 0
        self._hd_active: bool = False
        self._fi_active: bool = False
        # osu!lazer Hidden/FadeIn cover geometry (px), recomputed per frame
        # from the combo when HD/FI is active. See _flush_sprite_batch.
        self._cov_fill_px: float = 0.0
        self._cov_grad_px: float = 0.0
        self._cov_recep: float = 0.0
        # Cached single-layer texture arrays for full-res direct-draw sprites
        # (scorebar / stage panels) — built once, reused every frame.
        self._direct_arr_cache: dict = {}
        # Compute playfield geometry once. Honoured by all per-frame
        # draws via self.pf_x / self.pf_w / self.col_x / self.col_w.
        self._compute_playfield_geometry()

    def _make_quad_geometry(self) -> None:
        ctx = self.rc.ctx
        # Two-triangle unit quad with UVs.
        self._unit_quad = ctx.buffer(
            np.array([
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            ], dtype="f4").tobytes()
        )

    def _is_argon_default(self) -> bool:
        """True when NO user skin supplies mania content — the Argon default
        applies (matches argon.is_argon_default). Used to
        switch the playfield to lazer's Argon stage-unit geometry."""
        if self.mania_section is not None:
            return False
        a = self.atlas
        K = self.rc.key_count
        for kind in ("note_tap", "note_hold_head", "note_hold_body", "receptor_off"):
            for c in range(K):
                if a.column_source(kind, c) in ("user", "beatmap"):
                    return False
        for g in ("stage_left", "stage_right", "playfield_frame",
                  "stage_light", "hit_light"):
            if a.global_source(g) in ("user", "beatmap"):
                return False
        return True

    def _compute_playfield_geometry(self) -> None:
        """Resolve playfield X-position + per-column widths from either
        the skin's `[Mania]` block or the renderer's defaults.

        Sets these instance attrs:
          self.pf_x     — left edge of column 0 (render pixels)
          self.pf_w     — total playfield width (render pixels)
          self.col_x    — tuple of left edges per column (length K)
          self.col_w    — tuple of widths per column (length K)
          self.col_w_uniform — int width used by HUD code that doesn't
                                care about per-column variation

        osu! skin.ini geometry is in 480-ref pixels along an X-axis that
        runs 0..640 (the 4:3 region of the screen). We convert via
        `px_per_ref = render_height / 480` for the X scale too — peppy
        scales X by the same factor as Y when fitting the 4:3 region
        inside a wider frame — and centre horizontally.
        """
        rc = self.rc
        K = rc.key_count
        section = self.mania_section

        # osu!-stable scale: 1 ref-pixel ≡ render_height / 480.
        px_per_ref = rc.height / 480.0
        # Centre the 4:3 region horizontally within the 16:9 frame.
        region_w_px = rc.height * 4.0 / 3.0
        region_x0_px = (rc.width - region_w_px) / 2.0

        # Resolve per-column widths in stable's 480-ref pixels. Lazer
        # treats `ColumnWidth` as authoritative when the skin's [Mania]
        # block matches our key count, else fabricates a fresh default
        # (NOT closest-matching another Keys: N section).
        # lazer fills ColumnWidth[K] with the default, then overrides per
        # index from the skin.ini CSV — so a single value (the common
        # "all columns this wide" shorthand, e.g. Vio's `ColumnWidth: 30`)
        # applies to every column instead of being ignored.
        def _fill(vals, K, default):
            vals = list(vals)
            if not vals:
                return [default] * K
            if len(vals) == 1:
                return vals * K
            return (vals + [default] * K)[:K]

        col_w_ref = _fill(
            section.column_width if section is not None else (),
            K, LAZER_DEFAULT_COLUMN_SIZE_REF,
        )
        col_sp_ref = _fill(
            section.column_spacing if section is not None else (),
            K - 1, LAZER_DEFAULT_COLUMN_SPACING_REF,
        )

        # Convert stable-ref values → render pixels.
        col_w_list = [
            max(1, int(round(w * px_per_ref))) for w in col_w_ref
        ]
        col_spacing_list = [
            int(round(s * px_per_ref)) for s in col_sp_ref
        ]

        # Argon default (no user skin): ManiaArgonSkinTransformer overrides
        # ColumnWidth to `60 * (special ? 2 : 1)` STAGE units (NOT Column's
        # 80/70 defaults), scaled by height/768. Verified against the
        # reference: 60 units → 88px @1125h, and hit-target(110)/col(60) =
        # 1.83 column-widths, exactly as measured. Legacy skins keep their
        # own ColumnWidth handling above.
        is_argon = self._is_argon_default()
        if is_argon:
            col_w_list = [
                max(1, int(round(
                    (120.0 if column_variant(c, K) == "center" else 60.0)
                    * rc.height / 768.0)))
                for c in range(K)
            ]
            # lazer Argon spaces columns (LeftColumnSpacing+RightColumnSpacing
            # = 2 stage units; measured ~4px @1080p with the rounded column bg).
            # 3 units * height/768 matches that visible gap. (Was 0 = flush bug.)
            col_spacing_list = [max(1, int(round(3.0 * rc.height / 768.0)))] * (K - 1)

        # Total playfield width.
        pf_w_unaligned = sum(col_w_list) + sum(col_spacing_list)

        # Auto-centre the playfield within the 4:3 region rather than
        # honouring `ColumnStart` literally. Reasoning: osu!stable
        # renders at 4:3 (640×480); skin authors target a specific
        # in-game pixel position. Our renders are 16:9 and letterbox
        # the 4:3 region inside, so a skin authored with `ColumnStart:
        # 267` (FNF) would visually offset hard to the right within
        # the wider frame. Honouring `ColumnWidth` while auto-centring
        # preserves the skin's intended lane proportions without
        # per-skin hardcoding.
        pf_x = int(round(region_x0_px + (region_w_px - pf_w_unaligned) / 2.0))

        col_x: list[int] = []
        x = pf_x
        for c in range(K):
            col_x.append(x)
            x += col_w_list[c]
            if c < K - 1:
                x += col_spacing_list[c]

        self.col_x = tuple(col_x)
        self.col_w = tuple(col_w_list)
        self.pf_x = pf_x
        self.pf_w = x - pf_x
        # The "uniform" value HUD callers (hit error bar, key overlay)
        # use as a single column-width reference, sized at the average
        # so variable-pitch layouts still look proportional.
        self.col_w_uniform = max(1, self.pf_w // K)

        # Y positions. osu! reference is Y-down 0..480; our GL coords
        # are Y-up 0..rc.height. Conversion: gl_y = h - osu_y * h/480.
        def osu_y_to_gl(y_ref: float) -> int:
            return int(rc.height - y_ref * rc.height / 480.0)

        # Receptor centre (= judgement line). Skin's HitPosition wins
        # when set; otherwise keep the renderer's bottom-offset default
        # so non-skinned renders look unchanged.
        if section is not None and section.hit_position is not None:
            self.receptor_centre_y_gl = osu_y_to_gl(section.hit_position)
        elif is_argon:
            # lazer Stage.HIT_TARGET_POSITION = 110 stage units from the
            # bottom (×height/768). This gives the tall key area below the
            # hit line that lazer's Argon shows.
            self.receptor_centre_y_gl = int(round(110.0 * rc.height / 768.0))
        else:
            avg_rec_h = int(self.col_w_uniform * RECEPTOR_HEIGHT_REL_COL)
            self.receptor_centre_y_gl = (
                int(rc.height * RECEPTOR_BOTTOM_OFFSET_FRAC) + avg_rec_h // 2
            )

        # UpsideDown: when the skin's [Mania] block sets `UpsideDown: 1`
        # (FNF, several rhythm-game-themed skins) the playfield flips
        # vertically — receptors render at the TOP of the screen and notes
        # scroll UPWARD. Honour it by mirroring the receptor across the
        # screen midline; the per-frame `to_screen_y` reverses the note
        # scroll direction so notes spawn at the bottom and travel up.
        # Without this, FNF's HitPosition=464 (designed for the flip)
        # lands the receptors AT the very bottom of the 720-px frame
        # where they get clipped — the "clipping into the bottom" bug.
        self.upside_down = bool(
            section is not None and section.upside_down,
        )
        # Clamp the receptor centre so the full sprite (≈ col_w_uniform
        # tall, centred on the line) stays on-screen in either orientation.
        # Many skins author HitPosition right at the edge (Night05=447,
        # FNF=464, minimaly=470) — at 480-ref → 720-render that translates
        # to ~10-25 px from the screen edge in our rendering, so the lower
        # half of the receptor sprite ends up off-frame without a clamp.
        # FNF-style key sprites include arrow tips, outline strokes, and
        # animation halos that extend to the canvas edge, so we leave
        # ~half-a-col_w of breathing room rather than the bare sprite_half.
        clamp_margin = self.col_w_uniform + 4
        if self.upside_down:
            mirrored = rc.height - self.receptor_centre_y_gl
            max_gl = rc.height - clamp_margin
            self.receptor_centre_y_gl = min(mirrored, max_gl)
        else:
            # Normal mode: receptor near screen bottom (low gl_y). Sprite
            # extends symmetrically about the centre; the bottom half
            # would clip below gl_y=0 if HitPosition is near the bottom.
            min_gl = clamp_margin
            self.receptor_centre_y_gl = max(self.receptor_centre_y_gl, min_gl)

        # Combo counter / judgement popup Y. Defaults match pre-Phase-B
        # placement (combo baseline ≈ 58% of frame height from bottom).
        if section is not None and section.combo_position is not None:
            self.combo_baseline_y_gl = osu_y_to_gl(section.combo_position)
        else:
            self.combo_baseline_y_gl = int(rc.height * 0.58)

        if section is not None and section.score_position is not None:
            self.score_popup_y_gl = osu_y_to_gl(section.score_position)
        else:
            # Default lives just below combo baseline (renderer-historical).
            self.score_popup_y_gl = self.combo_baseline_y_gl - 8

    def set_background(self, path: Path | None) -> None:
        if path is None or not path.exists():
            self._bg_tex = None
            return
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as exc:  # noqa: BLE001 -- corrupt/unsupported bg must not kill the render
            log.warning("background image load failed (%s): %s", path, exc)
            self._bg_tex = None
            return
        # Resize to fit the canvas, preserving aspect (cover).
        canvas_aspect = self.rc.width / self.rc.height
        img_aspect = img.width / img.height
        if img_aspect > canvas_aspect:
            new_h = self.rc.height
            new_w = int(img_aspect * new_h)
        else:
            new_w = self.rc.width
            new_h = int(new_w / img_aspect)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Crop centered.
        left = (new_w - self.rc.width) // 2
        top = (new_h - self.rc.height) // 2
        img = img.crop((left, top, left + self.rc.width, top + self.rc.height))
        # bg_blur (0-10) → PIL GaussianBlur radius. Pre-blur once at load
        # time rather than every frame — the bg never changes mid-render.
        # Radius 2 per blur step gives a nice soft look at 10 without
        # making the image unrecognisable.
        blur = max(0, min(10, self.options.bg_blur))
        if blur > 0:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(radius=blur * 2))
        self._bg_tex = self.rc.ctx.texture((self.rc.width, self.rc.height), 4, img.tobytes())

    def _draw_background(self, scene: SceneState | None = None) -> None:
        if not getattr(self, "_bg_tex", None):
            return
        # Pick the active dim level. With a DimEnvelope (built in __init__
        # from the orchestrator's note/break data) the dim GLIDES: intro HELD
        # until the first note's approach then a 900 ms smoothstep to game
        # dim, brighten at each break's start, re-dim at the resume anchor —
        # std/catch's exact envelope, replacing the old 600 ms linear ramp
        # (which also never brightened for breaks). Constructions without the
        # envelope (unit tests, tools) keep the legacy ramp below.
        dim_game = (
            self.options.bg_dim_game if self.options.bg_dim_game is not None
            else self.options.background_dim
        )
        dim_intro = (
            self.options.bg_dim_intro if self.options.bg_dim_intro is not None
            else dim_game
        )
        if self._dim_env is not None and scene is not None:
            dim = self._dim_env.level(float(scene.t_ms))
        elif scene is None or self.first_note_ms <= 0:
            dim = dim_game
        elif scene.t_ms < self.first_note_ms - 600:
            dim = dim_intro
        elif scene.t_ms < self.first_note_ms:
            # Ramp from intro to game dim across the last 600ms before first note.
            ramp = (scene.t_ms - (self.first_note_ms - 600)) / 600.0
            dim = dim_intro + (dim_game - dim_intro) * max(0.0, min(1.0, ramp))
        else:
            dim = dim_game
        alpha = max(0.0, min(1.0, 1.0 - dim))

        # Blur: each pass shrinks-and-re-stretches via texture filtering,
        # which is a cheap softening hack — N passes ≈ N-pixel blur radius.
        # 0 = native; 10 = heavily abstract atmosphere only.
        # (For now we skip the actual blur pipeline — adding a proper
        # separable gaussian needs a new program; this is a placeholder
        # that just preserves dim behaviour. Real blur lands when the FBO
        # pingpong is wired.)
        self._draw_external_texture(
            self._bg_tex, x=0, y=0,
            w=self.rc.width, h=self.rc.height, alpha=alpha,
        )

    def _draw_watermark(self, text: str) -> None:
        """Bottom-right white text, ~22px, low-opacity. Cached so we don't
        re-rasterise the string every frame."""
        rc = self.rc
        tex, w, h = self._cached_text(text[:64], 22, (255, 255, 255, 200))
        self._draw_external_texture(
            tex,
            x=rc.width - w - 18,
            y=18,
            w=w, h=h, alpha=0.85,
        )

    def draw_logo_splash(self, t_ms: int) -> None:
        """R3D 'R' tile intro splash (show_logo) — ported from the std/catch
        renderers so the splash is identical across modes: a red additive
        glow + the shared assets/logo.png tile, centred at (w/2, 0.44h from
        the top), 220px in 1080-space with the settle scale, fading out
        exactly as the first note spawns (first_note_ms - approach window).
        No-op (and zero side effects) unless options.show_logo is on."""
        if not getattr(self.options, "show_logo", False):
            return
        from osu_mania_renderer_v2.render.logo import (
            LOGO_UI_SIZE,
            bake_logo_tile,
            logo_alpha,
            logo_glow_rgba,
            logo_scale,
        )

        # The splash window opens at the render's first frame (t=0) and
        # closes at the first note's spawn. Approach mirrors
        # render.build_render_plan's scroll-speed formula (lazy import —
        # render.py imports this module at load time).
        from osu_mania_renderer_v2.render.render import (
            APPROACH_MS,
            SCROLL_SPEED_BASELINE,
        )
        ss = getattr(self.options, "scroll_speed", None)
        approach = int(APPROACH_MS * SCROLL_SPEED_BASELINE / ss) if ss else APPROACH_MS
        gameplay_in = float(self.first_note_ms - approach)
        la = logo_alpha(float(t_ms), 0.0, gameplay_in)
        if la is None:
            return
        if self._logo_tex is None:
            tile = bake_logo_tile()
            self._logo_tex = self.rc.ctx.texture(
                (tile.shape[1], tile.shape[0]), 4, tile.tobytes())
            glow = logo_glow_rgba()
            self._logo_glow_tex = self.rc.ctx.texture(
                (glow.shape[1], glow.shape[0]), 4, glow.tobytes())
        rc = self.rc
        k_ui = rc.height / 1080.0
        d = LOGO_UI_SIZE * k_ui * logo_scale(float(t_ms), 0.0)
        cx = rc.width / 2.0
        # std/catch centre the splash 0.44 of the screen from the TOP;
        # this renderer's draw rects are bottom-left origin.
        cy = rc.height * (1.0 - 0.44)
        # Flush queued sprites (e.g. the fade-to-black wash) under the
        # CURRENT blend mode before switching to additive for the glow.
        self._flush_sprite_batch()
        ctx = rc.ctx
        g = d * 1.9
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        self._draw_external_texture(
            self._logo_glow_tex,
            x=int(cx - g / 2), y=int(cy - g / 2), w=int(g), h=int(g),
            alpha=0.45 * la, tint=(0.95, 0.28, 0.30),
        )
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._draw_external_texture(
            self._logo_tex,
            x=int(cx - d / 2), y=int(cy - d / 2), w=int(d), h=int(d),
            alpha=la,
        )

    def set_results_data(self, data, board=None) -> None:
        """Per-render data for the LAZER RESULTS SCREEN (hud/lazer_results.py
        — the osu!(lazer) ranking screen, parity with std/catch/taiko).
        Called by compositor.py with a ManiaResultsData
        (results_data_from_plan). None → the legacy argon
        card keeps drawing (fail-soft). `board` is the pre-baked flank
        leaderboard (hud/lb_cards.py BakedBoard | None — built + baked ONCE
        up front, exactly like catch render/render.py:580-588); None → the
        plain results screen, unchanged."""
        self._results_data = data
        self._results_board = board
        self._lazer_results = None          # (re)built on the first results frame
        self._lazer_results_token = None    # last-uploaded pose token

    def _try_draw_lazer_results(self, scene: SceneState) -> bool:
        """Draw the ported lazer ranking screen (hud/lazer_results.py). The
        screen composites itself as a full-frame RGBA layer on the CPU
        (exactly like catch's hud.draw_results path — see
        osu_catch_renderer/hud/hud.py:1828-1865 for the wiring this mirrors);
        this method uploads the layer and blends it over the GL frame as one
        fullscreen quad. Returns False (→ the caller falls back to the
        legacy argon card, LOUDLY) when no data was plumbed or the screen
        failed once this render."""
        data = getattr(self, "_results_data", None)
        if data is None:
            return False
        scr = getattr(self, "_lazer_results", None)
        if scr is False:                    # earlier failure → legacy card
            return False
        try:
            if scr is None:
                from osu_mania_renderer_v2.hud.lazer_results import (
                    ManiaLazerResults,
                )
                scr = ManiaLazerResults((self.rc.width, self.rc.height), data,
                                        board=getattr(self, "_results_board",
                                                      None))
                self._lazer_results = scr
            # ms since results_start — drives the ported centred-card timeline
            # (catch render loop passes the same age; scene.t_ms is already
            # the video clock the results_opacity ramp runs on).
            age_ms = max(0.0, float(scene.t_ms - data.results_start_ms))
            arr, token = scr.render_overlay(scene.results_opacity, age_ms)
            self._blit_results_overlay(arr, token)
            return True
        except Exception as e:  # noqa: BLE001 — results must never kill a render
            import sys
            import traceback
            print("[mania-renderer] !!! LAZER RESULTS SCREEN FAILED — "
                  f"falling back to the legacy results card: {e}",
                  file=sys.stderr)
            traceback.print_exc()
            self._lazer_results = False
            return False

    def _blit_results_overlay(self, arr, token) -> None:
        """Upload the CPU-composited results layer (full-frame RGBA, straight
        alpha, PIL row order) and draw it as one fullscreen quad through the
        sprite program. The quad flips v exactly like _draw_external_texture
        so the layer's top row lands at the top of the GL frame; the frame's
        blend state (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) makes this the same
        OVER composite catch performs on its CPU rgb array. The texture is
        re-written only when the pose token changes — the settled tail of
        the outro re-uploads nothing."""
        self._flush_sprite_batch()
        ctx = self.rc.ctx
        w, h = self.rc.width, self.rc.height
        tex = getattr(self, "_lazer_results_tex", None)
        if tex is None:
            tex = ctx.texture_array((w, h, 1), 4)
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._lazer_results_tex = tex
            self._lazer_results_token = None
        if token != self._lazer_results_token:
            tex.write(np.ascontiguousarray(arr).tobytes())
            self._lazer_results_token = token
        prog = self.programs["sprite"]
        tex.use(0)
        self._set_sprite_prog_uniforms(prog, h)
        verts = _EXT_QUAD_PACK(
            -1.0, -1.0, 0, 1, 0, 1.0, 1.0, 1.0, 1.0,
            1.0, -1.0, 1, 1, 0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1, 0, 0, 1.0, 1.0, 1.0, 1.0,
            -1.0, -1.0, 0, 1, 0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1, 0, 0, 1.0, 1.0, 1.0, 1.0,
            -1.0, 1.0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0,
        )
        vbo, vao = self._ext_quad_buffers()
        vbo.write(verts)
        vao.render(moderngl.TRIANGLES)

    _HUD_CACHE_MAX = 256
    _FONT_REFERENCE_HEIGHT = 1080

    def _cached_text(
        self, line: str, size: int,
        color: tuple[int, int, int, int] = (255, 255, 255, 255),
    ) -> tuple:
        """Bounded LRU cache of rasterised PIL text → GL texture. The
        requested size is in 1080p reference units and is automatically
        scaled down for smaller render targets so the layout looks the same
        at 720p as it does at 1080p."""
        if not hasattr(self, "_hud_cache"):
            self._hud_cache: dict[str, tuple] = {}
        size = max(8, int(size * self.rc.height / self._FONT_REFERENCE_HEIGHT))
        key = f"{size}:{color}:{line}"
        entry = self._hud_cache.get(key)
        if entry is None:
            while len(self._hud_cache) >= self._HUD_CACHE_MAX:
                oldest_key = next(iter(self._hud_cache))
                old_tex, _, _ = self._hud_cache.pop(oldest_key)
                try:
                    # Release the paired single-layer wrap array (built by
                    # _draw_external_texture, cached on `extra`) with its
                    # source texture so evictions don't leak GPU memory.
                    if old_tex.extra is not None:
                        old_tex.extra.release()
                        old_tex.extra = None
                    old_tex.release()
                except Exception:  # noqa: BLE001
                    pass
            entry = text_to_texture(self.rc.ctx, line, size=size, color=color)
            self._hud_cache[key] = entry
        else:
            self._hud_cache.pop(key)
            self._hud_cache[key] = entry
        return entry

    def _draw_combo_and_judgment(self, scene: SceneState) -> None:
        """Centred combo (top) + judgment popup over the playfield.

        Combo is the running hit-without-miss count. Judgment is the value of
        the most-recent hit (320 / 300 / 200 / 100 / 50 / MISS) and uses
        Night05's bundled sprite. Combo sits ABOVE the judgment so the two
        don't overlap, matching the OG web replay viewer layout.

        This is the PIL fallback used when no skin score font is available.
        """
        pf_x = self.pf_x
        pf_w = self.pf_w
        center_x = pf_x + pf_w // 2
        # Combo sits above judgment, tightly stacked, in the upper-middle of
        # the playfield — same arrangement the OG web replay viewer uses.
        # Combo baseline = bottom of the combo text, in GL (Y-up) coords; the
        # judgment sprite is drawn directly below it. Honours ComboPosition
        # from the skin's [Mania] block when set.
        combo_baseline_y = self.combo_baseline_y_gl
        # Judgment sprites (Night05's mania-hit300g etc.) are 384×384 square
        # with the value text centred in transparent padding. Drawing them at
        # square aspect (not stretched to a wide rect) keeps text proportions
        # right; pf_w * 0.55 makes the visible text comparable in size to the
        # combo number above it without dwarfing the playfield.
        jud_w = int(pf_w * 0.55)
        jud_h = jud_w

        # Judgment popup — uses the latest judgment event (most recent hit).
        if scene.active_judgments:
            j = scene.active_judgments[-1]
            alpha = max(0.0, 1.0 - j.age_ms / 500.0)
            sprite_name = f"judgment_{j.judgment}"
            # Animated hit-burst: skin can ship `mania-hit300-0.png`,
            # `mania-hit300-1.png`, … per the wiki. Spec says 60 fps,
            # plays once, holds the last frame during fade-out.
            base_idx = self.atlas.index_of(sprite_name)
            frame_count = self.atlas.frame_count(sprite_name)
            if frame_count > 1:
                frame_idx = min(int(j.age_ms * 60.0 / 1000.0), frame_count - 1)
                atlas_idx = base_idx + frame_idx
            else:
                atlas_idx = base_idx
            # Judgment top-edge sits just under the combo baseline, with a
            # small visual gap. (`_draw_sprite_idx` takes a bottom-left
            # origin in GL coords; the sprite extends UP from there.)
            self._draw_sprite_idx(
                atlas_idx,
                center_x - jud_w // 2,
                combo_baseline_y - jud_h - 8,
                jud_w, jud_h, (1, 1, 1, alpha),
            )

        # Combo: bold number ABOVE the judgment. "Pop" animation — the
        # digits scale up briefly on each increment, then settle back. The
        # text colour tiers by combo size: white at <100, gold at ≥100,
        # cyan at ≥250, magenta at ≥500 — small visual reward for streaks.
        if scene.combo > 0:
            combo_str = f"{scene.combo}"
            if scene.combo >= 500:
                combo_colour = (255, 150, 230, 255)
            elif scene.combo >= 250:
                combo_colour = (140, 230, 255, 255)
            elif scene.combo >= 100:
                combo_colour = (255, 215, 100, 255)
            else:
                combo_colour = (255, 255, 255, 255)
            ctex, cw, ch = self._cached_text(combo_str, 110, combo_colour)
            pop = 1.0
            if scene.combo_age_ms < 180:
                t = scene.combo_age_ms / 180.0
                # ease-out curve: starts at 1.18 (big), settles to 1.0.
                pop = 1.0 + 0.18 * (1.0 - t) ** 2
            pw = int(cw * pop)
            ph = int(ch * pop)
            self._draw_external_texture(
                ctex, x=center_x - pw // 2,
                y=combo_baseline_y - (ph - ch) // 2,
                w=pw, h=ph, alpha=0.95,
            )

    def _draw_progress_bar(self, scene: SceneState) -> None:
        """Thin song-progress bar at the very top of the screen. Empty
        gutter on the right, white fill on the left grows toward 100%."""
        rc = self.rc
        p = max(0.0, min(1.0, scene.song_progress))
        if p <= 0:
            return
        bar_h = max(2, int(rc.height * 0.005))
        bar_y = rc.height - bar_h
        self._draw_sprite(
            "column_bg", 0, bar_y, rc.width, bar_h,
            (0.15, 0.15, 0.2, 0.5),
        )
        self._draw_sprite(
            "column_bg", 0, bar_y, int(rc.width * p), bar_h,
            (1, 1, 1, 0.85),
        )

    def _set_sprite_prog_uniforms(self, prog, screen_h: int) -> None:
        """The non-instanced "sprite" program is only ever driven by
        `_draw_external_texture` / `_draw_direct`, and both always set the
        exact same constant uniform values. Uniform state persists on the
        program object, so write them once and skip the ~8 redundant GL
        uniform uploads on every subsequent call. Values identical to the
        per-call sets they replace."""
        if getattr(self, "_sprite_prog_uniforms_set", False):
            return
        prog["u_atlas"] = 0
        prog["u_projection"].value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        prog["u_hd"].value = 0.0
        prog["u_fi"].value = 0.0
        prog["u_hd_recep"].value = 0.0
        prog["u_pf_top"].value = float(screen_h)
        prog["u_cov_fill"].value = 0.0
        prog["u_cov_grad"].value = 0.0
        self._sprite_prog_uniforms_set = True

    def _ext_quad_buffers(self):
        """One persistent (VBO, VAO) pair for the ad-hoc 6-vertex quads that
        `_draw_external_texture` / `_draw_direct` emit — the old code
        created and released a fresh VBO+VAO on every call (dozens per
        frame). Same layout, same program, same draw; the VBO contents are
        rewritten before each render call."""
        pair = getattr(self, "_ext_quad_pair", None)
        if pair is None:
            ctx = self.rc.ctx
            vbo = ctx.buffer(reserve=6 * 9 * 4, dynamic=True)
            vao = ctx.simple_vertex_array(
                self.programs["sprite"], vbo,
                "in_pos", "in_uv", "in_atlas_index", "in_color",
            )
            pair = (vbo, vao)
            self._ext_quad_pair = pair
        return pair

    def _draw_external_texture(
        self, tex: moderngl.Texture, x: int, y: int, w: int, h: int, alpha: float,
        tint: tuple = (1.0, 1.0, 1.0), rotation_deg: float = 0.0,
    ) -> None:
        # Flush whatever's queued in the sprite batch FIRST so this draw
        # lands on top of any sprite that was enqueued earlier in this
        # frame. Required for correct alpha blending order — without it
        # the batch would render AFTER the texture, on top of it.
        self._flush_sprite_batch()
        # The sprite shader expects a sampler2DArray; the simplest way to draw
        # an ad-hoc 2D texture (e.g. text or background) with the same shader is
        # to wrap it in a single-layer texture array. The wrap is built ONCE
        # per source texture and cached on the texture's `extra` slot — the
        # old per-call rebuild did a full GPU→CPU `tex.read()` + re-upload on
        # EVERY draw (the background alone re-uploaded the whole frame each
        # frame). All wrapped textures are immutable after creation (text,
        # bg, logo, rings), so the cached wrap stays valid; the two texture
        # release sites (such as _cached_text eviction) release the
        # paired wrap alongside the texture.
        ctx = self.rc.ctx
        prog = self.programs["sprite"]
        screen_w, screen_h = self.rc.width, self.rc.height
        x0 = (x / screen_w) * 2 - 1
        x1 = ((x + w) / screen_w) * 2 - 1
        y0 = (y / screen_h) * 2 - 1
        y1 = ((y + h) / screen_h) * 2 - 1
        # The texture array's storage MUST match the source texture's pixel
        # dimensions (that's what tex.read() returns). The function's `w`/`h`
        # params describe the DRAW RECT on screen — they can differ from the
        # texture size when the caller wants to stretch/scale a glyph for an
        # animation (combo pop, results overlay). Treat them independently.
        tex_2d_to_array = tex.extra
        if tex_2d_to_array is None:
            tex_2d_to_array = ctx.texture_array(
                size=(tex.width, tex.height, 1),
                components=4,
                data=tex.read(),
            )
            tex.extra = tex_2d_to_array
        tex_2d_to_array.use(0)
        self._set_sprite_prog_uniforms(prog, screen_h)

        tr_, tg_, tb_ = tint
        if rotation_deg:
            import math as _m
            cxs, cys = x + w / 2.0, y + h / 2.0
            ct, st = _m.cos(_m.radians(rotation_deg)), _m.sin(_m.radians(rotation_deg))

            def _rot(px, py):
                dx, dy = px - cxs, py - cys
                rx = cxs + dx * ct - dy * st
                ry = cys + dx * st + dy * ct
                return (rx / screen_w) * 2 - 1, (ry / screen_h) * 2 - 1
            bl = _rot(x, y)
            br = _rot(x + w, y)
            tr2 = _rot(x + w, y + h)
            tl = _rot(x, y + h)
        else:
            bl = (x0, y0)
            br = (x1, y0)
            tr2 = (x1, y1)
            tl = (x0, y1)
        # struct.pack instead of np.array-of-nested-lists: same 54 little-
        # endian float32s on the wire, ~4µs less Python per call — and this
        # runs ~30x per frame (HUD text, judgments, combo, backgrounds).
        verts = _EXT_QUAD_PACK(
            bl[0], bl[1], 0, 1, 0, tr_, tg_, tb_, alpha,
            br[0], br[1], 1, 1, 0, tr_, tg_, tb_, alpha,
            tr2[0], tr2[1], 1, 0, 0, tr_, tg_, tb_, alpha,
            bl[0], bl[1], 0, 1, 0, tr_, tg_, tb_, alpha,
            tr2[0], tr2[1], 1, 0, 0, tr_, tg_, tb_, alpha,
            tl[0], tl[1], 0, 0, 0, tr_, tg_, tb_, alpha,
        )
        vbo, vao = self._ext_quad_buffers()
        vbo.write(verts)
        vao.render(moderngl.TRIANGLES)

    def _draw_direct(
        self, name: str, x: int, y: int, w: int, h: int,
        tint: tuple = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        """Draw a wide skin sprite (scorebar / stage panel) at full resolution,
        bypassing the layered 256² atlas (which would crush a 1366-wide bar).
        The source texture is cached as a single-layer array; `tint` (RGBA)
        multiplies the texture so the scorebar fill can be coloured."""
        if w <= 0 or h <= 0:
            return
        arr = self._direct_arr_cache.get(name)
        if arr is None:
            img = self.atlas.direct_image(name)
            if img is None:
                return
            arr = self.rc.ctx.texture_array(
                size=(img.width, img.height, 1), components=4, data=img.tobytes(),
            )
            arr.build_mipmaps()
            arr.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self._direct_arr_cache[name] = arr
        # Land on top of the queued batch (correct alpha order).
        self._flush_sprite_batch()
        prog = self.programs["sprite"]
        sw, sh = self.rc.width, self.rc.height
        x0, x1 = (x / sw) * 2 - 1, ((x + w) / sw) * 2 - 1
        y0, y1 = (y / sh) * 2 - 1, ((y + h) / sh) * 2 - 1
        if len(tint) == 3:
            r, g, b, a = tint[0], tint[1], tint[2], 1.0
        else:
            r, g, b, a = tint
        arr.use(0)
        self._set_sprite_prog_uniforms(prog, sh)
        verts = _EXT_QUAD_PACK(
            x0, y0, 0, 1, 0, r, g, b, a,
            x1, y0, 1, 1, 0, r, g, b, a,
            x1, y1, 1, 0, 0, r, g, b, a,
            x0, y0, 0, 1, 0, r, g, b, a,
            x1, y1, 1, 0, 0, r, g, b, a,
            x0, y1, 0, 0, 0, r, g, b, a,
        )
        vbo, vao = self._ext_quad_buffers()
        vbo.write(verts)
        vao.render(moderngl.TRIANGLES)

    _GRADE_COLOURS: dict[str, tuple[int, int, int]] = {
        "SS": (240, 220, 120),   # gold
        "S":  (240, 220, 120),   # gold
        "A":  (110, 220, 130),   # green
        "B":  (110, 180, 220),   # blue
        "C":  (200, 130, 220),   # purple
        "D":  (220, 110, 110),   # red
    }

    def _results_avatar_texture(self) -> moderngl.Texture | None:
        """Load the featured player's osu! avatar (options.featured_avatar_png)
        once, as a rounded-square GL texture, and cache it for the whole
        results screen. Returns None when no path is set, the file is missing,
        or decoding fails — the caller then draws the grey placeholder chip.
        Never raises (a bad avatar must not break the render)."""
        if getattr(self, "_results_avatar_tried", False):
            return getattr(self, "_results_avatar_tex", None)
        self._results_avatar_tried = True
        self._results_avatar_tex = None
        path = getattr(self.options, "featured_avatar_png", None)
        if not path:
            return None
        try:
            p = Path(path)
            if not p.is_file():
                return None
            from PIL import ImageDraw
            img = Image.open(p).convert("RGBA")
            w, h = img.size
            s = min(w, h)
            img = img.crop(
                ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
            ).resize((256, 256), Image.LANCZOS)
            mask = Image.new("L", (256, 256), 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                (0, 0, 255, 255), radius=46, fill=255)
            img.putalpha(mask)
            self._results_avatar_tex = self.rc.ctx.texture(
                (256, 256), 4, img.tobytes())
        except Exception:  # noqa: BLE001 — avatar is cosmetic; never fatal
            self._results_avatar_tex = None
        return self._results_avatar_tex

    def _draw_results_overlay(self, scene: SceneState, ctx) -> None:
        """Post-game results card — lazer-Argon faithful. Every NUMBER (score,
        accuracy, max combo, the six judgment counts, unstable rate, per-column
        UR, pp) is composed from the bundled argon-counter glyph font (lazer's
        ArgonCounterTextComponent — live digits over the dim wireframe backing)
        via `_argon_number`. Only *labels* (ACCURACY / MAX COMBO / the judgment
        band names / the signed avg-offset caption, which the argon font has no
        +/- glyph for) stay PIL text.

        Layout is authored in lazer's 1080p design space and scaled by
        `A = height/1080`, mirroring `_draw_argon_hud`. The canonical
        `results_overlay` element supplies the live ``FrameContext`` required
        for the argon glyph path.

        DEFAULT PATH (2026-08): the osu!(lazer) RANKING SCREEN ported from
        the catch renderer (hud/lazer_results.py — parity with std/catch/
        taiko). The argon card below is the FAIL-SOFT FALLBACK only: it draws
        when no results data was plumbed (legacy callers) or the lazer
        screen failed once this render (loud stderr trace either way).
        """
        if self._try_draw_lazer_results(scene):
            return
        rc = self.rc
        a = max(0.0, min(1.0, scene.results_opacity))
        use_argon = ctx.has_argon_font()
        # Lazily import the shared argon-number primitive to avoid a module
        # cycle with the registered HUD elements at import time.
        _argon_number = None
        _argon_overlap = 0
        if use_argon:
            try:
                from osu_mania_renderer_v2.hud.elements import (
                    ARGON_OVERLAP as _ao,
                )
                from osu_mania_renderer_v2.hud.elements import (
                    _argon_number as _an,
                )
                _argon_number = _an
                _argon_overlap = _ao
            except Exception:  # noqa: BLE001
                use_argon = False

        A = rc.height / 1080.0
        cx = rc.width / 2.0

        # Dim the whole scene under the card.
        self._draw_sprite("column_bg", 0, 0, rc.width, rc.height,
                          (0, 0, 0, 0.92 * a))

        def _gl_center(from_top_px: float) -> float:
            """1080-design from-top y → GL (bottom-left origin) centre y."""
            return rc.height - from_top_px * A

        def _num(text: str, gh: float, from_top_px: float,
                 tint=(1.0, 1.0, 1.0), *, x=None, align: str = "center") -> None:
            """Draw a numeric value in the argon-counter font (falls back to a
            PIL glyph when the argon font is somehow unavailable). `gh` is the
            1080-design glyph height; `x`/`from_top_px` place the anchor."""
            xx = cx if x is None else x
            cyg = _gl_center(from_top_px)
            if use_argon:
                _argon_number(ctx, text, x=xx, center_y=cyg, glyph_h=gh * A,
                              align=align, alpha=a, tint=tint)
                return
            col = (int(tint[0] * 255), int(tint[1] * 255),
                   int(tint[2] * 255), 255)
            tex, w, h = self._cached_text(text, int(gh), col)
            xpix = xx - w if align == "right" else (
                xx - w / 2 if align == "center" else xx)
            self._draw_external_texture(tex, x=int(xpix), y=int(cyg - h / 2),
                                        w=w, h=h, alpha=a)

        def _label(text: str, size: int, col, from_top_px: float,
                   *, x=None, align: str = "center") -> None:
            """Draw a PIL label (size is 1080-reference, auto-scaled)."""
            xx = cx if x is None else x
            cyg = _gl_center(from_top_px)
            tex, w, h = self._cached_text(text, size, (col[0], col[1], col[2], 255))
            xpix = xx - w if align == "right" else (
                xx - w / 2 if align == "center" else xx)
            self._draw_external_texture(tex, x=int(xpix), y=int(cyg - h / 2),
                                        w=w, h=h, alpha=a)

        # ── Header: featured player avatar (rounded square) + name ──────────
        av = 104.0
        av_top = 42.0
        av_gl_bottom = rc.height - (av_top + av) * A
        fp = 6.0
        # Dark rounded frame behind the avatar (argon_card sprite, tinted).
        self._draw_direct(
            "argon_card", cx - (av / 2 + fp) * A, av_gl_bottom - fp * A,
            (av + 2 * fp) * A, (av + 2 * fp) * A, (0.10, 0.11, 0.14, 0.92 * a))
        avatar_tex = self._results_avatar_texture()
        if avatar_tex is not None:
            self._draw_external_texture(
                avatar_tex, x=int(cx - av * A / 2), y=int(av_gl_bottom),
                w=int(av * A), h=int(av * A), alpha=a)
        else:
            # Grey placeholder chip (matches the leaderboard-card fallback).
            self._draw_direct(
                "argon_card", cx - av * A / 2, av_gl_bottom, av * A, av * A,
                (0.55, 0.57, 0.60, a))
        # Player name — parsed off the banner text ("… [diff]   <player>"),
        # exactly like the gameplay leaderboard card.
        bt = getattr(self, "_banner_text", "") or ""
        name = bt.rsplit("   ", 1)[-1].strip() if "   " in bt else (bt or "Player")
        _label(name[:22], 22, (235, 238, 246), av_top + av + 22)

        # ── Grade letter (huge; argon font has no letters, so PIL) ──────────
        if self.options.show_grade:
            grade = scene.grade or "D"
            g_r, g_g, g_b = self._GRADE_COLOURS.get(grade, (200, 200, 220))
            _label(grade, 148, (g_r, g_g, g_b), 250)

        # ── Score (argon, large) ────────────────────────────────────────────
        _num(str(int(scene.score)), 74, 372, (1.0, 1.0, 1.0))

        # ── Stat row: ACCURACY / MAX COMBO / (PP) / (STAR RATING) ───────────
        has_pp = scene.max_pp > 0
        has_sr = scene.stars > 0
        stat_label_top = 472
        stat_num_top = 514
        # Cells in fixed order; each = (label, value, tint, is_star). The star
        # rating (--sr override, else rosu) reads "X.XX★" like the other engines;
        # the argon-counter font has no ★, so its digits are argon and the ★ is a
        # PIL suffix (mirrors the card's PIL grade letter).
        cells = [
            ("ACCURACY", f"{scene.accuracy:.2f}%", (1.0, 1.0, 1.0), False),
            ("MAX COMBO", f"{scene.max_combo}x", (1.0, 1.0, 1.0), False),
        ]
        if has_pp:
            cells.append(("PP", f"{int(round(scene.pp))}", (1.0, 0.86, 0.55), False))
        if has_sr:
            cells.append(("STAR RATING", f"{scene.stars:.2f}", (1.0, 0.82, 0.30), True))
        n = len(cells)
        # Symmetric spacing: keeps the classic ±200 (2 cells) / ±330 (3) spans
        # and extends to 4 cells (±480) without crowding.
        span = {2: 400.0, 3: 660.0, 4: 960.0}.get(n, 320.0 * max(1, n - 1))
        for i, (lab, val, tint, is_star) in enumerate(cells):
            dx = (-span / 2.0 + span / (n - 1) * i) if n > 1 else 0.0
            xc = cx + dx * A
            _label(lab, 16, (196, 203, 222), stat_label_top, x=xc)
            if not is_star:
                _num(val, 46, stat_num_top, tint, x=xc, align="center")
                continue
            # argon digits + a PIL ★ suffix, drawn as one centred "X.XX★" unit
            gh = 46
            star_col = (int(tint[0] * 255), int(tint[1] * 255),
                        int(tint[2] * 255), 255)
            star_tex, sw, sh = self._cached_text("★", 34, star_col)
            gap = 5.0 * A
            if use_argon:
                num_w = ctx.number_width(val, gh * A, _argon_overlap, "argon")
            else:
                _nt, num_w, _nh = self._cached_text(val, gh, star_col)
            total_w = num_w + gap + sw
            left = xc - total_w / 2.0
            _num(val, gh, stat_num_top, tint, x=left, align="left")
            self._draw_external_texture(
                star_tex, x=int(round(left + num_w + gap)),
                y=int(round(_gl_center(stat_num_top) - sh / 2.0)),
                w=sw, h=sh, alpha=a)

        # ── Judgment counts: 6 colour-coded cells (label + argon number) ────
        jlabels = ("320", "300", "200", "100", "50", "MISS")
        counts = scene.judgment_counts
        jcolours = (
            (150, 215, 255), (255, 230, 120), (140, 220, 140),
            (240, 220, 90), (185, 190, 200), (240, 96, 96),
        )
        j_label_top = 592
        j_num_top = 630
        step = 152.0  # design px between cells
        for i in range(6):
            xc = cx + (i - 2.5) * step * A
            _label(jlabels[i], 16, jcolours[i], j_label_top, x=xc)
            _num(str(counts[i]), 34,
                 j_num_top,
                 (jcolours[i][0] / 255, jcolours[i][1] / 255, jcolours[i][2] / 255),
                 x=xc, align="center")

        # ── Unstable rate (argon) + signed avg-offset caption (PIL) ─────────
        _label("UNSTABLE RATE", 15, (196, 203, 222), 686)
        _num(f"{scene.unstable_rate:.1f}", 40, 720, (0.80, 0.86, 1.0))
        _label(f"avg {scene.avg_hit_offset_ms:+.1f} ms", 16,
               (168, 184, 210), 756)

        # ── Per-column UR (labels + argon numbers) ──────────────────────────
        pcols = scene.per_column_ur
        if pcols:
            n = len(pcols)
            pstep = min(150.0, 620.0 / max(1, n))  # design px between keys
            for i, ur in enumerate(pcols):
                xc = cx + (i - (n - 1) / 2.0) * pstep * A
                _label(f"K{i + 1}", 14, (150, 175, 205), 784, x=xc)
                _num(f"{ur:.0f}", 24, 812, (0.70, 0.78, 0.92), x=xc,
                     align="center")

        # ── UR histogram (reused gameplay primitive) ────────────────────────
        if scene.recent_offsets:
            hist_top = 842.0
            hist_w = 760.0
            hist_h = 94.0
            self._draw_ur_histogram(
                int(cx - hist_w * A / 2),
                int(rc.height - (hist_top + hist_h) * A),
                int(hist_w * A), int(hist_h * A),
                scene.recent_offsets, alpha=a,
            )

    def _draw_ur_histogram(
        self,
        x: int, y: int, w: int, h: int,
        offsets: tuple[float, ...],
        alpha: float,
    ) -> None:
        """Bin the offsets into 25 cells across [-127, +127] ms and draw a
        vertical bar per bin, color-coded by the timing window the bin sits
        in (cyan=320, gold=300, green=200, yellow=100, gray=50). The bar's
        height is proportional to that bin's hit count vs the most populous
        bin, so the centre always touches the top."""
        if not offsets:
            return
        n_bins = 25
        rng = 127.0
        bins = [0] * n_bins
        for off in offsets:
            clipped = max(-rng, min(rng, off))
            idx = int((clipped + rng) / (2 * rng) * (n_bins - 1) + 0.5)
            bins[idx] += 1
        peak = max(bins) or 1
        # Background strip (dim).
        self._draw_sprite("column_bg", x, y, w, h, (0.1, 0.1, 0.15, 0.6 * alpha))
        bar_w = max(2, w // n_bins - 2)
        for i, count in enumerate(bins):
            if count == 0:
                continue
            # Bin centre offset in ms → judgment colour for that timing band.
            # osu!mania judgment colours: 320=light blue, 300=blue,
            # 200=green, 100=yellow, 50=orange. Matches the hit-strip
            # gradient under the receptors.
            centre_ms = -rng + (i + 0.5) * (2 * rng / n_bins)
            absms = abs(centre_ms)
            if absms <= 16.5:
                r, g, b = 150, 215, 255   # 320 light blue
            elif absms <= 40:
                r, g, b =  80, 150, 240   # 300 blue
            elif absms <= 73:
                r, g, b = 100, 220, 130   # 200 green
            elif absms <= 103:
                r, g, b = 240, 220,  90   # 100 yellow
            else:
                r, g, b = 240, 160,  80   # 50 orange
            bar_h = int(h * count / peak)
            bx = x + i * (w // n_bins) + (w // n_bins - bar_w) // 2
            self._draw_sprite(
                "column_bg", bx, y, bar_w, bar_h,
                (r / 255, g / 255, b / 255, alpha),
            )
        # Centre tick (0 ms).
        cx = x + w // 2
        self._draw_sprite("column_bg", cx - 1, y, 2, h,
                          (1, 1, 1, 0.6 * alpha))

    def _draw_sprite(
        self, name: str, x: int, y: int, w: int, h: int, tint: tuple,
    ) -> None:
        """Append one quad to the instance buffer by named atlas slot."""
        self._draw_sprite_idx(self.atlas.index_of(name), x, y, w, h, tint)

    def _draw_sprite_idx(
        self, atlas_idx: int, x: int, y: int, w: int, h: int, tint: tuple,
    ) -> None:
        """Append one quad to the instance buffer by raw atlas layer index.
        Issues NO GL calls; the actual `glDrawArraysInstanced` fires in
        `_flush_sprite_batch` when HD/FI state changes, an external
        texture is about to draw, or the frame ends."""
        if w <= 0 or h <= 0:
            return
        if self._instance_count >= self._INSTANCE_CAP:
            self._flush_sprite_batch()
        sw = self.rc.width
        sh = self.rc.height
        x_clip = (x / sw) * 2 - 1
        y_clip = (y / sh) * 2 - 1
        w_clip = (w / sw) * 2
        h_clip = (h / sh) * 2
        if len(tint) == 3:
            r, g, b = tint
            a = 1.0
        else:
            r, g, b, a = tint
        # One C-level numpy slice assignment — way faster than Python
        # list growth + np.asarray.
        self._instance_arr[self._instance_count] = (
            x_clip, y_clip, w_clip, h_clip, atlas_idx, r, g, b, a,
        )
        self._instance_count += 1

    def apply_note_cover(self, hidden: bool, fade_in: bool, combo: int) -> None:
        """Toggle Hidden/FadeIn and recompute the lazer combo-scaling cover.
        Called by the canonical notes element before its scrolling-note pass.

        osu!lazer ManiaModHidden: coverage scales with combo —
          min(MAX, MIN + combo*rate) / reference_playfield_height
        with MIN=160, MAX=400, rate=0.5, ref=768. FadeIn shares the same
        coverage (it only flips the anchored side). The fade gradient is a
        further 0.25 of the playfield height. The 768 reference maps to the
        full frame (skin Y coords are 0..768)."""
        self._hd_active = hidden
        self._fi_active = fade_in
        if hidden or fade_in:
            cov = min(400.0, 160.0 + max(0, combo) * 0.5) / 768.0
            self._cov_fill_px = cov * self.rc.height
            self._cov_grad_px = 0.25 * self.rc.height
            self._cov_recep = float(self.receptor_centre_y_gl)

    def _flush_sprite_batch(self) -> None:
        """Upload the queued instances to the GPU and fire one
        `glDrawArraysInstanced(TRIANGLE_STRIP, 0, 4, N)` covering them all.
        Must be called before any state change a queued sprite would be
        affected by (HD/FI toggle, `_draw_external_texture`)."""
        n = self._instance_count
        if n == 0:
            return
        slab = self._instance_arr[:n]
        # ndarray implements the buffer protocol — write it directly instead
        # of paying a tobytes() copy per flush. Same bytes hit the GPU.
        self._instance_vbo.write(slab)
        prog = self.programs["sprite_instanced"]
        self.atlas.texture_array.use(0)
        # Uniform state persists on the program between draws, so only
        # re-upload when a value actually changed (HD/FI toggles and the
        # combo-driven cover geometry change a handful of times per frame
        # at most; everything else is constant for the render). Identical
        # GL state to the unconditional per-flush writes this replaces.
        u_state = (
            1.0 if self._hd_active else 0.0,
            1.0 if self._fi_active else 0.0,
            self._cov_recep,
            self._cov_fill_px,
            self._cov_grad_px,
        )
        if getattr(self, "_flush_u_state", None) != u_state:
            if getattr(self, "_flush_u_state", None) is None:
                # First flush: also set the per-render constants once.
                prog["u_atlas"] = 0
                prog["u_pf_top"].value = float(self.rc.height)
            prog["u_hd"].value = u_state[0]
            prog["u_fi"].value = u_state[1]
            prog["u_hd_recep"].value = u_state[2]
            prog["u_cov_fill"].value = u_state[3]
            prog["u_cov_grad"].value = u_state[4]
            self._flush_u_state = u_state
        self._instance_vao.render(
            moderngl.TRIANGLE_STRIP, vertices=4, instances=n,
        )
        self._instance_count = 0
