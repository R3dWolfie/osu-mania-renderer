"""osu!lazer's BreakOverlay, ported onto the mania v2 GL renderer.

This is the mania-engine sibling of the catch reference implementation
(osu-catch/osu_catch_renderer/break_overlay.py, commit d8ccb60) — same
lazer semantics, drawn with THIS engine's own primitives: CPU-baked RGBA
textures + FrameRenderer._draw_external_texture quads (the same path the
HUD text, logo splash and results overlay already use — there is no
CPU-side PIL frame layer in this engine), with the logo splash's
blend-func switch for the additive blurred arrows.

Source of truth — ppy/osu master (read 2026-07-23, keep in sync):
  osu.Game/Screens/Play/BreakOverlay.cs        fade/slide timings, progress
                                               bar semantics, layout Y=±15
  osu.Game/Screens/Play/BreakTracker.cs        which breaks count (HasEffect),
                                               Period = (Start, End - FADE)
  osu.Game/Screens/Play/Break/BreakInfo.cs     "CURRENT PROGRESS" + info lines
  osu.Game/Screens/Play/Break/BreakInfoLine.cs label Yellow/value YellowLight,
                                               2px split margins, acc format
  osu.Game/Screens/Play/Break/RemainingTimeCounter.cs  ceil(ms/1000) seconds
  osu.Game/Screens/Play/Break/BreakArrows.cs   chevron pair geometry/offsets
  osu.Game/Screens/Play/Break/GlowIcon.cs      sharp icon + BlueLighter glow
  osu.Game/Screens/Play/Break/BlurredIcon.cs   blur-only + additive + a=0.7
  osu.Game/Beatmaps/Timing/BreakPeriod.cs      MIN_BREAK_DURATION = 650

The exact lazer timeline (BreakOverlay.updateDisplay, absolute from b.Start;
the tracker's Period trims BREAK_FADE_DURATION=325ms off the END, so with
D = period duration = break duration - 325):
  t'=0..325   fadeContainer.FadeIn(325) [linear]; arrows slide in (OutQuint,
              325ms); counter X -50->0 / info X +50->0 (OutQuint, 325ms);
              progress-bar CONTAINER width 0 -> 0.3 rel (OutQuint, 325ms)
  t'=0..D+325 counter counts (D+325 = full break duration) -> 0, linear;
              display = ceil(count/1000)
  every frame bar width DampContinuously(current, target, halfTime=40ms);
              target = max(0, (Period.End - now - 325) / D)  [reaches 0
              already 325ms BEFORE the fade-out starts]
  t'=D        fadeContainer.FadeOut(325); arrows slide back out (OutQuint,
              325ms); bar container width snaps to 0 — gone at t'=D+325,
              exactly the break's end.

ARROWS — deliberately NOT blinking: lazer master's BreakArrows only slide
in/out and hold (Show/Hide MoveToX, Easing.OutQuint). The pair per side is
the sharp GlowIcon (60px chevron, sigma-10 BlueLighter glow) in front of
the big BlurredIcon (130px, sigma-20, blur-only, additive, alpha 0.7):
the blurs are ADDED first, the sharp glows alpha-composite over them
(lazer child order). Cursor parallax has no analogue in a fixed render
and is dropped — the same call as the catch reference.

TIME AXIS — this engine rescales everything (notes, breaks, frame clock)
to REAL/video time via mods.apply_mods (real = map / audio_rate). lazer
runs the overlay transforms on the rate-adjusted FrameStableClock — the
MAP timeline — so the wiring converts back: t_map = scene.t_ms * rate and
the break periods are un-rescaled the same way at construction. The 650ms
HasEffect gate, the 325ms fades and the countdown seconds are all
map-time quantities, exactly like lazer under DT/HT.

VALUES — live, not snapshotted: BreakOverlay.LoadComplete BindTo()s the
ScoreProcessor's Accuracy/Rank bindables. scene.accuracy is this engine's
live running accuracy (render.build_frame_state acc_so_far, 0..100) and
scene.live_grade is the engine's own mania grade rule applied to the
judgments so far (render._compute_grade_from_replay boundaries — SS only
while nothing below a geki has been judged). Accuracy is formatted with
lazer's FormatAccuracy floor (never rounded up), as in the catch port.
Silver: lazer AdjustRank silvers X/S under ModHidden — for mania that is
HD, FL *and* FadeIn (ManiaModFadeIn : ManiaModWithPlayfieldCover :
ModHidden), so any of the HD/FL/FI pills silvers the Grade line.

TYPOGRAPHY — lazer draws Torus (info) + Venera numerals (counter). This
engine's HUD stack stands in exactly like the rest of its HUD: the
gpu/text.py bold face (the score/accuracy/banner font) for the countdown
digits and the text lines, sized in the renderer's 1080-reference units
(1080/768 x the lazer sizes, so the visible scale matches the lazer
768-space layout).

Z-ORDER: lazer's BreakOverlay is a LATER overlay-component child than
HUDOverlay (Player.createOverlayComponents) — the canonical pipeline calls
this after every HUD element (hud/top chrome/flashlight/UR), and before
the fade-to-black / results / watermark layers, matching lazer's Player
container order. Only single-replay renders construct it (versus/multi
composites are stitched downstream from single renders; no lazer
analogue).

All bakes and GL objects are created lazily on the first visible break
frame (the logo-splash pattern), so a no-break render never touches GL
state here — byte-identical output, proven with framemd5 under
PYTHONHASHSEED=0."""
from __future__ import annotations

import math
from bisect import bisect_right

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# --- lazer constants (files cited in the module docstring) -------------------
MIN_BREAK_DURATION = 650.0        # BreakPeriod.MIN_BREAK_DURATION (HasEffect)
BREAK_FADE_MS = MIN_BREAK_DURATION / 2.0   # BreakOverlay.BREAK_FADE_DURATION
REMAINING_MAX_W = 0.3             # remaining_time_container_max_size
VERTICAL_MARGIN = 15.0            # BreakOverlay.vertical_margin (lazer px)
BAR_H = 8.0                       # remainingTimeBox Height
DAMP_HALF_MS = 40.0               # Interpolation.DampContinuously halfTime
SLIDE_X = 50.0                    # counter/info MoveToX slide distance

GLOW_ICON_SIZE = 60.0             # BreakArrows glow_icon_*
GLOW_ICON_SIGMA = 10.0
GLOW_ICON_FINAL = 0.22            # X offsets, RELATIVE TO WIDTH
GLOW_ICON_OFFSCREEN = 0.6
BLUR_ICON_SIZE = 130.0            # BreakArrows blurred_icon_*
BLUR_ICON_SIGMA = 20.0
BLUR_ICON_FINAL = 0.38
BLUR_ICON_OFFSCREEN = 0.7
BLUR_ICON_ALPHA = 0.7

BLUE_LIGHTER = (0xDD, 0xFF, 0xFF)   # OsuColour.BlueLighter (glow colour)
YELLOW = (0xFF, 0xCC, 0x22)         # OsuColour.Yellow (info labels)
YELLOW_LIGHT = (0xFF, 0xDD, 0x55)   # OsuColour.YellowLight (info values)
SHADOW_GRAY = (51, 51, 51)          # OsuColour.Gray(0.2f)
SHADOW_ALPHA = 0.8                  # .Opacity(0.8f)
SHADOW_RADIUS = 260.0               # EdgeEffect shadow radius (lazer px)
SHADOW_CORE_W, SHADOW_CORE_H = 80.0, 4.0   # the CircularContainer core

COUNTER_SIZE = 33.0               # RemainingTimeCounter OsuFont.Numeric 33
TITLE_SIZE = 15.0                 # "CURRENT PROGRESS" bold 15
LINE_SIZE = 17.0                  # BreakInfoLine label/value size
LINE_MARGIN = 2.0                 # BreakInfoLine margin each side of centre
FLOW_SPACING = 5.0                # BreakInfo FillFlow Spacing(5)

LAZER_UI_HEIGHT = 768.0           # lazer's HUD DrawSizePreserving space
_1080 = 1080.0 / LAZER_UI_HEIGHT  # lazer px -> the engine's 1080-ref units

_CIRCLE_BAKE = 64                 # bake resolution of the bar cap circle

# lazer's ModHidden AdjustRank silvers X/S; mania's HD, FL and FadeIn all
# derive from ModHidden (ManiaModFadeIn : ManiaModWithPlayfieldCover :
# ModHidden). GetLocalisableDescription: XH="Silver SS", SH="Silver S".
_SILVER_ACRONYMS = {"HD", "FL", "FI"}


def _out_quint(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 5


def grade_display(grade: str, mod_acronyms) -> str:
    """The break overlay's Grade line text: the engine's own live mania
    grade (scene.live_grade, render.build_frame_state — single source for
    cutoffs) mapped to lazer's rank display strings, with the HD/FL/FI
    silver adjustment lazer applies."""
    if _SILVER_ACRONYMS & set(mod_acronyms or ()):
        if grade == "SS":
            return "Silver SS"
        if grade == "S":
            return "Silver S"
    return grade


class LazerBreakOverlay:
    """Stateful per-render overlay: bakes the static art lazily (first
    visible break frame), then draws per frame during effective breaks
    (map-time duration >= MIN_BREAK_DURATION). Frames arrive in monotonic
    time order (the compositor runs once per output frame); the
    damped bar width is replayed statefully like lazer's always-running
    Update()."""

    def __init__(self, fr, breaks, rate: float = 1.0):
        """fr: the FrameRenderer. breaks: REAL/video-time (start_ms,
        end_ms) periods, exactly what render.py already hands the dim
        envelope (plan.modded.breaks). rate: plan.audio_rate — converts
        the periods (and each frame's clock) back to the map-time axis
        the lazer transforms run on."""
        self.fr = fr
        self.rate = float(rate or 1.0)
        if self.rate <= 0:
            self.rate = 1.0
        self.w = int(fr.rc.width)
        self.h = int(fr.rc.height)
        self.lk = self.h / LAZER_UI_HEIGHT
        r = self.rate
        # BreakTracker.Breaks: only HasEffect breaks (>= 650 MAP ms), Period
        # end trimmed by BREAK_FADE_DURATION. Stored on the MAP axis as
        # (start, D) with D = period duration; on screen over
        # [start, start + D + 325].
        self.periods = sorted(
            (float(s) * r, float(e - s) * r - BREAK_FADE_MS)
            for s, e in (breaks or ())
            if (e - s) * r >= MIN_BREAK_DURATION)
        self._starts = [p[0] for p in self.periods]
        # DampContinuously state (remainingTimeBox.Width, RELATIVE 0..1)
        self._bar_w = 0.0
        self._last_t: float | None = None
        self._tex: dict[str, tuple] = {}   # name -> (texture, w, h)

    # --- bakes (lazy — a no-break render never reaches these) ----------------

    def _bake(self, name: str, rgba: np.ndarray) -> None:
        tex = self.fr.rc.ctx.texture(
            (rgba.shape[1], rgba.shape[0]), 4,
            np.ascontiguousarray(rgba).tobytes())
        self._tex[name] = (tex, rgba.shape[1], rgba.shape[0])

    def _ensure_baked(self) -> None:
        if self._tex:
            return
        self._bake("shadow", self._bake_shadow())
        glow = self._bake_glow_icon()
        self._bake("glow_r", glow)
        self._bake("glow_l", glow[:, ::-1])
        blur = self._bake_blurred_icon()
        self._bake("blur_r", blur)
        self._bake("blur_l", blur[:, ::-1])
        # remainingTimeBox is a fully-rounded white Circle: full circle +
        # half caps (this engine's external-texture quads have no uv
        # sub-rect, so the caps are their own bakes)
        circle = self._bake_circle()
        self._bake("circle", circle)
        self._bake("cap_l", circle[:, :_CIRCLE_BAKE // 2])
        self._bake("cap_r", circle[:, _CIRCLE_BAKE // 2:])
        self._bake("white", np.full((4, 4, 4), 255, np.uint8))

    def _bake_shadow(self) -> np.ndarray:
        """The fadeContainer's first child: an invisible 80x4 pill whose
        EdgeEffect SHADOW (radius 260, gray(0.2) @ 0.8) is the big soft dark
        blob behind the centre block. Approximated as a quadratic falloff
        over the radius from the pill edge (o!f edge-effect profile) —
        identical math to the catch reference."""
        lk = self.lk
        R = SHADOW_RADIUS * lk
        cw, ch = SHADOW_CORE_W * lk, SHADOW_CORE_H * lk
        W = int(math.ceil(cw + 2 * R))
        H = int(math.ceil(ch + 2 * R))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = ch / 2.0                      # pill corner radius
        qx = np.abs(xx - W / 2.0) - (cw / 2.0 - r)
        qy = np.abs(yy - H / 2.0) - (ch / 2.0 - r)
        d = (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
             + np.minimum(np.maximum(qx, qy), 0.0) - r)
        fall = np.clip(1.0 - d / R, 0.0, 1.0) ** 2
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = SHADOW_GRAY
        rgba[..., 3] = np.round(fall * SHADOW_ALPHA * 255.0).astype(np.uint8)
        return rgba

    def _chevron_mask(self, size_px: int) -> Image.Image:
        """FontAwesome Solid.ChevronRight silhouette: a bold '>' polyline
        (glyph aspect ~0.63 in a square SpriteIcon cell), round caps/joint."""
        s = size_px
        m = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(m)
        w = max(2, int(round(s * 0.17)))
        pts = [(0.36 * s, 0.14 * s), (0.67 * s, 0.50 * s),
               (0.36 * s, 0.86 * s)]
        d.line(pts, fill=255, width=w, joint="curve")
        for px, py in (pts[0], pts[2]):
            d.ellipse([px - w / 2, py - w / 2, px + w / 2, py + w / 2],
                      fill=255)
        return m

    def _bake_glow_icon(self) -> np.ndarray:
        """GlowIcon: sharp white chevron over its BlueLighter gaussian glow
        (GlowingDrawable: blurred silhouette tinted GlowColour, original on
        top). RGBA uint8, right-pointing; the left pair mirrors the bake."""
        lk = self.lk
        s = max(4, int(round(GLOW_ICON_SIZE * lk)))
        sigma = GLOW_ICON_SIGMA * lk
        pad = int(math.ceil(3 * sigma)) + 1
        cv = Image.new("L", (s + 2 * pad, s + 2 * pad), 0)
        cv.paste(self._chevron_mask(s), (pad, pad))
        glow_a = cv.filter(ImageFilter.GaussianBlur(sigma))
        glow = Image.new("RGBA", cv.size, BLUE_LIGHTER + (0,))
        glow.putalpha(glow_a)
        sharp = Image.new("RGBA", cv.size, (255, 255, 255, 0))
        sharp.putalpha(cv)
        return np.asarray(Image.alpha_composite(glow, sharp)).copy()

    def _bake_blurred_icon(self) -> np.ndarray:
        """BlurredIcon: blur-only (DrawOriginal=false), additive, alpha 0.7.
        Baked as RGBA with rgb=BlueLighter, a=blur*0.7 — drawn under the
        logo splash's additive blend func (SRC_ALPHA, ONE), which blends
        dst + rgb*a, the catch reference's additive field exactly."""
        lk = self.lk
        s = max(4, int(round(BLUR_ICON_SIZE * lk)))
        sigma = BLUR_ICON_SIGMA * lk
        pad = int(math.ceil(3 * sigma)) + 1
        cv = Image.new("L", (s + 2 * pad, s + 2 * pad), 0)
        cv.paste(self._chevron_mask(s), (pad, pad))
        a = np.asarray(cv.filter(ImageFilter.GaussianBlur(sigma)),
                       np.float32) * BLUR_ICON_ALPHA
        rgba = np.zeros(cv.size[::-1] + (4,), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = BLUE_LIGHTER
        rgba[..., 3] = np.round(a).astype(np.uint8)
        return rgba

    def _bake_circle(self) -> np.ndarray:
        """A white antialiased circle (supersampled), the bar's round caps."""
        ss = 4
        s = _CIRCLE_BAKE * ss
        im = Image.new("L", (s, s), 0)
        ImageDraw.Draw(im).ellipse([0, 0, s - 1, s - 1], fill=255)
        im = im.resize((_CIRCLE_BAKE, _CIRCLE_BAKE), Image.LANCZOS)
        rgba = np.full((_CIRCLE_BAKE, _CIRCLE_BAKE, 4), 255, np.uint8)
        rgba[..., 3] = np.asarray(im)
        return rgba

    # --- draw helpers (top-left coords in; the engine's rects are GL
    # bottom-left, converted here once) ---------------------------------------

    def _blit(self, name: str, cx: float, cy: float, alpha: float,
              w: float | None = None, h: float | None = None) -> None:
        tex, tw, th = self._tex[name]
        dw = tw if w is None else w
        dh = th if h is None else h
        self.fr._draw_external_texture(
            tex, x=int(round(cx - dw / 2.0)),
            y=int(round(self.h - (cy + dh / 2.0))),
            w=int(round(dw)), h=int(round(dh)), alpha=alpha)

    def _text(self, text: str, size_1080: int, color, cx: float, cy: float,
              alpha: float, align: str = "cc") -> None:
        """One engine text run (gpu/text bold face via _cached_text — the
        same stack as the score/acc HUD), centred at (cx, cy) top-left-
        space; align 'lc'/'rc' anchor the left/right edge at cx instead."""
        tex, w, h = self.fr._cached_text(text, size_1080, tuple(color) + (255,))
        if align == "lc":
            x = cx
        elif align == "rc":
            x = cx - w
        else:
            x = cx - w / 2.0
        self.fr._draw_external_texture(
            tex, x=int(round(x)),
            y=int(round(self.h - (cy + h / 2.0))),
            w=w, h=h, alpha=alpha)

    def _bar(self, cx: float, cy: float, bw: int, bh: int,
             alpha: float) -> None:
        """remainingTimeBox: a white fully-rounded Circle, h = min(8, w) —
        two half-cap quads + a solid white strip. At w <= h it IS a
        circle (lazer's Circle degenerates the same way)."""
        if bw <= bh:
            self._blit("circle", cx, cy, alpha, w=bw, h=bh)
            return
        cap = bh / 2.0
        self._blit("cap_l", cx - bw / 2.0 + cap / 2.0, cy, alpha,
                   w=cap, h=bh)
        self._blit("cap_r", cx + bw / 2.0 - cap / 2.0, cy, alpha,
                   w=cap, h=bh)
        self._blit("white", cx, cy, alpha, w=bw - 2.0 * cap, h=bh)

    # --- per-frame -----------------------------------------------------------

    def draw(self, scene) -> None:
        """Compose the overlay for this frame. Called every frame by the
        registered break-overlay element (the bar damp runs continuously, like lazer's
        Update); outside break windows it returns before ANY GL call, so
        non-break frames are untouched."""
        if not self.periods:
            return
        t = float(scene.t_ms) * self.rate          # map-time clock
        dt = 16.7 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        # active period: overlay lives over [start, start + D + FADE]
        idx = bisect_right(self._starts, t) - 1
        cur = None
        if idx >= 0:
            s0, D = self.periods[idx]
            if t <= s0 + D + BREAK_FADE_MS:
                cur = (s0, D)

        # remainingTimeBox.Width — DampContinuously toward
        # max(0, (Period.End - now - FADE) / D), EVERY frame, in/out of breaks
        if cur is None:
            target = 0.0
        else:
            s0, D = cur
            target = max(0.0, (s0 + D - t - BREAK_FADE_MS) / D) if D > 0 else 0.0
        self._bar_w = target + (self._bar_w - target) * (0.5 ** (dt / DAMP_HALF_MS))

        if cur is None:
            return
        s0, D = cur
        tp = t - s0                       # time since break start
        # fadeContainer alpha: linear FadeIn/FadeOut over BREAK_FADE_MS
        if tp >= D:
            alpha = max(0.0, 1.0 - (tp - D) / BREAK_FADE_MS)
        else:
            alpha = min(1.0, tp / BREAK_FADE_MS)
        if alpha <= 0.004:
            return

        self._ensure_baked()
        lk = self.lk
        cx, cy = self.w / 2.0, self.h / 2.0
        p_in = _out_quint(tp / BREAK_FADE_MS)

        # 1) shadow blob (first fadeContainer child)
        self._blit("shadow", cx, cy, alpha)

        # 2) progress bar: container width 0 -> 0.3 (OutQuint, 325ms), snap
        #    to 0 at t'=D; pill width rides the damped fraction
        wc = REMAINING_MAX_W * p_in if tp < D else 0.0
        bw = int(round(wc * self.w * max(0.0, min(1.0, self._bar_w))))
        if bw >= 2:
            bh = max(1, int(round(min(BAR_H * lk, bw))))
            self._bar(cx, cy, bw, bh, alpha)

        # 3) remaining-time counter: ceil(count/1000); count runs linearly
        #    from the FULL break duration to 0 at the break's end. Digits =
        #    the engine's HUD text stack at lazer's Numeric-33 scale.
        count = max(0.0, (D + BREAK_FADE_MS) - tp)
        text = str(int(math.ceil(count / 1000.0)))
        dx = -SLIDE_X * lk * (1.0 - p_in)          # MoveToX(-50 -> 0)
        dh = COUNTER_SIZE * lk
        self._text(text, int(round(COUNTER_SIZE * _1080)), (255, 255, 255),
                   cx + dx, cy - VERTICAL_MARGIN * lk - dh / 2.0, alpha)

        # 4) BreakInfo (slides +50 -> 0): title, then Accuracy / Grade lines
        #    split 2px either side of centre; values LIVE like lazer's
        #    bindables (constant mid-break in practice): scene.accuracy is
        #    the live running acc (0..100), scene.live_grade the engine's
        #    own mania grade at t. FormatAccuracy floors, never rounds up.
        dxi = SLIDE_X * lk * (1.0 - p_in)
        cxi = cx + dxi
        y0 = cy + VERTICAL_MARGIN * lk
        self._text("CURRENT PROGRESS", int(round(TITLE_SIZE * _1080)),
                   (255, 255, 255), cxi, y0 + TITLE_SIZE * lk / 2.0, alpha)
        acc = max(0.0, min(100.0, float(scene.accuracy)))
        acc_txt = f"{math.floor(acc * 100.0) / 100.0:.2f}%"  # FormatAccuracy
        grade = grade_display(
            str(getattr(scene, "live_grade", "SS") or "SS"),
            getattr(scene, "mod_acronyms", ()))
        rows = [("Accuracy", acc_txt), ("Grade", grade)]
        line_px = int(round(LINE_SIZE * _1080))
        ly = y0 + (TITLE_SIZE + FLOW_SPACING) * lk
        for label, value in rows:
            mid = ly + LINE_SIZE * lk / 2.0
            self._text(label, line_px, YELLOW,
                       cxi - LINE_MARGIN * lk, mid, alpha, align="rc")
            self._text(value, line_px, YELLOW_LIGHT,
                       cxi + LINE_MARGIN * lk, mid, alpha, align="lc")
            ly += LINE_SIZE * lk

        # 5) arrows, topmost: slide in over the fade (OutQuint), hold, slide
        #    back out from t'=D. X offsets are fractions of the WIDTH.
        if tp >= D:
            po = _out_quint((tp - D) / BREAK_FADE_MS)
            g_off = GLOW_ICON_FINAL + (GLOW_ICON_OFFSCREEN - GLOW_ICON_FINAL) * po
            b_off = BLUR_ICON_FINAL + (BLUR_ICON_OFFSCREEN - BLUR_ICON_FINAL) * po
        else:
            g_off = GLOW_ICON_OFFSCREEN + (GLOW_ICON_FINAL - GLOW_ICON_OFFSCREEN) * p_in
            b_off = BLUR_ICON_OFFSCREEN + (BLUR_ICON_FINAL - BLUR_ICON_OFFSCREEN) * p_in
        # origins CentreRight/CentreLeft: the offset is the icon's inner
        # LAYOUT edge (AutoSize box = the 60/130px icon; the glow/blur
        # overhang is draw-only, like o!f's inflated draw quad) -> shift
        # each sprite centre outward by half the LAYOUT size, not the
        # padded canvas
        g_half = GLOW_ICON_SIZE * lk / 2.0
        b_half = BLUR_ICON_SIZE * lk / 2.0
        # additive pass FIRST (BlurredIcon behind GlowIcon in lazer's child
        # order) — the logo splash's blend-func switch, restored after.
        # Flush any queued batch sprites under the NORMAL blend first (the
        # logo splash takes the same precaution).
        self.fr._flush_sprite_batch()
        ctx = self.fr.rc.ctx
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        self._blit("blur_r", cx - b_off * self.w - b_half, cy, alpha)
        self._blit("blur_l", cx + b_off * self.w + b_half, cy, alpha)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._blit("glow_r", cx - g_off * self.w - g_half, cy, alpha)
        self._blit("glow_l", cx + g_off * self.w + g_half, cy, alpha)
