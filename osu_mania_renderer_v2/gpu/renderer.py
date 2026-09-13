"""GPU draw passes: background, playfield columns, notes, receptors, HUD.

This file is implemented incrementally:
  Task 13: playfield + tap/hold notes (this commit)
  Task 14: receptors, key flash, hit-light, judgments
  Task 15: HUD, banner, flashlight post-pass
  Task 16: background image
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from osu_mania_renderer_v2.beatmap.models import RenderOptions
from osu_mania_renderer_v2.beatmap.mods import LegacyModIcon, legacy_mod_icons
from osu_mania_renderer_v2.beatmap.skin_ini import (
    DEFAULT_COLOUR_BREAK,
    ManiaSection,
    parse_skin_ini,
)
from osu_mania_renderer_v2.gpu.atlas import (
    SpriteAtlas,
    column_variant,
    legacy_mod_slot_name,
)
from osu_mania_renderer_v2.gpu.shaders import load_programs
from osu_mania_renderer_v2.gpu.text import text_to_texture
from osu_mania_renderer_v2.render.dim import build_dim_envelope
from osu_mania_renderer_v2.render.scene import SceneState

log = logging.getLogger("osu_mania_renderer_v2")

# 6 vertices x 9 float32 attrs for the ad-hoc external-texture quads.
# Little-endian f32 == the np.array(dtype="f4").tobytes() it replaces.
_EXT_QUAD_PACK = struct.Struct("<54f").pack

# Playfield dimensions, expressed as fractions of the screen.
# Kept as the legacy "fraction of screen" fallback for code paths that
# don't go through `_compute_playfield_geometry` (HUD elements that
# read PLAYFIELD_X/W_FRAC directly). The geometry resolver itself now
# always derives per-column widths from skin.ini OR lazer defaults.
PLAYFIELD_X_FRAC = 0.36
PLAYFIELD_W_FRAC = 0.28

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
# Notes fill the column width — height ≈ column width, like in-game mania.
NOTE_HEIGHT_REL_COL = 0.95
# Receptor strip lives at the bottom; ≈ column-width tall so receptors are
# rendered close to circular.
RECEPTOR_HEIGHT_REL_COL = 1.0
RECEPTOR_BOTTOM_OFFSET_FRAC = 0.05
# Hit-feedback rainbow strip at the very bottom of the playfield.
HIT_STRIP_HEIGHT_FRAC = 0.012

# osu!lazer LegacyHitExplosion timings. The authored lighting sprite fades in
# from zero for 80 ms, then fades out over 120 ms.
LEGACY_HIT_EXPLOSION_FADE_IN_MS = 80.0
LEGACY_HIT_EXPLOSION_FADE_OUT_MS = 120.0
LEGACY_HIT_EXPLOSION_DURATION_MS = (
    LEGACY_HIT_EXPLOSION_FADE_IN_MS + LEGACY_HIT_EXPLOSION_FADE_OUT_MS
)

LEGACY_JUDGMENT_DURATION_MS = 220.0
LEGACY_JUDGMENT_FRAME_MS = 50.0


def legacy_mania_position_gl(
    position: float,
    render_height: int,
    upside_down: bool,
) -> float:
    """Stable ``ScaleFlipPosition`` followed by top-down to GL conversion."""
    flipped = 480.0 - position if upside_down else position
    return render_height - flipped * render_height / 480.0


def legacy_judgment_frame(age_ms: float, frame_count: int) -> int:
    """Loop a legacy Mania judgment animation from frame zero at 20fps."""
    if frame_count <= 1 or age_ms <= 0:
        return 0
    return int(age_ms / LEGACY_JUDGMENT_FRAME_MS) % frame_count


def legacy_judgment_alpha(age_ms: float) -> float:
    """Stable/lazer 20ms fade-in, 160ms hold, 40ms fade-out envelope."""
    if age_ms < 0 or age_ms >= LEGACY_JUDGMENT_DURATION_MS:
        return 0.0
    if age_ms < 20.0:
        return age_ms / 20.0
    if age_ms < 180.0:
        return 1.0
    return 1.0 - (age_ms - 180.0) / 40.0


def legacy_judgment_scale(age_ms: float, *, miss: bool) -> float:
    """Deterministic equivalent of LegacyManiaJudgementPiece transforms."""
    age = max(0.0, age_ms)
    if miss:
        progress = min(1.0, age / 100.0)
        eased = 1.0 - (1.0 - progress) ** 2  # Easing.Out
        return 1.2 + (1.0 - 1.2) * eased
    if age < 40.0:
        return 0.8 + 0.2 * (age / 40.0)
    if age < 80.0:
        # The source applies an immediate ScaleTo(0.85), then the next
        # 40ms transform runs from 0.85 to 0.7.
        return 0.85 + (0.7 - 0.85) * ((age - 40.0) / 40.0)
    if age < 180.0:
        return 0.7
    if age < 220.0:
        progress = (age - 180.0) / 40.0
        return 0.7 + (0.4 - 0.7) * progress ** 2  # Easing.In
    return 0.4


def legacy_combo_y_scale(age_ms: float) -> float:
    """Stable Mania combo increment stretch: Y 1.4 -> 1 over 300ms."""
    if age_ms < 0 or age_ms >= 300.0:
        return 1.0
    progress = age_ms / 300.0
    return 1.0 + 0.4 * (1.0 - progress) ** 2  # Easing.Out


def legacy_combo_break_animation(age_ms: float) -> tuple[float, float]:
    """Return (scale, alpha) for stable/lazer's 200ms old-combo burst."""
    if age_ms < 0 or age_ms >= 200.0:
        return 1.0, 0.0
    progress = age_ms / 200.0
    return 1.0 + 3.0 * progress, 0.8 * (1.0 - progress)


@dataclass(frozen=True)
class LegacyGlyphPlacement:
    slot: str
    x: float
    top: float
    width: float
    height: float


@dataclass(frozen=True)
class LegacyTextLayout:
    width: float
    height: float
    glyphs: tuple[LegacyGlyphPlacement, ...]


def legacy_glyph_slot(font: str, character: str) -> str | None:
    """Map a legacy score/combo text character to an atlas slot."""
    if character.isdigit():
        return f"{font}_{character}"
    if font == "score":
        return {
            ",": "score_comma",
            ".": "score_dot",
            "%": "score_percent",
            "x": "score_x",
        }.get(character)
    if font == "combo" and character == "x":
        return "combo_x"
    return None


def legacy_text_layout(
    text: str,
    glyph_sizes: dict[str, tuple[float, float]],
    *,
    font: str = "score",
    scale: float,
    overlap: float,
    fixed_digits: bool = True,
) -> LegacyTextLayout:
    """Compose legacy sprite text with fixed digit cells based on glyph 5.

    ``glyph_sizes`` contains only visible/resolved slots. Missing or explicitly
    transparent punctuation contributes neither a draw nor an empty advance.
    """
    reference = glyph_sizes.get("5")
    reference_width = reference[0] if reference is not None else 0.0
    cursor = 0.0
    placements: list[LegacyGlyphPlacement] = []
    max_height = 0.0
    for character in text:
        size = glyph_sizes.get(character)
        slot = legacy_glyph_slot(font, character)
        if size is None or slot is None:
            continue
        native_width, native_height = size
        width = native_width * scale
        height = native_height * scale
        cell_width = (
            reference_width * scale
            if fixed_digits and character.isdigit() and reference_width > 0
            else width
        )
        if placements:
            cursor -= overlap * scale
        placements.append(LegacyGlyphPlacement(
            slot=slot,
            x=cursor + (cell_width - width) / 2.0,
            top=0.0,
            width=width,
            height=height,
        ))
        cursor += cell_width
        max_height = max(max_height, height)
    return LegacyTextLayout(
        width=max(0.0, cursor),
        height=max_height,
        glyphs=tuple(placements),
    )


@dataclass(frozen=True)
class LegacyModIconGeometry:
    position_scale: float
    texture_scale: float
    center: tuple[float, float]
    rect: tuple[float, float, float, float]


def legacy_mod_icon_geometry(
    render_width: int,
    render_height: int,
    index: int,
    native_size: tuple[float, float],
) -> LegacyModIconGeometry:
    """Stable replay-mod placement (480-space) and texture size (768-space)."""
    position_scale = render_height / 480.0
    texture_scale = render_height / 768.0
    center_x = render_width - (30.0 + 50.0 * index) * position_scale
    center_y = render_height - 94.0 * position_scale
    width = native_size[0] * texture_scale
    height = native_size[1] * texture_scale
    return LegacyModIconGeometry(
        position_scale=position_scale,
        texture_scale=texture_scale,
        center=(center_x, center_y),
        rect=(center_x - width / 2.0, center_y - height / 2.0, width, height),
    )


def legacy_lighting_scale(
    section: ManiaSection | None,
    column: int,
    kind: str,
    *,
    effective_column_width: float | None = None,
    legacy_version: float | None = None,
) -> float:
    """Latest-skin lighting scale in R3D's raw 480-reference units.

    Explicit LightingN/LWidth wins when the entry for this column is nonzero;
    otherwise the effective ColumnWidth is used. osu!lazer stores both values
    after multiplying by 1.6 and divides by a 48-unit default. R3D stores the
    raw values, making the equivalent ratio ``width / 30``.
    """
    if kind not in ("n", "l"):
        raise ValueError(f"unknown legacy lighting kind: {kind!r}")
    if legacy_version is not None and legacy_version < 2.5:
        return 1.0

    configured_width = 0.0
    if section is not None:
        widths = section.lighting_n_width if kind == "n" else section.lighting_l_width
        if 0 <= column < len(widths):
            configured_width = widths[column]

    if configured_width != 0:
        return configured_width / 30.0

    if effective_column_width is None:
        column_widths = section.column_width if section is not None else ()
        if len(column_widths) == 1:
            effective_column_width = float(column_widths[0])
        elif 0 <= column < len(column_widths):
            effective_column_width = float(column_widths[column])
        else:
            effective_column_width = 30.0

    return effective_column_width / 30.0


def legacy_hit_explosion_alpha(age_ms: float) -> float:
    """Deterministic 80 ms fade-in followed by a 120 ms fade-out."""
    if age_ms < 0 or age_ms >= LEGACY_HIT_EXPLOSION_DURATION_MS:
        return 0.0
    if age_ms < LEGACY_HIT_EXPLOSION_FADE_IN_MS:
        return age_ms / LEGACY_HIT_EXPLOSION_FADE_IN_MS
    return 1.0 - (
        (age_ms - LEGACY_HIT_EXPLOSION_FADE_IN_MS)
        / LEGACY_HIT_EXPLOSION_FADE_OUT_MS
    )


def legacy_hit_explosion_frame(age_ms: float, frame_count: int) -> int:
    """Source-faithful one-shot frame selection for LightingN."""
    if frame_count <= 1:
        return 0
    frame_length_ms = max(1000.0 / 60.0, 170.0 / frame_count)
    return min(max(0, int(age_ms / frame_length_ms)), frame_count - 1)


@dataclass(frozen=True)
class LegacyHudGeometry:
    """osu!lazer legacy HUD geometry in render pixels."""

    ui_scale: float
    score_height: float
    accuracy_height: float
    score_right_margin: float
    accuracy_right_margin: float
    accuracy_top: float
    mod_height: float


@dataclass(frozen=True)
class ManiaHealthGeometry:
    """osu!stable ``HpBarMania`` geometry in top-left render pixels."""

    stage_scale: float
    texture_scale: float
    rotation_degrees: float
    background_anchor: tuple[float, float]
    background_size: tuple[float, float]
    fill_anchor: tuple[float, float]
    fill_size: tuple[float, float]
    visible_fill_width: float


class LegacyScorebarLayout(StrEnum):
    """R3D presentation choice for a resolved legacy scorebar pair."""

    MANIA_SIDE = "mania_side"
    STANDARD_HUD = "standard_hud"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class LegacyScorebarAssetMetrics:
    """ScaleAdjust-aware alpha geometry for one legacy scorebar image."""

    design_size: tuple[float, float]
    alpha_bbox: tuple[float, float, float, float] | None
    alpha_area: float


@dataclass(frozen=True)
class LegacyScorebarClassification:
    """Cached evidence behind R3D's conservative scorebar layout heuristic."""

    layout: LegacyScorebarLayout
    confidence: str
    background: LegacyScorebarAssetMetrics
    fill: LegacyScorebarAssetMetrics
    gauge_corridor: tuple[float, float, float, float] | None
    outside_gauge_alpha_ratio: float
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class StandardHealthGeometry:
    """osu!stable generic ``HpBar`` geometry in top-left render pixels."""

    position_scale: float
    background_scale: float
    fill_scale: float
    marker_scale: float
    background_anchor: tuple[float, float]
    background_size: tuple[float, float]
    fill_anchor: tuple[float, float]
    fill_size: tuple[float, float]
    visible_fill_width: float
    marker_center: tuple[float, float]
    marker_size: tuple[float, float]


@dataclass(frozen=True)
class ManiaStageSideGeometry:
    """Stable stage-left/right rectangles in OpenGL render pixels."""

    texture_scale: float
    left_rect: tuple[float, float, float, float]
    right_rect: tuple[float, float, float, float]


@dataclass(frozen=True)
class LazerHitErrorMeterGeometry:
    """LegacySkin BarHitErrorMeter dimensions in OpenGL render pixels."""

    ui_scale: float
    left: float
    axis_y: float
    length: float
    bar_thickness: float
    judgment_width: float
    judgment_thickness: float
    chevron_size: float
    centre_outer_size: float
    centre_inner_size: float


@dataclass(frozen=True)
class LazerHitWindowBand:
    judgment: str
    window_ms: float
    relative_length: float
    colour: tuple[float, float, float]


@dataclass(frozen=True)
class LazerHitErrorTickState:
    alpha: float
    width_fraction: float


# OsuColour.ForHitResult() values used by lazer's BarHitErrorMeter.
LAZER_HIT_RESULT_COLOURS: dict[str, tuple[float, float, float]] = {
    "geki": (0x99 / 255, 0xEE / 255, 1.0),
    "300": (0x66 / 255, 0xCC / 255, 1.0),
    "katu": (0xB3 / 255, 0xD9 / 255, 0x44 / 255),
    "100": (0x88 / 255, 0xB3 / 255, 0.0),
    "50": (1.0, 0xCC / 255, 0x22 / 255),
}


def lazer_hit_error_meter_geometry(
    render_width: int,
    render_height: int,
) -> LazerHitErrorMeterGeometry:
    """Scale lazer's 768-space legacy horizontal meter at bottom-centre."""
    ui_scale = render_height / 768.0
    length = 200.0 * ui_scale
    return LazerHitErrorMeterGeometry(
        ui_scale=ui_scale,
        left=(render_width - length) / 2.0,
        # Leave enough of LegacySkin's bottom margin for upright labels and
        # the 14-unit judgment lines without clipping either at the screen.
        axis_y=18.0 * ui_scale,
        length=length,
        bar_thickness=2.0 * ui_scale,
        judgment_width=14.0 * ui_scale,
        judgment_thickness=4.0 * ui_scale,
        chevron_size=8.0 * ui_scale,
        centre_outer_size=8.0 * ui_scale,
        centre_inner_size=4.0 * ui_scale,
    )


def lazer_hit_window_bands(
    windows: tuple[float, float, float, float, float],
) -> tuple[LazerHitWindowBand, ...]:
    """Return Perfect→Meh band sizes relative to the widest hit window."""
    if len(windows) != 5 or windows[-1] <= 0:
        return ()
    maximum = windows[-1]
    return tuple(
        LazerHitWindowBand(judgment, float(window), float(window) / maximum,
                           LAZER_HIT_RESULT_COLOURS[judgment])
        for judgment, window in zip(
            ("geki", "300", "katu", "100", "50"), windows, strict=True,
        )
    )


def hit_error_offset_position(offset_ms: float, max_hit_window: float) -> float:
    """Map a signed hit offset to lazer's clamped 0..1 bar position."""
    if max_hit_window <= 0:
        return 0.5
    return max(0.0, min(1.0, (offset_ms / max_hit_window + 1.0) / 2.0))


def lazer_hit_error_tick_state(age_ms: float) -> LazerHitErrorTickState:
    """Match lazer's 100 ms OutQuint entrance and 5000 ms exit."""
    if age_ms < 0 or age_ms >= 5100:
        return LazerHitErrorTickState(alpha=0.0, width_fraction=0.0)
    if age_ms < 100:
        progress = age_ms / 100.0
        eased = 1.0 - (1.0 - progress) ** 5  # Easing.OutQuint
        return LazerHitErrorTickState(
            alpha=0.6 * eased,
            width_fraction=eased,
        )
    progress = (age_ms - 100.0) / 5000.0
    return LazerHitErrorTickState(
        alpha=0.6 * (1.0 - progress),
        # ResizeWidthTo(0, ..., Easing.InQuint).
        width_fraction=1.0 - progress ** 5,
    )


def next_hit_error_ema(old_average: float, offset_ms: float) -> float:
    """Apply lazer's moving-average fold for one new scored hit."""
    return old_average * 0.9 + offset_ms * 0.1


def configure_direct_texture_sampling(name: str, texture) -> None:
    """Apply the per-slot sampling policy for full-resolution skin art."""
    if name in ("scorebar_bg", "scorebar_colour"):
        # These semi-transparent bars may be rotated for HpBarMania. Repeating
        # their coloured V=0 edge against transparent-black V=1 pixels creates
        # a long dark seam; generated mip levels widen it.
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        return
    texture.build_mipmaps()
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)


def mania_stage_side_geometry(
    playfield_left: float,
    playfield_right: float,
    stage_height: float,
    left_native_width: float,
    right_native_width: float,
) -> ManiaStageSideGeometry:
    """Apply stable's side-sprite origins and 768-reference width scale."""
    texture_scale = stage_height / 768.0
    left_width = max(1.0, left_native_width * texture_scale)
    right_width = max(1.0, right_native_width * texture_scale)
    return ManiaStageSideGeometry(
        texture_scale=texture_scale,
        # StageLeft uses Origins.TopRight; StageRight uses Origins.TopLeft.
        left_rect=(
            playfield_left - left_width,
            0.0,
            left_width,
            stage_height,
        ),
        right_rect=(
            playfield_right,
            0.0,
            right_width,
            stage_height,
        ),
    )


def mania_health_geometry(
    stage_right: float,
    stage_top: float,
    stage_height: float,
    background_native_size: tuple[float, float],
    fill_native_size: tuple[float, float],
    hp: float,
    new_default: bool,
) -> ManiaHealthGeometry:
    """Map stable's 480-space anchors and 768-space sprite scale to R3D."""
    stage_scale = stage_height / 480.0
    # pSprite.ScaleToWindowRatio multiplies HpBarMania's 0.7 scale by
    # WindowManager.RatioInverse (output height / SpriteRes, where SpriteRes
    # is 768). R3D's single Mania stage maps that full logical height onto
    # ``stage_height``.
    texture_scale = 0.7 * stage_height / 768.0
    hp = max(0.0, min(1.0, hp))
    background_size = tuple(
        value * texture_scale for value in background_native_size
    )
    fill_size = tuple(value * texture_scale for value in fill_native_size)
    fill_x, fill_y = (6.6, 474.8) if new_default else (8.0, 478.0)
    visible_fill_width = fill_size[0] * hp
    return ManiaHealthGeometry(
        stage_scale=stage_scale,
        texture_scale=texture_scale,
        rotation_degrees=-90.0,
        background_anchor=(
            stage_right + stage_scale,
            stage_top + stage_height,
        ),
        background_size=background_size,
        fill_anchor=(
            stage_right + fill_x * stage_scale,
            stage_top + fill_y * stage_scale,
        ),
        fill_size=fill_size,
        visible_fill_width=visible_fill_width,
    )


def mania_health_new_default(fill_source: str, marker_source: str) -> bool:
    """Mirror stable's new-default scorebar asset-source test."""
    return fill_source == "bundle" or marker_source in ("user", "beatmap")


def _scorebar_asset_metrics(image: Image.Image | None) -> LegacyScorebarAssetMetrics:
    """Measure one already-decoded scorebar image in design-space units."""
    if image is None:
        return LegacyScorebarAssetMetrics((0.0, 0.0), None, 0.0)
    scale_adjust = float(image.info.get("scale_adjust", 1) or 1)
    if scale_adjust <= 0:
        scale_adjust = 1.0
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    raw_bbox = image.getchannel("A").getbbox()
    alpha_bbox = (
        None
        if raw_bbox is None
        else tuple(float(value) / scale_adjust for value in raw_bbox)
    )
    return LegacyScorebarAssetMetrics(
        design_size=(
            float(image.width) / scale_adjust,
            float(image.height) / scale_adjust,
        ),
        alpha_bbox=alpha_bbox,
        # Alpha mass, rather than RGB or canvas area, ignores transparent
        # padding and treats partially transparent decoration proportionally.
        alpha_area=float(alpha.sum()) / (255.0 * scale_adjust * scale_adjust),
    )


def classify_legacy_scorebar(
    background_image: Image.Image | None,
    fill_image: Image.Image | None,
    *,
    new_default: bool,
) -> LegacyScorebarClassification:
    """Classify custom scorebar art for R3D's presentation-only divergence.

    osu!stable always applies ``HpBarMania`` in Mania. R3D only selects the
    generic horizontal ``HpBar`` when several strong signals identify an
    obvious standard-mode decorative composite. Unusual or incomplete assets
    deliberately remain ``UNCERTAIN`` and therefore use stable's Mania path.
    """
    background = _scorebar_asset_metrics(background_image)
    fill = _scorebar_asset_metrics(fill_image)
    if background.alpha_bbox is None or fill.alpha_bbox is None:
        return LegacyScorebarClassification(
            layout=LegacyScorebarLayout.UNCERTAIN,
            confidence="low",
            background=background,
            fill=fill,
            gauge_corridor=None,
            outside_gauge_alpha_ratio=0.0,
            evidence=("missing-visible-alpha",),
        )

    fill_offset = (7.5, 7.8) if new_default else (3.0, 10.0)
    fill_left, fill_top, fill_right, fill_bottom = fill.alpha_bbox
    fill_visible_width = fill_right - fill_left
    fill_visible_height = fill_bottom - fill_top

    # Stable positions the visible fill bbox at this offset. Padding admits
    # ordinary scorebar frames, glows and anti-aliased flourishes without
    # counting them as separate composite HUD artwork.
    corridor_pad_x = max(24.0, fill_visible_height)
    corridor_pad_y = max(32.0, fill_visible_height)
    corridor = (
        fill_offset[0] + fill_left - corridor_pad_x,
        fill_offset[1] + fill_top - corridor_pad_y,
        fill_offset[0] + fill_right + corridor_pad_x,
        fill_offset[1] + fill_bottom + corridor_pad_y,
    )

    background_alpha = np.asarray(
        background_image.getchannel("A"), dtype=np.float64,
    )
    background_scale_adjust = float(
        background_image.info.get("scale_adjust", 1) or 1,
    )
    if background_scale_adjust <= 0:
        background_scale_adjust = 1.0
    raw_left = max(0, int(np.floor(corridor[0] * background_scale_adjust)))
    raw_top = max(0, int(np.floor(corridor[1] * background_scale_adjust)))
    raw_right = min(
        background_image.width,
        int(np.ceil(corridor[2] * background_scale_adjust)),
    )
    raw_bottom = min(
        background_image.height,
        int(np.ceil(corridor[3] * background_scale_adjust)),
    )
    total_alpha = float(background_alpha.sum())
    inside_alpha = 0.0
    if raw_right > raw_left and raw_bottom > raw_top:
        inside_alpha = float(
            background_alpha[raw_top:raw_bottom, raw_left:raw_right].sum(),
        )
    outside_ratio = (
        max(0.0, min(1.0, 1.0 - inside_alpha / total_alpha))
        if total_alpha > 0
        else 0.0
    )
    outside_area = background.alpha_area * outside_ratio

    bg_width, bg_height = background.design_size
    fill_width, fill_height = fill.design_size
    bg_bbox_height = background.alpha_bbox[3] - background.alpha_bbox[1]
    fill_is_long_and_shallow = (
        fill_visible_width >= 240.0
        and fill_visible_height > 0
        and fill_visible_width / fill_visible_height >= 4.0
    )
    canvas_is_very_tall = (
        bg_width >= 500.0
        and bg_height >= 240.0
        and fill_height > 0
        and bg_height / fill_height >= 4.0
    )
    visible_art_is_very_tall = (
        bg_bbox_height >= 180.0
        and bg_bbox_height / fill_visible_height >= 5.0
    )
    outside_art_is_dominant = (
        outside_ratio >= 0.55
        and outside_area >= max(8000.0, fill.alpha_area * 0.75)
    )

    if (
        fill_is_long_and_shallow
        and canvas_is_very_tall
        and visible_art_is_very_tall
        and outside_art_is_dominant
    ):
        return LegacyScorebarClassification(
            layout=LegacyScorebarLayout.STANDARD_HUD,
            confidence="high",
            background=background,
            fill=fill,
            gauge_corridor=corridor,
            outside_gauge_alpha_ratio=outside_ratio,
            evidence=(
                "long-shallow-fill",
                "very-tall-canvas",
                "very-tall-visible-art",
                "dominant-outside-gauge-alpha",
            ),
        )

    compact_canvas = (
        bg_height <= max(160.0, fill_height * 3.5)
        and bg_bbox_height <= max(140.0, fill_visible_height * 4.0)
    )
    comparable_width = fill_width > 0 and 0.65 <= bg_width / fill_width <= 1.5
    if fill_is_long_and_shallow and compact_canvas and comparable_width:
        return LegacyScorebarClassification(
            layout=LegacyScorebarLayout.MANIA_SIDE,
            confidence="high",
            background=background,
            fill=fill,
            gauge_corridor=corridor,
            outside_gauge_alpha_ratio=outside_ratio,
            evidence=("long-shallow-fill", "compact-scorebar-pair"),
        )

    return LegacyScorebarClassification(
        layout=LegacyScorebarLayout.UNCERTAIN,
        confidence="low",
        background=background,
        fill=fill,
        gauge_corridor=corridor,
        outside_gauge_alpha_ratio=outside_ratio,
        evidence=("composite-signals-incomplete",),
    )


def standard_health_geometry(
    render_height: int,
    background_native_size: tuple[float, float],
    fill_native_size: tuple[float, float],
    marker_native_size: tuple[float, float],
    hp: float,
    *,
    new_default: bool,
) -> StandardHealthGeometry:
    """Port osu!stable's unrotated generic ``HpBar`` presentation."""
    position_scale = render_height / 480.0
    texture_scale = render_height / 768.0
    background_scale = 0.96 * texture_scale
    fill_scale = 0.965 * texture_scale
    marker_scale = 0.97 * texture_scale
    hp = max(0.0, min(1.0, hp))
    fill_x, fill_y = (7.5, 7.8) if new_default else (3.0, 10.0)
    marker_y = 10.625 if new_default else 10.0
    background_size = tuple(
        value * background_scale for value in background_native_size
    )
    fill_size = tuple(value * fill_scale for value in fill_native_size)
    marker_size = tuple(value * marker_scale for value in marker_native_size)
    return StandardHealthGeometry(
        position_scale=position_scale,
        background_scale=background_scale,
        fill_scale=fill_scale,
        marker_scale=marker_scale,
        background_anchor=(0.0, 0.0),
        background_size=background_size,
        fill_anchor=(fill_x * position_scale, fill_y * position_scale),
        fill_size=fill_size,
        visible_fill_width=fill_size[0] * hp,
        # Stable's CurrentXPosition advances by DrawWidth / 1.6 in 480-space,
        # independently of the fill sprite's 0.965 draw scale.
        marker_center=(
            fill_x * position_scale
            + fill_native_size[0] * hp * texture_scale,
            marker_y * position_scale,
        ),
        marker_size=marker_size,
    )


def standard_health_marker_slot(*, hp: float, new_default: bool) -> str:
    """Select the exact marker family used by osu!stable's generic HpBar."""
    if new_default:
        return "scorebar_marker"
    if hp < 0.2:
        return "scorebar_kidanger2"
    if hp < 0.5:
        return "scorebar_kidanger"
    return "scorebar_ki"


def legacy_hud_geometry(
    render_height: int,
    score_native_height: float,
) -> LegacyHudGeometry:
    """Scale legacy score/accuracy/mod components from lazer's 768p space."""
    ui_scale = render_height / 768.0
    score_height = score_native_height * ui_scale * 0.96
    return LegacyHudGeometry(
        ui_scale=ui_scale,
        score_height=score_height,
        accuracy_height=score_native_height * ui_scale * 0.576,
        score_right_margin=10.0 * ui_scale,
        accuracy_right_margin=17.0 * ui_scale,
        accuracy_top=score_height + 9.0 * ui_scale,
        mod_height=48.0 * ui_scale,
    )


def _texture_size_at_height(
    source_width: int,
    source_height: int,
    target_height: float,
) -> tuple[int, int]:
    """Fit raster text to a target height without changing its aspect."""
    height = max(1, int(round(target_height)))
    if source_width <= 0 or source_height <= 0:
        return 1, height
    width = max(1, int(round(source_width * height / source_height)))
    return width, height


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
        # provides note_starts (render.py / wiki_renderer.py); constructions
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
            from osu_mania_renderer_v2.gpu.break_overlay import \
                LazerBreakOverlay
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
            score_prefix=(self.skin_ini.score_prefix
                          if self.skin_ini is not None else "score"),
            combo_prefix=(self.skin_ini.combo_prefix
                          if self.skin_ini is not None else "score"),
        )
        self._legacy_scorebar_classification = (
            self._classify_legacy_scorebar_layout()
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
        applies (matches wiki_elements._common._skin_provides_mania). Used to
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

    def _shared_frame_context(self, scene: SceneState):
        """Reuse the wiki element context when the GPU path shares a pass."""
        ctx = getattr(self, "_shared_element_context", None)
        if ctx is None:
            from osu_mania_renderer_v2.wiki_elements.context import FrameContext

            ctx = FrameContext(
                fr=self,
                skin=None,
                gl=self.rc.ctx,
                fbo=self.rc.fbo,
                width=self.rc.width,
                height=self.rc.height,
                key_count=self.rc.key_count,
            )
            self._shared_element_context = ctx
        ctx.scene = scene
        ctx.t_ms = getattr(scene, "t_ms", 0)
        return ctx

    def _draw_argon_stage_decorations(self, scene: SceneState) -> None:
        from osu_mania_renderer_v2.wiki_elements.stage import stage_decorations

        stage_decorations(
            element=None,
            skin=None,
            assets=None,
            variables=None,
            ctx=self._shared_frame_context(scene),
        )

    def _draw_argon_columns(self, scene: SceneState) -> None:
        from osu_mania_renderer_v2.wiki_elements.stage import columns

        columns(
            element=None,
            skin=None,
            assets=None,
            variables=None,
            ctx=self._shared_frame_context(scene),
        )

    def _draw_argon_notes(self, scene: SceneState) -> None:
        from osu_mania_renderer_v2.wiki_elements.notes import _draw_notes_body

        _draw_notes_body(self._shared_frame_context(scene))

    def _draw_argon_receptors(self, scene: SceneState) -> None:
        from osu_mania_renderer_v2.wiki_elements.notes import _receptors

        _receptors(self._shared_frame_context(scene))

    def _draw_argon_hud(self, scene: SceneState) -> None:
        from osu_mania_renderer_v2.wiki_elements.hud import (
            _draw_argon_hud,
            _draw_fallback_hud,
        )

        ctx = self._shared_frame_context(scene)
        if ctx.has_argon_font():
            _draw_argon_hud(ctx)
        else:
            _draw_fallback_hud(ctx)

    def _legacy_timing_overlay_visibility(self) -> tuple[bool, bool]:
        """Compatibility gate for removed popups and standalone Argon UR."""
        is_argon = self._is_argon_default()
        return (
            False,
            is_argon
            and getattr(self.options, "hud_opacity", 1.0) > 0.0
            and getattr(self.options, "show_unstable_rate", True)
            and getattr(self.options, "show_ur_bar", True),
        )

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
        # Legacy key sprites are screen-edge anchored by LegacyKeyArea, so a
        # custom skin's authored HitPosition must not be moved merely to fit a
        # formerly centre-anchored key rect. Keep the historical clamp only
        # for the Argon/default path whose target remains centred.
        if self.upside_down:
            mirrored = rc.height - self.receptor_centre_y_gl
            if is_argon:
                clamp_margin = self.col_w_uniform + 4
                mirrored = min(mirrored, rc.height - clamp_margin)
            self.receptor_centre_y_gl = mirrored
        elif is_argon:
            clamp_margin = self.col_w_uniform + 4
            min_gl = clamp_margin
            self.receptor_centre_y_gl = max(self.receptor_centre_y_gl, min_gl)

        # Stable's combo and judgment are independent, centre-origin controls.
        # Their 480-space positions flip before stage conversion on upscroll.
        # Keep historical defaults only when the custom skin omits a value.
        if section is not None and section.combo_position is not None:
            self.combo_baseline_y_gl = int(round(legacy_mania_position_gl(
                section.combo_position, rc.height, self.upside_down,
            )))
        else:
            self.combo_baseline_y_gl = int(rc.height * 0.58)

        if section is not None and section.score_position is not None:
            self.score_popup_y_gl = int(round(legacy_mania_position_gl(
                section.score_position, rc.height, self.upside_down,
            )))
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

    def _draw_stage_decorations(self, scene: SceneState | None = None) -> None:
        """Draw the four stage-decoration sprite slots — stage_left,
        stage_right, stage_bottom, stage_hint — that osu!mania skins use
        to theme the playfield (frame textures, hit-line indicators, ...).
        Parsed from `skin.ini` and resolved by the atlas; previously the
        renderer never actually painted any of them, so even skins that
        shipped a full-screen starfield (Night05's 1200x770 stage_left+
        stage_right pair) rendered against plain black.

        Behaviour is generic across all skins: blindly draw each slot at
        the conventional rect. Skins that ship 1x1 transparent
        placeholders (FNF) end up drawing nothing visible, so the change
        is opt-in by what the author shipped — no per-skin special-casing.

        Coordinate convention:
          stage_left  : right edge at the playfield's left edge, full height
          stage_right : left edge at the playfield's right edge, full height
          stage_bottom: horizontal strip flush with the receptor row,
                        roughly the column-width tall
          stage_hint  : thin horizontal indicator at the receptor centre

        UpsideDown skins have already had `receptor_centre_y_gl` flipped
        by `_compute_geometry`, so positions tied to it automatically
        invert; left/right don't depend on orientation."""
        if scene is not None and self._is_argon_default():
            self._draw_argon_stage_decorations(scene)
            return

        rc = self.rc
        h = rc.height
        w = rc.width

        # Side-panel dim: a translucent overlay covering the screen
        # area OUTSIDE the playfield column band. Suppresses the
        # beatmap background's competition with gameplay regardless
        # of whether the skin ships meaningful stage chrome. Skin
        # sprites are drawn on top of this dim, so a starfield-style
        # stage_left/right (Night05) still reads through; skins that
        # ship 1x1 transparent placeholders (Pii AR11, Aristia,
        # Kori's pick) just get the dim. Skin-agnostic by design.
        SIDE_DIM = (0.0, 0.0, 0.0, 0.55)
        # Left side: from screen left to the playfield's left edge.
        if self.pf_x > 0:
            self._draw_sprite(
                "column_bg",
                0, 0, self.pf_x, h, SIDE_DIM,
            )
        # Right side: from the playfield's right edge to the screen right.
        right_x = self.pf_x + self.pf_w
        if right_x < w:
            self._draw_sprite(
                "column_bg",
                right_x, 0, w - right_x, h, SIDE_DIM,
            )

        # Column-area dim: applied only when the skin doesn't ship
        # meaningful stage chrome. Without this, transparent skins
        # like Kori's pick let the beatmap BG bleed THROUGH the column
        # area where notes fall. Skin-agnostic test: a chrome sprite
        # is "meaningful" when its native pixel area exceeds a small
        # threshold — many skins ship 1×1 transparent placeholders
        # for stage_left/right (Pii AR11, Kori, SC arrows, Aristia in
        # some configs) which register as "user" source but contribute
        # zero visible art. Threshold 100px² lets a 10×10 sprite
        # through while rejecting the 1×1 placeholders.
        PLACEHOLDER_THRESHOLD_PX2 = 100
        def _has_meaningful(name: str) -> bool:
            src = self.atlas.global_source(name)
            if src not in ("beatmap", "user"):
                return False
            sw, sh = self.atlas.global_native_size(name)
            return sw * sh > PLACEHOLDER_THRESHOLD_PX2
        if not (_has_meaningful("stage_left")
                or _has_meaningful("stage_right")):
            COL_DIM = (0.0, 0.0, 0.0, 0.55)
            self._draw_sprite(
                "column_bg",
                self.pf_x, 0, self.pf_w, h, COL_DIM,
            )

        # Stage side sprites follow stable's distinct origins: StageLeft is
        # TopRight-anchored at the playfield's left edge and StageRight is
        # TopLeft-anchored at its right edge. Do not clamp the left sprite's
        # transparent canvas into the 4:3 region; off-screen clipping is what
        # keeps its authored right-edge artwork on the actual left side.
        sl_src = self.atlas.global_source("stage_left")
        sr_src = self.atlas.global_source("stage_right")
        sl_w, _sl_h = self.atlas.global_native_size("stage_left")
        sr_w, _sr_h = self.atlas.global_native_size("stage_right")
        side_geometry = mania_stage_side_geometry(
            playfield_left=self.pf_x,
            playfield_right=self.pf_x + self.pf_w,
            stage_height=float(h),
            left_native_width=sl_w,
            right_native_width=sr_w,
        )
        if sl_src in ("beatmap", "user"):
            self._draw_direct(
                "stage_left",
                *side_geometry.left_rect,
                tint=(1, 1, 1, 1),
            )
        if sr_src in ("beatmap", "user"):
            self._draw_direct(
                "stage_right",
                *side_geometry.right_rect,
                tint=(1, 1, 1, 1),
            )

        # Stage-bottom + stage-hint are positioned RELATIVE TO THE
        # RECEPTOR ROW, which already accounts for UpsideDown. In normal
        # mode the receptor sits near the bottom and these draw just
        # below it; in upside-down the receptor is near the top so they
        # flip with it.
        rec_y = self.receptor_centre_y_gl
        rec_h = self.col_w_uniform
        # Stage bottom: a base panel anchored with its TOP at the receptor
        # row, extending down. Size at the source's NATIVE aspect scaled to
        # the playfield width (like stage_left/right) so a TALL key-panel
        # skin (e.g. a 380x576 mania-stage-bottom) stays tall instead of
        # being squashed into a thin ~1.2x-column strip. Falls back to the
        # strip height when the skin ships no meaningful stage-bottom.
        sb_h = int(rec_h * 1.2)
        _pf_src = self.atlas.global_source("playfield_frame")
        if _pf_src in ("beatmap", "user"):
            _pf_asp = self.atlas.global_aspect("playfield_frame")  # width/height
            if _pf_asp and _pf_asp > 0:
                sb_h = max(1, int(self.pf_w / _pf_asp))
        sb_y_gl = rec_y - sb_h
        # When upside-down, mirror so the panel sits ABOVE the receptor.
        if self.upside_down:
            sb_y_gl = rec_y
        # Atlas internal name for `mania-stage-bottom.png` is
        # "playfield_frame" (legacy); `mania-stage-hint.png` is "hit_light".
        self._draw_sprite(
            "playfield_frame",
            self.pf_x, sb_y_gl, self.pf_w, sb_h,
            (1, 1, 1, 1),
        )
        # Stage hint: thin indicator at the receptor centre.
        hint_h = max(2, int(rec_h * 0.15))
        self._draw_sprite(
            "hit_light",
            self.pf_x, rec_y - hint_h // 2, self.pf_w, hint_h,
            (1, 1, 1, 1),
        )

    def draw(self, scene: SceneState) -> None:
        ctx = self.rc.ctx
        fbo = self.rc.fbo
        fbo.use()
        fbo.clear(0.03, 0.03, 0.05, 1.0)

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self._draw_background(scene)
        # Stage decoration sprites (stage_left/right/bottom/hint) — drawn
        # AFTER the song background but BEFORE the column overlays so they
        # form a themed backdrop behind the playfield. Skins that ship
        # real assets (Night05's 1200x770 starfield) get their look;
        # skins that ship 1x1 transparent placeholders (FNF) draw nothing
        # visible, so non-decorated skins are unaffected.
        self._draw_stage_decorations(scene)
        is_argon = self._is_argon_default()
        custom_scorebar_layout = self._selected_legacy_scorebar_layout()
        draw_mania_side_health = (
            self.options.show_hp_bar
            and not is_argon
            and self._has_custom_health_bar_assets()
            and custom_scorebar_layout is not LegacyScorebarLayout.STANDARD_HUD
        )
        if draw_mania_side_health:
            # HpBarMania belongs to StageMania.SpriteManagerBelow. Keep the
            # authored side gauge with stage decoration, before playfield
            # columns/notes and global HUD chrome.
            self._draw_hp_bar(scene)
        self._draw_columns(scene)
        if self.options.show_key_overlay:
            self._draw_stage_lights(scene)

        # KeysUnderNotes (from skin's [Mania] block): when true, the
        # receptors are drawn UNDER the notes so falling notes appear to
        # pass on top of the key strip. Default (and most skins) is the
        # opposite — keys on top of notes — which is the renderer's
        # historical behaviour.
        keys_under_notes = bool(
            self.mania_section is not None
            and self.mania_section.keys_under_notes
        )

        if keys_under_notes:
            self._draw_receptors(scene)

        # HD/FI uniforms only apply to scrolling notes. Flush the pre-notes
        # batch before changing those uniforms, otherwise any queued sprite
        # would be drawn with the wrong HD state.
        self._flush_sprite_batch()
        self.apply_note_cover(
            scene.visual_mods.hidden, scene.visual_mods.fade_in, scene.combo)
        self._draw_notes(scene)
        self._flush_sprite_batch()
        self._hd_active = False
        self._fi_active = False

        # Preserve Argon's established ordering. Custom legacy judgment lives
        # in stable's stage-above manager, so it is drawn after receptors too.
        if is_argon:
            self._draw_combo_and_judgment(scene)

        if not keys_under_notes:
            self._draw_receptors(scene)
        if not is_argon:
            self._draw_combo_and_judgment(scene)
        _show_hit_error_popups, show_ur_summary = self._legacy_timing_overlay_visibility()
        if (
            getattr(self.options, "hud_opacity", 1.0) > 0.0
            and getattr(self.options, "show_hit_error_meter", True)
        ):
            self._draw_hit_error_meter(scene)
        if self.options.show_progress_bar:
            self._draw_progress_bar(scene)
        if scene.hp <= 0.001 and scene.results_opacity <= 0:
            self._draw_fail_overlay()
        if self.options.show_hp_bar and not draw_mania_side_health:
            self._draw_hp_bar(scene)
        self._draw_banner()
        self._draw_hud(scene)
        self._draw_top_chrome(scene)
        if scene.visual_mods.flashlight:
            self._draw_flashlight_pass()
        if show_ur_summary:
            self._draw_ur_summary(scene)
        # lazer z-order: BreakOverlay is a LATER overlay-component child
        # than HUDOverlay (Player.createOverlayComponents) — drawn above
        # every HUD element (hud/top chrome/flashlight/UR), under the
        # miss-flash/fade/results/watermark layers below, matching lazer's
        # Player container order. No-op (zero GL calls) outside breaks.
        if self._break_overlay is not None:
            self._break_overlay.draw(scene)
        # Combo-break red flash: when a ≥20-combo break happened in the
        # last 300 ms, paint a fading red wash over the playfield so the
        # break reads visually as well as audibly. Matches lazer's punchy
        # "you just lost it" feedback.
        if scene.miss_break_age_ms < 300 and scene.results_opacity <= 0:
            t = scene.miss_break_age_ms / 300.0
            alpha = max(0.0, 0.35 * (1.0 - t))
            self._draw_sprite("column_bg", 0, 0,
                              self.rc.width, self.rc.height,
                              (0.95, 0.20, 0.20, alpha))
        if scene.fade_to_black > 0:
            self._draw_sprite("column_bg", 0, 0,
                              self.rc.width, self.rc.height,
                              (0, 0, 0, scene.fade_to_black))
        # R3D intro splash — topmost intro element (over the start fade),
        # matching std/catch. No-op unless options.show_logo is on.
        self.draw_logo_splash(scene.t_ms)
        if scene.results_opacity > 0 and self.options.show_result_screen:
            self._draw_results_overlay(scene)
        # Watermark goes after every other overlay so it's never covered.
        if self.options.watermark_text:
            self._draw_watermark(self.options.watermark_text)
        # Final flush — anything still queued goes out before the FBO
        # readback in render.py picks up this frame's pixels.
        self._flush_sprite_batch()

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

    def set_banner_text(self, text: str) -> None:
        """Build the banner texture, shrinking the font automatically if the
        rendered text would overflow ~35% of the screen width. Avoids the
        old behaviour where long "Artist - Title (Mapper) [Diff]" strings
        ran into the score on the right or got brutally cropped."""
        if hasattr(self, "_banner_text") and self._banner_text == text:
            return
        base_size = max(12, int(48 * self.rc.height / self._FONT_REFERENCE_HEIGHT))
        max_w = int(self.rc.width * 0.35)
        size = base_size
        prev_tex = None  # the previous-iteration tex, safe to release each loop
        while size >= 12:
            tex, w, h = text_to_texture(
                self.rc.ctx, text, size=size, color=(235, 235, 240, 255),
            )
            # Release the prior loop's tex (if any) now that we have a
            # fresh one — never release the current iteration's tex, since
            # it might be the one we end up assigning to self._banner_tex.
            if prev_tex is not None:
                try:
                    if prev_tex.extra is not None:
                        prev_tex.extra.release()
                        prev_tex.extra = None
                    prev_tex.release()
                except Exception:  # noqa: BLE001
                    pass
            if w <= max_w:
                self._banner_tex, self._banner_w, self._banner_h = tex, w, h
                self._banner_text = text
                return
            prev_tex = tex
            size = int(size * 0.9)
        # Fell through with a still-too-wide minimum at size 12; the last
        # tex generated is the smallest we'll get, and it's still live
        # (prev_tex isn't released past the final iteration).
        self._banner_tex, self._banner_w, self._banner_h = tex, w, h
        self._banner_text = text

    def _draw_banner(self) -> None:
        if not hasattr(self, "_banner_tex"):
            return
        # Top-left, with margin matching the score's right-side padding.
        self._draw_external_texture(
            self._banner_tex, x=28,
            y=self.rc.height - self._banner_h - 24,
            w=self._banner_w, h=self._banner_h, alpha=1.0,
        )

    _HUD_CACHE_MAX = 256
    # Reference height the font sizes were authored for. Smaller render
    # targets (720p, etc.) scale every cached text by rc.height / this.
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

    def _legacy_asset_is_visible(self, slot: str) -> bool:
        checker = getattr(self.atlas, "global_has_visible_pixels", None)
        return checker(slot) if checker is not None else True

    def _legacy_font_available(self, font: str) -> bool:
        """A usable custom font has all ten visible, skin-authored digits."""
        for digit in "0123456789":
            slot = f"{font}_{digit}"
            if self.atlas.global_source(slot) not in ("user", "beatmap"):
                return False
            width, height = self.atlas.global_native_size(slot)
            if width <= 0 or height <= 0 or not self._legacy_asset_is_visible(slot):
                return False
        return True

    def _legacy_glyph_sizes(
        self, font: str, text: str,
    ) -> dict[str, tuple[float, float]]:
        sizes: dict[str, tuple[float, float]] = {}
        for character in set(text + "5"):
            slot = legacy_glyph_slot(font, character)
            if slot is None:
                continue
            if self.atlas.global_source(slot) not in ("user", "beatmap"):
                continue
            size = self.atlas.global_native_size(slot)
            if size[0] <= 0 or size[1] <= 0:
                continue
            if not self._legacy_asset_is_visible(slot):
                continue
            sizes[character] = size
        return sizes

    def _draw_legacy_text(
        self,
        font: str,
        text: str,
        *,
        anchor_x: float,
        anchor_y: float,
        align: str,
        scale: float,
        overlap: float,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        tint: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        additive: bool = False,
    ) -> LegacyTextLayout:
        """Draw atlas-backed legacy sprite text and return its base layout."""
        layout = legacy_text_layout(
            text,
            self._legacy_glyph_sizes(font, text),
            font=font,
            scale=scale,
            overlap=overlap,
            fixed_digits=True,
        )
        if align == "right_top":
            center_x = anchor_x - layout.width / 2.0
            center_y = anchor_y - layout.height / 2.0
        elif align == "center":
            center_x = anchor_x
            center_y = anchor_y
        else:
            raise ValueError(f"unknown legacy text alignment: {align!r}")

        left = center_x - layout.width / 2.0
        top = center_y + layout.height / 2.0
        for glyph in layout.glyphs:
            glyph_center_x = left + glyph.x + glyph.width / 2.0
            glyph_center_y = top - glyph.top - glyph.height / 2.0
            glyph_center_x = center_x + (glyph_center_x - center_x) * scale_x
            glyph_center_y = center_y + (glyph_center_y - center_y) * scale_y
            width = glyph.width * scale_x
            height = glyph.height * scale_y
            rect = (
                int(round(glyph_center_x - width / 2.0)),
                int(round(glyph_center_y - height / 2.0)),
                max(1, int(round(width))),
                max(1, int(round(height))),
            )
            atlas_index = self.atlas.index_of(glyph.slot)
            if additive:
                self._draw_additive_sprite_idx(
                    atlas_index, *rect, tint,
                )
            else:
                self._draw_sprite_idx(atlas_index, *rect, tint)
        return layout

    def _draw_generated_mod_fallback(
        self,
        icon: LegacyModIcon,
        *,
        center_x: float,
        center_y: float,
        ui_scale: float,
    ) -> tuple[float, float, float, float]:
        """Keep the existing R3D badge as a missing/modern-mod fallback."""
        body, text_colour = self._PILL_COLOURS.get(
            icon.acronym, self._PILL_DEFAULT,
        )
        height = max(1, int(round(48.0 * ui_scale)))
        texture, text_width, text_height = self._cached_text(
            icon.acronym, 48, (*text_colour, 255),
        )
        text_width, text_height = _texture_size_at_height(
            text_width, text_height, height * 0.48,
        )
        width = max(height, text_width + max(1, int(round(23.0 * ui_scale))))
        x = int(round(center_x - width / 2.0))
        y = int(round(center_y - height / 2.0))
        border = max(1, int(round(height * 0.06)))
        self._draw_sprite(
            "column_bg", x, y, width, height,
            (body[0] / 510, body[1] / 510, body[2] / 510, 0.95),
        )
        self._draw_sprite(
            "column_bg", x + border, y + border,
            width - 2 * border, height - 2 * border,
            (body[0] / 255, body[1] / 255, body[2] / 255, 0.95),
        )
        self._draw_external_texture(
            texture,
            x=x + (width - text_width) // 2,
            y=y + (height - text_height) // 2,
            w=text_width,
            h=text_height,
            alpha=1.0,
        )
        return x, y, width, height

    def _draw_legacy_mod_icons(
        self, scene: SceneState, *, fallback_anchor_y: float,
    ) -> float:
        """Draw custom-skin mods from raw replay flags at stable positions."""
        icons = legacy_mod_icons(int(getattr(scene, "replay_mods", 0)))
        if not icons:
            return fallback_anchor_y
        lowest = fallback_anchor_y
        for index, icon in enumerate(icons):
            slot = (
                legacy_mod_slot_name(icon.asset_name)
                if icon.asset_name is not None else None
            )
            source = self.atlas.global_source(slot) if slot is not None else "missing"
            if source in ("user", "beatmap"):
                native_size = self.atlas.global_native_size(slot)
                geometry = legacy_mod_icon_geometry(
                    self.rc.width, self.rc.height, index, native_size,
                )
                self._draw_direct(slot, *geometry.rect)
                lowest = min(lowest, geometry.rect[1])
                continue

            # Stable has no filename contract for modern-only mods, and custom
            # skins may omit any legacy icon. Retain a generated badge in the
            # same stable position cell as a safe fallback.
            geometry = legacy_mod_icon_geometry(
                self.rc.width, self.rc.height, index, (48.0, 48.0),
            )
            rect = self._draw_generated_mod_fallback(
                icon,
                center_x=geometry.center[0],
                center_y=geometry.center[1],
                ui_scale=geometry.texture_scale,
            )
            lowest = min(lowest, rect[1])
        return lowest

    def _draw_custom_legacy_hud(
        self, scene: SceneState, display_score: int, display_acc: float,
    ) -> None:
        """Skin-sprite score/accuracy plus stable custom-skin mod icons."""
        rc = self.rc
        ui_scale = rc.height / 768.0
        score_bottom = rc.height - 22.0
        if self.options.show_score:
            if self._legacy_font_available("score"):
                score_scale = ui_scale * 0.96
                score_layout = self._draw_legacy_text(
                    "score", f"{display_score:08d}",
                    anchor_x=rc.width - 10.0 * ui_scale,
                    anchor_y=rc.height,
                    align="right_top",
                    scale=score_scale,
                    overlap=(self.skin_ini.score_overlap
                             if self.skin_ini is not None else 0),
                )
                accuracy_top = (
                    rc.height - score_layout.height - 9.0 * ui_scale
                )
                accuracy_layout = self._draw_legacy_text(
                    "score", f"{display_acc:05.2f}%",
                    anchor_x=rc.width - 17.0 * ui_scale,
                    anchor_y=accuracy_top,
                    align="right_top",
                    scale=ui_scale * 0.576,
                    overlap=(self.skin_ini.score_overlap
                             if self.skin_ini is not None else 0),
                    tint=(1.0, 1.0, 1.0, 0.95),
                )
                score_bottom = accuracy_top - accuracy_layout.height
            else:
                # Narrow safety fallback for partial skins: keep the prior PIL
                # readout while all valid sprite fonts take the path above.
                score_texture, score_width, score_height = self._cached_text(
                    f"{display_score:08d}", 120, (255, 255, 255, 255),
                )
                _native_width, native_height = self.atlas.global_native_size(
                    "score_0",
                )
                source = self.atlas.global_source("score_0")
                geometry = (
                    legacy_hud_geometry(rc.height, native_height)
                    if source in ("user", "beatmap") and native_height > 0
                    else None
                )
                if geometry is not None:
                    score_width, score_height = _texture_size_at_height(
                        score_width, score_height, geometry.score_height,
                    )
                    score_x = int(round(
                        rc.width - geometry.score_right_margin - score_width,
                    ))
                    score_y = rc.height - score_height
                else:
                    score_x = rc.width - score_width - 28
                    score_y = rc.height - score_height - 22
                self._draw_external_texture(
                    score_texture,
                    x=score_x, y=score_y,
                    w=score_width, h=score_height, alpha=1.0,
                )
                accuracy_texture, accuracy_width, accuracy_height = self._cached_text(
                    f"{display_acc:.2f}%", 60, (235, 235, 245, 255),
                )
                if geometry is not None:
                    accuracy_width, accuracy_height = _texture_size_at_height(
                        accuracy_width,
                        accuracy_height,
                        geometry.accuracy_height,
                    )
                    accuracy_x = int(round(
                        rc.width - geometry.accuracy_right_margin
                        - accuracy_width,
                    ))
                    accuracy_y = int(round(
                        rc.height - geometry.accuracy_top - accuracy_height,
                    ))
                else:
                    accuracy_x = rc.width - accuracy_width - 28
                    accuracy_y = score_y - accuracy_height - 8
                self._draw_external_texture(
                    accuracy_texture,
                    x=accuracy_x,
                    y=accuracy_y,
                    w=accuracy_width, h=accuracy_height, alpha=0.95,
                )
                score_bottom = accuracy_y

        if self.options.show_mods:
            mods_bottom = self._draw_legacy_mod_icons(
                scene,
                fallback_anchor_y=score_bottom - 16.0 * ui_scale,
            )
        else:
            mods_bottom = score_bottom - 16.0 * ui_scale
        if self.options.show_pp_counter and scene.max_pp > 0:
            texture, width, height = self._cached_text(
                f"{scene.pp:.0f}pp", 44, (255, 220, 140, 255),
            )
            self._draw_external_texture(
                texture,
                x=rc.width - width - 28,
                y=int(round(mods_bottom - height - 14)),
                w=width, h=height, alpha=0.95,
            )

    def _draw_hud(self, scene: SceneState) -> None:
        """Draw the shared Argon HUD or the custom legacy Mania HUD."""
        if self._is_argon_default():
            if getattr(self.options, "hud_opacity", 1.0) > 0:
                self._draw_argon_hud(scene)
            return

        # Score + acc both use the *smoothed* values during gameplay so the
        # counter rolls up across frames rather than snapping each hit.
        # During the results screen we snap to the authoritative replay
        # value (already captured into scene.score / .accuracy).
        if scene.results_opacity > 0:
            display_score = scene.score
            display_acc = scene.accuracy
        else:
            display_score = (scene.score_smoothed
                             if scene.score_smoothed > 0 else scene.score)
            display_acc = scene.accuracy_smoothed

        self._draw_custom_legacy_hud(scene, display_score, display_acc)

    # Per-mod pill colour table. Key-count pill stays red (OG style);
    # other mods get colour-coded so a glance tells you what's active.
    _PILL_COLOURS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
        # label: (body RGB, text RGB)
        "EZ": ((110, 180, 110), (235, 250, 235)),
        "NF": ((110, 130, 180), (225, 235, 250)),
        "HT": ((180, 150, 110), (250, 240, 220)),
        "DT": ((200, 80, 150), (255, 220, 235)),
        "NC": ((200, 80, 150), (255, 220, 235)),
        "HR": ((220, 90, 70), (255, 220, 210)),
        "SD": ((220, 90, 70), (255, 220, 210)),
        "PF": ((220, 90, 70), (255, 220, 210)),
        "HD": ((150, 110, 200), (240, 225, 255)),
        "FI": ((150, 110, 200), (240, 225, 255)),
        "FL": ((90, 80, 100), (235, 230, 245)),
        "MR": ((90, 160, 200), (220, 240, 255)),
        "RD": ((90, 160, 200), (220, 240, 255)),
        "KC": ((90, 160, 200), (220, 240, 255)),
        "V2": ((180, 170, 90), (255, 250, 220)),
    }
    _PILL_KEYCOUNT = ((200, 60, 80), (255, 215, 225))   # red, like the OG "4K"
    _PILL_DEFAULT  = ((60, 110, 200), (215, 230, 255))  # fallback blue

    def _draw_mode_pills(
        self,
        scene: SceneState,
        anchor_y: int,
        legacy_ui_scale: float | None = None,
    ) -> int:
        """Render one pill per active mod, driven by `scene.mod_acronyms`.
        Returns the Y of the pill row's bottom edge (so callers can stack
        further HUD elements — e.g. the PP readout — directly under it)."""
        if not scene.mod_acronyms:
            return anchor_y
        rc = self.rc
        if legacy_ui_scale is None:
            right_pad = max(12, int(
                28 * self.rc.height / self._FONT_REFERENCE_HEIGHT,
            ))
            pill_h = max(40, int(
                100 * self.rc.height / self._FONT_REFERENCE_HEIGHT,
            ))
            min_pill_w = 108
            horizontal_padding = 48
            spacing = 12
        else:
            right_pad = max(1, int(round(10 * legacy_ui_scale)))
            pill_h = max(1, int(round(48 * legacy_ui_scale)))
            min_pill_w = max(1, int(round(pill_h * 1.08)))
            horizontal_padding = max(1, int(round(pill_h * 0.48)))
            spacing = max(1, int(round(pill_h * 0.12)))
        # Lay out right-to-left so the rightmost pill stays anchored.
        x_right = rc.width - right_pad
        for i, label in enumerate(reversed(scene.mod_acronyms)):
            is_keycount = (i == len(scene.mod_acronyms) - 1)
            body, text_col = (
                self._PILL_KEYCOUNT if is_keycount
                else self._PILL_COLOURS.get(label, self._PILL_DEFAULT)
            )
            tex, tw, th = self._cached_text(label, 48, (*text_col, 255))
            if legacy_ui_scale is not None:
                tw, th = _texture_size_at_height(tw, th, pill_h * 0.48)
            pill_w = max(min_pill_w, tw + horizontal_padding)
            px = x_right - pill_w
            py = anchor_y - pill_h
            # Two-layer badge: a darker outer "border" sprite + a brighter
            # inner body for a flatter, icon-like look (matches the visual
            # weight of lazer's mod chips).
            border_w = (
                max(1, int(round(pill_h * 0.06)))
                if legacy_ui_scale is not None
                else max(2, int(pill_h * 0.06))
            )
            self._draw_sprite(
                "column_bg", px, py, pill_w, pill_h,
                (body[0] / 510, body[1] / 510, body[2] / 510, 0.95),
            )
            self._draw_sprite(
                "column_bg",
                px + border_w, py + border_w,
                pill_w - 2 * border_w, pill_h - 2 * border_w,
                (body[0] / 255, body[1] / 255, body[2] / 255, 0.95),
            )
            self._draw_external_texture(
                tex,
                x=px + (pill_w - tw) // 2,
                y=py + (pill_h - th) // 2,
                w=tw, h=th, alpha=1.0,
            )
            x_right = px - spacing
        # py is the GL bottom-left Y of the last pill — same row for all.
        return py

    def _draw_custom_legacy_judgment(self, scene: SceneState) -> None:
        if not self.options.show_judgment or not scene.active_judgments:
            return
        judgment = scene.active_judgments[-1]
        alpha = legacy_judgment_alpha(judgment.age_ms)
        if alpha <= 0:
            return
        slot = f"judgment_{judgment.judgment}"
        native_width, native_height = self.atlas.global_native_size(slot)
        if native_width <= 0 or native_height <= 0:
            return
        texture_scale = self.rc.height / 768.0
        animation_scale = legacy_judgment_scale(
            judgment.age_ms, miss=judgment.judgment == "miss",
        )
        width = max(1, int(round(
            native_width * texture_scale * animation_scale,
        )))
        height = max(1, int(round(
            native_height * texture_scale * animation_scale,
        )))
        center_x = self.pf_x + self.pf_w / 2.0
        center_y = self.score_popup_y_gl
        base = self.atlas.index_of(slot)
        frame = legacy_judgment_frame(
            judgment.age_ms, self.atlas.frame_count(slot),
        )
        self._draw_sprite_idx(
            base + frame,
            int(round(center_x - width / 2.0)),
            int(round(center_y - height / 2.0)),
            width,
            height,
            (1.0, 1.0, 1.0, alpha),
        )

    def _draw_custom_legacy_combo(
        self, scene: SceneState, *, draw_combo: bool,
    ) -> None:
        if not draw_combo or not self.options.show_combo:
            return
        center_x = self.pf_x + self.pf_w / 2.0
        center_y = self.combo_baseline_y_gl
        texture_scale = self.rc.height / 768.0
        overlap = self.skin_ini.combo_overlap if self.skin_ini is not None else 0
        if self._legacy_font_available("combo"):
            if scene.combo > 0:
                self._draw_legacy_text(
                    "combo", str(scene.combo),
                    anchor_x=center_x,
                    anchor_y=center_y,
                    align="center",
                    scale=texture_scale,
                    overlap=overlap,
                    scale_y=legacy_combo_y_scale(scene.combo_age_ms),
                )
            break_scale, break_alpha = legacy_combo_break_animation(
                getattr(scene, "combo_break_age_ms", 9999),
            )
            previous = int(getattr(scene, "combo_break_previous_value", 0))
            if previous > 0 and break_alpha > 0:
                break_colour = (
                    self.mania_section.colour_break
                    if self.mania_section is not None
                    and self.mania_section.colour_break is not None
                    else DEFAULT_COLOUR_BREAK
                )
                self._draw_legacy_text(
                    "combo", str(previous),
                    anchor_x=center_x,
                    anchor_y=center_y,
                    align="center",
                    scale=texture_scale,
                    overlap=overlap,
                    scale_x=break_scale,
                    scale_y=break_scale,
                    tint=(
                        break_colour[0] / 255.0,
                        break_colour[1] / 255.0,
                        break_colour[2] / 255.0,
                        break_alpha,
                    ),
                    additive=True,
                )
            return

        # Partial/no legacy combo font: keep a deterministic white fallback,
        # but still use the source position and vertical-only stretch.
        if scene.combo <= 0:
            return
        texture, width, height = self._cached_text(
            str(scene.combo), 110, (255, 255, 255, 255),
        )
        stretch_y = legacy_combo_y_scale(scene.combo_age_ms)
        drawn_height = max(1, int(round(height * stretch_y)))
        self._draw_external_texture(
            texture,
            x=int(round(center_x - width / 2.0)),
            y=int(round(center_y - drawn_height / 2.0)),
            w=width,
            h=drawn_height,
            alpha=0.95,
        )

    def _draw_combo_and_judgment(
        self, scene: SceneState, draw_combo: bool = True,
    ) -> None:
        """Draw the shared Argon or custom legacy stage presentation."""
        if not self._is_argon_default():
            self._draw_custom_legacy_judgment(scene)
            self._draw_custom_legacy_combo(scene, draw_combo=draw_combo)
            return
        self._draw_argon_combo_and_judgment(scene, draw_combo=draw_combo)

    def _draw_argon_combo_and_judgment(
        self, scene: SceneState, draw_combo: bool = True,
    ) -> None:
        """Draw the shared Argon counter and text-judgment presentation."""
        from osu_mania_renderer_v2.wiki_elements.notes import (
            _argon_combo_and_judgment,
        )

        _argon_combo_and_judgment(
            self._shared_frame_context(scene),
            draw_combo=draw_combo,
        )

    def _draw_judgments(self, scene: SceneState) -> None:
        # Sprite-based per-column popups are superseded by the big centred
        # display in _draw_combo_and_judgment. Kept as a no-op so the draw
        # ordering in draw() doesn't have to change.
        return

    def _draw_hit_strip(self, scene: SceneState | None = None) -> None:
        """Unstable-rate bar at the bottom of the playfield.

        Draws the rainbow timing gradient (colour reference for hit windows)
        and overlays a tick for each recent hit at its signed offset (left =
        early, right = late). The 50-hit window (±127 ms) maps to the strip
        edges so a tick at far-right means a barely-counted late press.
        """
        rc = self.rc
        pf_x = self.pf_x
        pf_w = self.pf_w
        strip_h = max(6, int(rc.height * HIT_STRIP_HEIGHT_FRAC))
        self._draw_sprite("hit_strip", pf_x, 0, pf_w, strip_h, (1, 1, 1, 1))
        # Centre tick mark for the 0 ms line.
        tick_w = 2
        self._draw_sprite("column_bg", pf_x + pf_w // 2 - tick_w // 2,
                          0, tick_w, strip_h * 2, (1, 1, 1, 1))

        if scene is None or not scene.recent_offsets:
            return

        # Each recent hit drops a thin white tick at its x = offset_to_x_px.
        # Map [-127ms, +127ms] (= 50-hit window) → [pf_x, pf_x + pf_w].
        max_off = 127.0
        tick_pixel_w = max(2, int(pf_w * 0.004))
        tick_full_h = strip_h * 2 + max(2, int(rc.height * 0.008))
        for i, off in enumerate(scene.recent_offsets):
            clipped = max(-max_off, min(max_off, off))
            x = int(pf_x + (0.5 + clipped / (2 * max_off)) * pf_w)
            # Newer ticks brighter, older ticks fade.
            age = 1.0 - i / max(1, len(scene.recent_offsets))
            alpha = 0.35 + 0.6 * (1.0 - age)
            self._draw_sprite(
                "column_bg",
                x - tick_pixel_w // 2, 0,
                tick_pixel_w, tick_full_h,
                (1, 1, 1, alpha),
            )

    def _draw_hit_error_meter(self, scene: SceneState) -> None:
        """Draw Argon's dual vertical or LegacySkin's horizontal meter."""
        if self._is_argon_default():
            from osu_mania_renderer_v2.wiki_elements.hud import _argon_hit_error

            _argon_hit_error(self._shared_frame_context(scene))
        else:
            self._draw_lazer_hit_error_meter(scene)

    def _draw_lazer_hit_error_meter(self, scene: SceneState) -> None:
        """Draw lazer's compact horizontal legacy BarHitErrorMeter."""
        geometry = lazer_hit_error_meter_geometry(
            self.rc.width, self.rc.height,
        )
        bands = lazer_hit_window_bands(scene.hit_error_windows)
        if not bands:
            return

        centre_x = geometry.left + geometry.length / 2.0
        half_length = geometry.length / 2.0
        axis_y = geometry.axis_y
        bar_h = max(1, int(round(geometry.bar_thickness)))
        bar_y = int(round(axis_y - bar_h / 2.0))

        def draw_band_side(
            side: int,
            extent: float,
            colour: tuple[float, float, float],
            alpha: float = 1.0,
        ) -> None:
            width = max(1, int(round(extent)))
            x = centre_x if side > 0 else centre_x - extent
            self._draw_sprite(
                "column_bg", int(round(x)), bar_y, width, bar_h,
                (*colour, alpha),
            )

        # The widest (Meh) layer is solid through 80% then fades to transparent
        # over its outer 20%, matching BarHitErrorMeter.createColourBar().
        outer = bands[-1]
        solid_extent = half_length * 0.8
        for side in (-1, 1):
            draw_band_side(side, solid_extent, outer.colour)
            gradient_start = solid_extent
            gradient_width = half_length * 0.2
            slices = 8
            for index in range(slices):
                start = gradient_start + gradient_width * index / slices
                end = gradient_start + gradient_width * (index + 1) / slices
                alpha = 1.0 - (index + 0.5) / slices
                width = max(1, int(round(end - start)))
                x = centre_x + start if side > 0 else centre_x - end
                self._draw_sprite(
                    "column_bg", int(round(x)), bar_y, width, bar_h,
                    (*outer.colour, alpha),
                )

        # Overlay successively narrower windows so the visible axis reads
        # centre → Perfect → Great → Good → Ok → Meh on both sides.
        for band in reversed(bands[:-1]):
            extent = half_length * band.relative_length
            for side in (-1, 1):
                draw_band_side(side, extent, band.colour)

        # Circle-style centre marker (8-unit outer, 4-unit darkened inner).
        outer_size = max(1, int(round(geometry.centre_outer_size)))
        inner_size = max(1, int(round(geometry.centre_inner_size)))
        marker_colour = outer.colour
        self._draw_sprite(
            "note_circle",
            int(round(centre_x - outer_size / 2.0)),
            int(round(axis_y - outer_size / 2.0)),
            outer_size, outer_size, (*marker_colour, 1.0),
        )

        # Judgment lines use their real ages, result colours and additive
        # blending. A single additive batch avoids one GL flush per event.
        ticks: list[tuple[int, int, int, int, tuple]] = []
        max_window = bands[-1].window_ms
        for event in scene.hit_error_events[-50:]:
            state = lazer_hit_error_tick_state(event.age_ms)
            if state.alpha <= 0 or state.width_fraction <= 0:
                continue
            position = hit_error_offset_position(event.offset_ms, max_window)
            x = geometry.left + position * geometry.length
            tick_w = max(1, int(round(geometry.judgment_thickness)))
            tick_h = max(1, int(round(
                geometry.judgment_width * state.width_fraction,
            )))
            colour = LAZER_HIT_RESULT_COLOURS.get(
                event.judgment, (1.0, 1.0, 1.0),
            )
            ticks.append((
                int(round(x - tick_w / 2.0)),
                int(round(axis_y - tick_h / 2.0)),
                tick_w, tick_h, (*colour, state.alpha),
            ))
        if ticks:
            self._flush_sprite_batch()
            ctx = self.rc.ctx
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            try:
                for x, y, width, height, tint in ticks:
                    self._draw_sprite(
                        "column_bg", x, y, width, height, tint,
                    )
                self._flush_sprite_batch()
            finally:
                ctx.blend_func = (
                    moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
                )

        inner_colour = tuple(channel * 0.7 for channel in marker_colour)
        self._draw_sprite(
            "note_circle",
            int(round(centre_x - inner_size / 2.0)),
            int(round(axis_y - inner_size / 2.0)),
            inner_size, inner_size, (*inner_colour, 1.0),
        )

        # Lazer eases this chevron over 800 ms. The renderer carries the exact
        # EMA and positions it once per output frame, avoiding mutable draw
        # state while retaining the same 0.9/0.1 smoothing semantics.
        if scene.hit_error_ema_ms is not None:
            position = hit_error_offset_position(
                scene.hit_error_ema_ms, max_window,
            )
            arrow_x = geometry.left + position * geometry.length
            size = max(2, int(round(geometry.chevron_size)))
            row_h = max(1, int(round(size / 4.0)))
            arrow_bottom = axis_y + geometry.centre_outer_size / 2.0 + 2
            for row in range(4):
                width = max(1, int(round(size * (row + 1) / 4.0)))
                self._draw_sprite(
                    "column_bg",
                    int(round(arrow_x - width / 2.0)),
                    int(round(arrow_bottom + row * row_h)),
                    width, row_h, (1.0, 1.0, 1.0, 0.9),
                )

        # Text is the narrow fallback for lazer's upright hare/tortoise icons.
        label_y = int(round(
            axis_y + geometry.judgment_width / 2.0 + 4 * geometry.ui_scale,
        ))
        for text, x in (("EARLY", geometry.left),
                        ("LATE", geometry.left + geometry.length)):
            texture, width, height = self._cached_text(
                text, 14, (225, 225, 235, 210),
            )
            self._draw_external_texture(
                texture,
                x=int(round(x - width / 2.0)), y=label_y,
                w=width, h=height, alpha=0.82,
            )

    def _draw_hp_bar(self, scene: SceneState) -> None:
        """Draw the cached custom layout choice, or the procedural fallback."""
        if not getattr(self.options, "show_hp_bar", True):
            return
        # The Argon health tube is part of the shared Argon HUD. Standard-mode
        # scorebar assets do not replace it when the skin supplies no Mania
        # presentation contract.
        if self._is_argon_default():
            return
        hp = max(0.0, min(1.0, scene.hp))
        if self._has_custom_health_bar_assets():
            if (
                self._selected_legacy_scorebar_layout()
                is LegacyScorebarLayout.STANDARD_HUD
            ):
                self._draw_standard_legacy_health_bar(hp)
                return
            self._draw_mania_health_bar(hp)
            return

        # Existing R3D fallback: vertical HP track on the left of the
        # playfield, fed by actual per-judgment HP deltas.
        rc = self.rc
        pf_x = self.pf_x
        pf_w = self.pf_w
        bar_w = max(6, int(rc.width * 0.008))
        bar_x = pf_x - bar_w - max(4, int(rc.width * 0.004))
        col_w = self.col_w_uniform
        rec_h = int(col_w * RECEPTOR_HEIGHT_REL_COL)
        bar_y = int(rc.height * RECEPTOR_BOTTOM_OFFSET_FRAC) + rec_h + 8
        bar_h = rc.height - bar_y - 80
        # Dim track behind the fill.
        self._draw_sprite("column_bg", bar_x, bar_y, bar_w, bar_h,
                          (0.15, 0.15, 0.2, 0.6))
        # Filled portion colour shifts red as HP gets low so failing plays
        # are visually obvious even without an actual "fail" event.
        if hp > 0.5:
            r, g, b = 0.65, 0.35, 0.95   # purple
        elif hp > 0.2:
            r, g, b = 0.95, 0.55, 0.25   # orange
        else:
            r, g, b = 0.95, 0.30, 0.30   # red
        fill_h = int(bar_h * hp)
        self._draw_sprite("column_bg", bar_x, bar_y, bar_w, fill_h,
                          (r, g, b, 0.95))

    def _has_custom_health_bar_assets(self) -> bool:
        """Whether a same-tier custom background/fill pair was resolved."""
        source = self.atlas.global_source("scorebar_bg")
        if source not in ("user", "beatmap"):
            return False
        bg_w, bg_h = self.atlas.global_native_size("scorebar_bg")
        fill_w, fill_h = self.atlas.global_native_size("scorebar_colour")
        return (
            self.atlas.global_source("scorebar_colour") == source
            and bg_w > 0
            and bg_h > 0
            and fill_w > 0
            and fill_h > 0
        )

    def _classify_legacy_scorebar_layout(
        self,
    ) -> LegacyScorebarClassification | None:
        """Classify and report a custom pair once, during renderer setup."""
        # Structural Mania evidence is authoritative. A standard-only skin's
        # global scorebar art neither disables Argon nor enters the legacy
        # Mania scorebar heuristic.
        if self._is_argon_default():
            return None
        if not self._has_custom_health_bar_assets():
            return None
        atlas = self.atlas
        new_default = mania_health_new_default(
            atlas.global_source("scorebar_colour"),
            atlas.global_source("scorebar_marker"),
        )
        classification = classify_legacy_scorebar(
            atlas.direct_image("scorebar_bg"),
            atlas.direct_image("scorebar_colour"),
            new_default=new_default,
        )
        action = (
            "standard-hpbar"
            if classification.layout is LegacyScorebarLayout.STANDARD_HUD
            else "stable-hpbar-mania"
        )
        log.info(
            "legacy scorebar: layout=%s confidence=%s bg_design=%s "
            "fill_design=%s bg_alpha_bbox=%s fill_alpha_bbox=%s "
            "outside_gauge_alpha=%.3f evidence=%s action=%s",
            classification.layout.value,
            classification.confidence,
            classification.background.design_size,
            classification.fill.design_size,
            classification.background.alpha_bbox,
            classification.fill.alpha_bbox,
            classification.outside_gauge_alpha_ratio,
            ",".join(classification.evidence),
            action,
        )
        return classification

    def _selected_legacy_scorebar_layout(self) -> LegacyScorebarLayout:
        """Return the initialization-time choice; absent data stays faithful."""
        classification = getattr(
            self, "_legacy_scorebar_classification", None,
        )
        if classification is None:
            return LegacyScorebarLayout.UNCERTAIN
        return classification.layout

    def _draw_mania_health_bar(self, hp: float) -> None:
        """Draw skin-authored scorebar assets as stable's rotated side gauge."""
        atlas = self.atlas
        rc = self.rc
        background_native = atlas.global_native_size("scorebar_bg")
        fill_native = atlas.global_native_size("scorebar_colour")
        geometry = mania_health_geometry(
            stage_right=self.pf_x + self.pf_w,
            stage_top=0.0,
            stage_height=float(rc.height),
            background_native_size=background_native,
            fill_native_size=fill_native,
            hp=hp,
            new_default=mania_health_new_default(
                atlas.global_source("scorebar_colour"),
                atlas.global_source("scorebar_marker"),
            ),
        )

        self._draw_mania_health_piece(
            "scorebar_bg",
            anchor=geometry.background_anchor,
            source_size=geometry.background_size,
            visible_fraction=1.0,
        )
        self._draw_mania_health_piece(
            "scorebar_colour",
            anchor=geometry.fill_anchor,
            source_size=geometry.fill_size,
            visible_fraction=hp,
        )

    def _draw_standard_legacy_health_bar(self, hp: float) -> None:
        """Draw an obvious standard-mode composite using stable's HpBar."""
        atlas = self.atlas
        new_default = mania_health_new_default(
            atlas.global_source("scorebar_colour"),
            atlas.global_source("scorebar_marker"),
        )
        marker_slot = standard_health_marker_slot(
            hp=hp, new_default=new_default,
        )
        marker_source = atlas.global_source(marker_slot)
        marker_native = atlas.global_native_size(marker_slot)
        marker_visible = getattr(atlas, "global_has_visible_pixels", None)
        marker_is_usable = (
            marker_source in ("user", "beatmap")
            and marker_native[0] > 0
            and marker_native[1] > 0
            and (
                marker_visible(marker_slot)
                if callable(marker_visible)
                else True
            )
        )
        if not marker_is_usable:
            marker_native = (0.0, 0.0)

        geometry = standard_health_geometry(
            render_height=self.rc.height,
            background_native_size=atlas.global_native_size("scorebar_bg"),
            fill_native_size=atlas.global_native_size("scorebar_colour"),
            marker_native_size=marker_native,
            hp=hp,
            new_default=new_default,
        )
        background_x, background_top = geometry.background_anchor
        background_width, background_height = geometry.background_size
        self._draw_direct(
            "scorebar_bg",
            background_x,
            self.rc.height - background_top - background_height,
            background_width,
            background_height,
            tint=(1.0, 1.0, 1.0, 1.0),
        )

        fill_x, fill_top = geometry.fill_anchor
        fill_width, fill_height = geometry.fill_size
        self._draw_direct_clipped_x(
            "scorebar_colour",
            x=fill_x,
            y=self.rc.height - fill_top - fill_height,
            full_width=fill_width,
            height=fill_height,
            visible_fraction=hp,
            tint=(1.0, 1.0, 1.0, 1.0),
            rotation_deg=0.0,
        )

        if marker_is_usable:
            marker_width, marker_height = geometry.marker_size
            marker_x, marker_top = geometry.marker_center
            marker_rect = (
                marker_x - marker_width / 2.0,
                self.rc.height - marker_top - marker_height / 2.0,
                marker_width,
                marker_height,
            )
            self._draw_standard_health_marker(
                marker_slot,
                marker_rect,
                additive=new_default,
            )

    def _draw_standard_health_marker(
        self,
        slot: str,
        rect: tuple[float, float, float, float],
        *,
        additive: bool,
    ) -> None:
        """Draw stable's standard marker without leaking its blend mode."""
        if not additive:
            self._draw_direct(slot, *rect)
            return
        self._flush_sprite_batch()
        self.rc.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        try:
            self._draw_direct(slot, *rect)
        finally:
            self.rc.ctx.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )

    def _draw_mania_health_piece(
        self,
        name: str,
        *,
        anchor: tuple[float, float],
        source_size: tuple[float, float],
        visible_fraction: float,
    ) -> None:
        """Rotate a source-X crop upward from a stable top-left-space anchor."""
        visible_fraction = max(0.0, min(1.0, visible_fraction))
        visible_width = source_size[0] * visible_fraction
        if visible_width <= 0 or source_size[1] <= 0:
            return

        # Stable's -90-degree screen-space rotation is +90 degrees in this
        # OpenGL bottom-left coordinate system. Re-centre the pre-rotation
        # rectangle so its rotated bottom-left corner remains at ``anchor``.
        anchor_x, anchor_top_y = anchor
        anchor_y_gl = self.rc.height - anchor_top_y
        draw_x = anchor_x + (source_size[1] - visible_width) / 2.0
        draw_y = anchor_y_gl + (visible_width - source_size[1]) / 2.0
        self._draw_direct_clipped_x(
            name,
            x=draw_x,
            y=draw_y,
            full_width=source_size[0],
            height=source_size[1],
            visible_fraction=visible_fraction,
            tint=(1.0, 1.0, 1.0, 1.0),
            rotation_deg=90.0,
        )

    HIT_ERROR_FADE_MS = 600
    HIT_ERROR_RISE_PX = 80   # how far the label drifts upward over its life

    def _draw_hit_error_popups(self, scene: SceneState) -> None:
        """Floating per-column "+8 ms" / "−12 ms" labels that drift upward
        from the receptor after each hit and fade out — same readout
        lazer shows for instant timing feedback."""
        if not scene.hit_light_age_ms:
            return
        rc = self.rc
        pf_x = self.pf_x
        pf_w = self.pf_w
        col_w = self.col_w_uniform
        rec_h = int(col_w * RECEPTOR_HEIGHT_REL_COL)
        rec_y = int(rc.height * RECEPTOR_BOTTOM_OFFSET_FRAC)
        for c in range(rc.key_count):
            if c >= len(scene.hit_light_age_ms):
                break
            age = scene.hit_light_age_ms[c]
            if age >= self.HIT_ERROR_FADE_MS:
                continue
            jud = scene.hit_light_judgment[c] if c < len(scene.hit_light_judgment) else ""
            if jud == "":
                continue
            t = age / self.HIT_ERROR_FADE_MS
            offset = (scene.hit_offset_per_col[c]
                      if c < len(scene.hit_offset_per_col) else 0.0)
            sign = "+" if offset >= 0 else ""
            r, g, b = self._JUDGMENT_LIGHT.get(jud, (220, 220, 240))
            alpha = max(0.0, 1.0 - t)
            tex, w, h = self._cached_text(
                f"{sign}{offset:.0f} ms", 28, (r, g, b, 255),
            )
            x = self.col_x[c] + (self.col_w[c] - w) // 2
            y = rec_y + rec_h + 6 + int(self.HIT_ERROR_RISE_PX * t)
            self._draw_external_texture(
                tex, x=x, y=y, w=w, h=h, alpha=alpha,
            )

    def _draw_fail_overlay(self) -> None:
        """Greys out the playfield + paints a big red 'FAILED' label when
        HP hits zero. Triggered by `scene.hp <= 0`."""
        rc = self.rc
        self._draw_sprite("column_bg", 0, 0, rc.width, rc.height,
                          (0, 0, 0, 0.55))
        tex, w, h = self._cached_text(
            "FAILED", 140, (240, 70, 70, 255),
        )
        self._draw_external_texture(
            tex,
            x=(rc.width - w) // 2,
            y=(rc.height - h) // 2,
            w=w, h=h, alpha=1.0,
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

    def _draw_top_chrome(self, scene: SceneState) -> None:
        """No-op now that the 'UNRANKED' top-centre label is gone — it was
        overlapping the player name in the banner. Mode pills handled by
        _draw_hud already."""
        return

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
        # release sites (_cached_text eviction, set_banner_text) release the
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
            bl = _rot(x, y); br = _rot(x + w, y)
            tr2 = _rot(x + w, y + h); tl = _rot(x, y + h)
        else:
            bl = (x0, y0); br = (x1, y0); tr2 = (x1, y1); tl = (x0, y1)
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
        self, name: str, x: float, y: float, w: float, h: float,
        tint: tuple = (1.0, 1.0, 1.0, 1.0),
        source_u_end: float = 1.0,
        rotation_deg: float = 0.0,
    ) -> None:
        """Draw a wide skin sprite (scorebar / stage panel) at full resolution,
        bypassing the layered 256² atlas (which would crush a 1366-wide bar).
        The source texture is cached as a single-layer array. Source-U cropping
        happens before optional centre rotation, preserving authored pixels."""
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
            configure_direct_texture_sampling(name, arr)
            self._direct_arr_cache[name] = arr
        # Land on top of the queued batch (correct alpha order).
        self._flush_sprite_batch()
        ctx = self.rc.ctx
        prog = self.programs["sprite"]
        sw, sh = self.rc.width, self.rc.height
        x0, x1 = (x / sw) * 2 - 1, ((x + w) / sw) * 2 - 1
        y0, y1 = (y / sh) * 2 - 1, ((y + h) / sh) * 2 - 1
        if rotation_deg:
            import math as _m

            centre_x, centre_y = x + w / 2.0, y + h / 2.0
            cos_theta = _m.cos(_m.radians(rotation_deg))
            sin_theta = _m.sin(_m.radians(rotation_deg))

            def _rot(px: float, py: float) -> tuple[float, float]:
                dx, dy = px - centre_x, py - centre_y
                rotated_x = centre_x + dx * cos_theta - dy * sin_theta
                rotated_y = centre_y + dx * sin_theta + dy * cos_theta
                return (rotated_x / sw) * 2 - 1, (rotated_y / sh) * 2 - 1

            bottom_left = _rot(x, y)
            bottom_right = _rot(x + w, y)
            top_right = _rot(x + w, y + h)
            top_left = _rot(x, y + h)
        else:
            bottom_left = (x0, y0)
            bottom_right = (x1, y0)
            top_right = (x1, y1)
            top_left = (x0, y1)
        source_u_end = max(0.0, min(1.0, source_u_end))
        if len(tint) == 3:
            r, g, b, a = tint[0], tint[1], tint[2], 1.0
        else:
            r, g, b, a = tint
        arr.use(0)
        self._set_sprite_prog_uniforms(prog, sh)
        verts = _EXT_QUAD_PACK(
            bottom_left[0], bottom_left[1], 0, 1, 0, r, g, b, a,
            bottom_right[0], bottom_right[1], source_u_end, 1, 0, r, g, b, a,
            top_right[0], top_right[1], source_u_end, 0, 0, r, g, b, a,
            bottom_left[0], bottom_left[1], 0, 1, 0, r, g, b, a,
            top_right[0], top_right[1], source_u_end, 0, 0, r, g, b, a,
            top_left[0], top_left[1], 0, 0, 0, r, g, b, a,
        )
        vbo, vao = self._ext_quad_buffers()
        vbo.write(verts)
        vao.render(moderngl.TRIANGLES)

    def _draw_direct_clipped_x(
        self,
        name: str,
        *,
        x: float,
        y: float,
        full_width: float,
        height: float,
        visible_fraction: float,
        tint: tuple = (1.0, 1.0, 1.0, 1.0),
        rotation_deg: float = 0.0,
    ) -> None:
        """Reveal a direct texture left-to-right without squashing its UVs."""
        visible_fraction = max(0.0, min(1.0, visible_fraction))
        visible_width = full_width * visible_fraction
        if visible_width <= 0:
            return
        self._draw_direct(
            name,
            x,
            y,
            visible_width,
            height,
            tint=tint,
            source_u_end=visible_fraction,
            rotation_deg=rotation_deg,
        )

    _GRADE_COLOURS: dict[str, tuple[int, int, int]] = {
        "SS": (240, 220, 120),   # gold
        "S":  (240, 220, 120),   # gold
        "A":  (110, 220, 130),   # green
        "B":  (110, 180, 220),   # blue
        "C":  (200, 130, 220),   # purple
        "D":  (220, 110, 110),   # red
    }

    def _draw_ur_summary(self, scene: SceneState) -> None:
        """Draw the shared standalone Argon UR readout (never average text)."""
        if not self._is_argon_default():
            return
        from osu_mania_renderer_v2.wiki_elements.hud import _argon_unstable_rate

        _argon_unstable_rate(self._shared_frame_context(scene))

    def _results_avatar_texture(self) -> "moderngl.Texture | None":
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

    def _draw_results_overlay(self, scene: SceneState, ctx=None) -> None:
        """Post-game results card — lazer-Argon faithful. Every NUMBER (score,
        accuracy, max combo, the six judgment counts, unstable rate, per-column
        UR, pp) is composed from the bundled argon-counter glyph font (lazer's
        ArgonCounterTextComponent — live digits over the dim wireframe backing)
        via `_argon_number`. Only *labels* (ACCURACY / MAX COMBO / the judgment
        band names / the signed avg-offset caption, which the argon font has no
        +/- glyph for) stay PIL text.

        Layout is authored in lazer's 1080p design space and scaled by
        `A = height/1080`, mirroring `_draw_argon_hud`. A drawing `ctx`
        (FrameContext) is required for the argon glyph path; the wiki
        `results_overlay` element passes the live gameplay ctx, and the legacy
        monolithic `draw()` path (ctx=None) constructs an equivalent one so the
        Argon look is identical regardless of render path.
        """
        rc = self.rc
        a = max(0.0, min(1.0, scene.results_opacity))
        # Obtain / construct the FrameContext used for the argon glyph font.
        if ctx is None:
            try:
                from osu_mania_renderer_v2.wiki_elements.context import (
                    FrameContext,
                )
                ctx = FrameContext(
                    fr=self, skin=None, gl=self.rc.ctx, fbo=self.rc.fbo,
                    width=self.rc.width, height=self.rc.height,
                    key_count=self.rc.key_count,
                )
                ctx.scene = scene
            except Exception:  # noqa: BLE001
                ctx = None
        use_argon = ctx is not None and ctx.has_argon_font()
        # Lazily import the shared argon-number primitive (avoids a module-load
        # cycle with wiki_elements at import time).
        _argon_number = None
        _argon_overlap = 0
        if use_argon:
            try:
                from osu_mania_renderer_v2.wiki_elements.hud import (
                    _argon_number as _an, ARGON_OVERLAP as _ao,
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

    def _draw_flashlight_pass(self) -> None:
        # Approximation v1: darken overlay (full flashlight ping-pong fbo is
        # an optimization for later). Draws a semi-transparent black quad over
        # the whole frame, simulating a darkened scene.
        rc = self.rc
        h = rc.height
        w = rc.width
        self._draw_sprite("bg_vignette", 0, 0, w, h, (0, 0, 0, 0.65))

    STAGE_LIGHT_DURATION_MS = 200

    def _draw_stage_lights(self, scene: SceneState) -> None:
        """Bright vertical strip in any column that has just had a key
        press, fading over ~200 ms. Lazer-style "stage light" — gives the
        playfield a much more reactive feel during streams.

        Tinted by `ColourLight{N}` from the skin's [Mania] block when
        the author set it; otherwise plain white for the legacy fallback.
        If the skin authored `mania-stage-light.png`
        (animated or static), that sprite is used; else the flat
        `column_bg` rectangle is drawn at full column height (the
        renderer's pre-skinning default look)."""
        # Argon press illumination is owned by the shared key-area renderer.
        # Do not layer the old full-column synthetic flash on top of it.
        if self._is_argon_default() or not scene.key_press_age_ms:
            return
        rc = self.rc
        src = self.atlas.global_source("stage_light")
        use_skin_sprite = src in ("beatmap", "user")
        sl_frames = self.atlas.frame_count("stage_light") if use_skin_sprite else 1
        sl_base_idx = self.atlas.index_of("stage_light") if use_skin_sprite else 0
        for c in range(rc.key_count):
            if c >= len(scene.key_press_age_ms):
                break
            age = scene.key_press_age_ms[c]
            if age >= self.STAGE_LIGHT_DURATION_MS:
                continue
            t = age / self.STAGE_LIGHT_DURATION_MS
            alpha = 0.35 * (1.0 - t)
            tint = self._stage_light_tint(c)
            tint_rgba = (tint[0], tint[1], tint[2], alpha)
            if use_skin_sprite:
                # Animated mania-stage-light loops at LightFramePerSecond
                # if the skin set it; else 60fps (spec default). Each
                # column has its own age, so columns at different phases
                # of the animation show different frames.
                if sl_frames > 1:
                    fps = self._stage_light_fps(sl_frames)
                    frame_idx = int(age * fps / 1000.0) % sl_frames
                else:
                    frame_idx = 0
                # Stage-light typically sits in the lower portion of the
                # column from `LightPosition` upward. Approximate: draw
                # from y=0 up to receptor centre + col_w height.
                sl_top = self.receptor_centre_y_gl + self.col_w[c]
                self._draw_sprite_idx(
                    sl_base_idx + frame_idx,
                    self.col_x[c], 0,
                    self.col_w[c], sl_top, tint_rgba,
                )
            else:
                # Backward-compat: flat tinted rectangle, full height.
                self._draw_sprite(
                    "column_bg", self.col_x[c], 0,
                    self.col_w[c], rc.height, tint_rgba,
                )

    def _note_anim_fps(self, frame_count: int) -> float:
        """FPS for tap-note / hold-head / hold-tail animations.

        Per the wiki: notes animate at `[General] AnimationFramerate`
        (no [Mania]-level override). `-1` derives danser-style
        (1000/frame_count ms per frame ⇒ fps == frame_count). Spec hard
        default is 60.
        """
        if self.skin_ini is not None and self.skin_ini.animation_framerate:
            af = self.skin_ini.animation_framerate
            if af > 0:
                return float(af)
            if af == -1 and frame_count > 1:
                return float(frame_count)
        return 60.0

    def _stage_light_fps(self, frame_count: int) -> float:
        """Frames-per-second for the stage-light animation.

        Precedence per the osu! wiki + danser parity rules:
          1. [Mania] LightFramePerSecond (positive) — explicit override
          2. -1 → derive (danser: 1000/frame_count ms per frame ⇒ fps == frame_count)
          3. [General] AnimationFramerate — global default
          4. 60 fps — spec hard default for hit-burst/lighting class
        """
        section = self.mania_section
        if section is not None and section.light_frame_per_second is not None:
            v = section.light_frame_per_second
            if v > 0:
                return float(v)
            if v == -1 and frame_count > 1:
                return float(frame_count)
        if self.skin_ini is not None and self.skin_ini.animation_framerate:
            af = self.skin_ini.animation_framerate
            if af > 0:
                return float(af)
            if af == -1 and frame_count > 1:
                return float(frame_count)
        return 60.0

    def _stage_light_tint(self, col: int) -> tuple[float, float, float]:
        """RGB tint for a column's stage-light press flash.

        Priority: skin's `ColourLight{N}` (1-indexed → 0-indexed
        fallback) → plain white (backward-compat for non-skinned)."""
        section = self.mania_section
        if section is not None:
            rgb = section.colour_light.get(col + 1)
            if rgb is None:
                rgb = section.colour_light.get(col)
            if rgb is not None:
                return rgb[0] / 255, rgb[1] / 255, rgb[2] / 255
        return 1.0, 1.0, 1.0

    def _draw_columns(self, scene: SceneState | None = None) -> None:
        """Per-column lane backgrounds. If the skin authors `Colour{N}` in
        skin.ini, that tint wins; otherwise the renderer's default
        alternating-shade palette is used. Kiai lifts each column's tint
        slightly so the playfield brightens during chorus parts of the
        song (osu!'s kiai highlight)."""
        if scene is not None and self._is_argon_default():
            self._draw_argon_columns(scene)
            return

        rc = self.rc
        w = rc.width
        h = rc.height
        pf_x = self.pf_x
        pf_w = self.pf_w
        col_w = self.col_w_uniform
        kiai_boost = 0.04 if (scene is not None and scene.is_kiai) else 0.0
        section = self.mania_section
        for c in range(rc.key_count):
            # Skin's Colour{N} is 1-indexed; column index is 0-based.
            skin_colour = None
            if section is not None:
                skin_colour = section.colour.get(c + 1)
                if skin_colour is None:
                    # Some skins use 0-indexed Colour entries — accept both.
                    skin_colour = section.colour.get(c)
            if skin_colour is not None:
                sr, sg, sb, sa = skin_colour
                r = sr / 255
                g = sg / 255
                b = sb / 255
                a = sa / 255
            else:
                variant = column_variant(c, rc.key_count)
                if variant == "outer":
                    r, g, b, a = 0.04, 0.04, 0.09, 0.55
                elif variant == "center":
                    r, g, b, a = 0.07, 0.06, 0.12, 0.55
                else:
                    r, g, b, a = 0.05, 0.05, 0.11, 0.45
            self._draw_sprite("column_bg", self.col_x[c], 0,
                              self.col_w[c], h,
                              (r + kiai_boost, g + kiai_boost,
                               b + kiai_boost * 1.5, a))

        # Column dividers + outer borders. Spec: ColumnLineWidth is a
        # csv of (N+1) ints — width per divider in 480-ref pixels — and
        # ColourColumnLine tints all of them. We treat missing skin
        # values as "draw default-thin white outer borders + no inner
        # dividers", matching the renderer's pre-Phase-B look.
        line_widths = section.column_line_width if section else ()
        if section is not None and section.colour_column_line is not None:
            sr, sg, sb, sa = section.colour_column_line
            line_tint = (sr / 255, sg / 255, sb / 255, sa / 255)
        else:
            line_tint = (1.0, 1.0, 1.0, 0.9)

        def _divider_x(idx: int) -> int:
            """X of the idx-th divider. idx 0 = left of col 0,
            idx K = right of col K-1, anything in between = between cols."""
            if idx >= rc.key_count:
                return self.col_x[-1] + self.col_w[-1]
            return self.col_x[idx]

        if line_widths and len(line_widths) >= rc.key_count + 1:
            # Convert from 480-ref pixels to render pixels. osu! pixels
            # scale by target_height / 480 (the 4:3 region's height).
            px_per_ref = h / 480.0
            for c in range(rc.key_count + 1):
                lw_ref = line_widths[c]
                if lw_ref <= 0:
                    continue
                lw = max(1, int(round(lw_ref * px_per_ref)))
                x_centre = _divider_x(c)
                self._draw_sprite("column_bg",
                                  x_centre - lw // 2, 0,
                                  lw, h, line_tint)
        # Skins that don't author ColumnLineWidth get NO dividers. The
        # previous behaviour was to draw a thin white outer border on
        # the left+right edges of the playfield as a pre-skinning visual
        # cue — but every uploaded skin we test ends up looking less
        # polished with those uninvited lines. The user explicitly asked
        # to suppress them ("Lines"). Skins that WANT dividers ship a
        # ColumnLineWidth (even all-zero suppresses dividers explicitly).

    def _draw_notes(self, scene: SceneState) -> None:
        if self._is_argon_default():
            self._draw_argon_notes(scene)
            return

        rc = self.rc
        w = rc.width
        h = rc.height
        pf_x = self.pf_x
        pf_w = self.pf_w
        col_w = self.col_w_uniform
        # Notes are circles in the web replay viewer — square aspect ratio
        # so they render as true circles, not stretched ovals.
        note_h = col_w
        # Note colours — osu!stable default mania palette: outer columns are
        # near-white (the "1" / "4" lanes), inner columns are vivid osu!
        # blue (the "2" / "3" lanes), and the centre column on odd-key maps
        # is a warm gold ("S" lane in K3/K5/K7/K9).
        tints = {
            "outer":  (240 / 255, 240 / 255, 245 / 255, 1.0),
            "inner":  ( 70 / 255, 165 / 255, 255 / 255, 1.0),
            "center": (255 / 255, 210 / 255,  90 / 255, 1.0),
        }
        # Map y_fraction → on-screen Y (GL, Y-up; ffmpeg vflips at encode).
        # y_fraction = 0 → top of playfield (just spawned)
        # y_fraction = 1 → centre of the receptor row (note "hits" here)
        receptor_y = self.receptor_centre_y_gl
        upside_down = self.upside_down

        def to_screen_y(yf: float) -> int:
            # Normal: yf=0 at TOP of screen (high gl_y), yf=1 at receptor.
            # Upside-down: yf=0 at BOTTOM (gl_y=0), yf=1 at receptor (now
            # near the TOP of the screen). Notes scroll UP toward the
            # receptor instead of down.
            if upside_down:
                return int(yf * receptor_y)
            return int(receptor_y + (1.0 - yf) * (h - receptor_y))
        # If the skin author provided per-column note sprites, draw them
        # untinted (the skin's own colours stay intact). Otherwise use
        # the renderer's signature tinted-circle look.
        use_skin_notes = self.atlas.has_skin_notes()
        # Per-note animation phase. Spec says each note's animation
        # starts from frame 0 at the note's spawn moment (≈ time_ms
        # minus approach_ms). Using note.time_ms directly as the
        # phase reference is visually equivalent for looped animations
        # and saves the approach_ms lookup — the modulo just shifts
        # which frame is on screen at any given world time.
        world_ms = scene.t_ms

        def _animated_idx(kind: str, col: int, note_time_ms: int = 0) -> int:
            """Resolve a per-column slot to its current frame layer
            index. Each note gets its own animation phase keyed off
            `note_time_ms`; pass 0 for world-time-synced animation
            (used by non-note slots that share a single phase)."""
            base = self.atlas.column_slot_index(kind, col)
            n_frames = self.atlas.column_frame_count(kind, col)
            if n_frames <= 1:
                return base
            fps = self._note_anim_fps(n_frames)
            elapsed_ms = world_ms - note_time_ms
            return base + int(elapsed_ms * fps / 1000.0) % n_frames

        for n in scene.visible_notes:
            x0 = self.col_x[n.column]
            cw = self.col_w[n.column]
            tint = tints[column_variant(n.column, rc.key_count)]
            col_has_skin = use_skin_notes and self.atlas.has_skin_note(n.column)
            # Note height: native aspect of the skin's tap sprite when
            # available, else square (cw × cw). Per ppy/osu
            # LegacyNotePiece.cs — both axes divide by texture.width, so
            # height = cw × (tex.h / tex.w) == cw / aspect. Anchoring is
            # kept centred (centre at to_screen_y(yf)) so the gameplay
            # "hit point" stays where users expect — lazer's bottom-
            # anchor moves the visual landing point up by note_h/2,
            # which our renderer's centred convention doesn't follow.
            if col_has_skin:
                note_asp = self.atlas.column_aspect("note_tap", n.column)
                local_note_h = (
                    max(1, int(cw / note_asp)) if note_asp > 0 else cw
                )
            else:
                local_note_h = cw  # circle fallback: square
            # Hold head/tail get their own aspect since the head/tail
            # sprite may differ from the tap sprite.
            if col_has_skin:
                head_asp = self.atlas.column_aspect("note_hold_head", n.column)
                head_h = max(1, int(cw / head_asp)) if head_asp > 0 else cw
                tail_asp = self.atlas.column_aspect("note_hold_tail", n.column)
                tail_h = max(1, int(cw / tail_asp)) if tail_asp > 0 else cw
            else:
                head_h = cw
                tail_h = cw
            # Holds need a stricter check — only use skin hold sprites
            # when the author shipped head + body + tail together.
            # Mixing skin parts with bundled fallback rectangles
            # produces a Frankenstein slider (rectangle body, square
            # caps). For partial skins, fall back to the capsule +
            # circle look.
            col_has_skin_hold = (
                col_has_skin and self.atlas.has_skin_hold(n.column)
            )
            if n.is_hold:
                y_head = to_screen_y(n.head_y_fraction)
                y_tail = to_screen_y(n.tail_y_fraction)
                body_top = min(y_head, y_tail)
                body_h = abs(y_head - y_tail)
                if col_has_skin_hold:
                    body_idx = _animated_idx("note_hold_body", n.column, n.time_ms)
                    head_idx = _animated_idx("note_hold_head", n.column, n.time_ms)
                    tail_idx = _animated_idx("note_hold_tail", n.column, n.time_ms)
                    # NoteBodyStyle per ppy/osu LegacyNoteBodyStyle.cs:
                    #   0  Stretch              — single stretch of L sprite
                    #   2  RepeatTop            — tile vertically; top cap static
                    #   3  RepeatBottom         — tile vertically; bottom cap static
                    #   4  RepeatTopAndBottom   — tile vertically; both caps static
                    # Value `1` exists in the osu! wiki but isn't in lazer's
                    # enum source — we treat it as tile to match our
                    # historical behaviour for cascade-style L sprites.
                    # The cap-static-vs-scroll distinction (mode 2 vs 3 vs
                    # 4) requires a 9-slice draw that the current sprite
                    # primitive doesn't support; all three tile modes use
                    # a simple repeat for now.
                    body_style = (
                        self.mania_section.note_body_style
                        if self.mania_section is not None
                        and self.mania_section.note_body_style is not None
                        else 0  # default per lazer
                    )
                    if body_style != 0:
                        # Cascade body: repeat the L sprite vertically at
                        # its natural aspect. Tile height preserves the
                        # source sprite's aspect so the visual element
                        # doesn't stretch.
                        body_aspect = self.atlas.column_aspect(
                            "note_hold_body", n.column,
                        )
                        tile_h = (
                            max(1, int(round(cw / body_aspect)))
                            if body_aspect > 0 else cw
                        )
                        seg_y = body_top
                        while seg_y < body_top + body_h:
                            seg_h = min(tile_h, body_top + body_h - seg_y)
                            self._draw_sprite_idx(body_idx, x0, seg_y,
                                                  cw, seg_h, (1, 1, 1, 1))
                            seg_y += tile_h
                    else:
                        # Styles 0 and 2: stretch L sprite once across the
                        # entire body length. Single draw call.
                        self._draw_sprite_idx(body_idx, x0, body_top,
                                              cw, body_h, (1, 1, 1, 1))
                    # Head sits at the head position (top of the hold while
                    # falling, sticks to the judgement line during a hold).
                    # Always drawn AFTER the body so it visually caps the
                    # top end and isn't covered by a stretched body.
                    self._draw_sprite_idx(head_idx, x0,
                                          y_head - head_h // 2,
                                          cw, head_h, (1, 1, 1, 1))
                    self._draw_sprite_idx(tail_idx, x0,
                                          y_tail - tail_h // 2,
                                          cw, tail_h, (1, 1, 1, 1))
                else:
                    pad = cw // 6
                    self._draw_sprite("column_bg", x0 + pad, body_top,
                                      cw - 2 * pad, body_h, tint)
                    # Head stays visible at receptor during hold (mania
                    # convention — the head sticks to the judgement line).
                    self._draw_sprite("note_circle", x0,
                                      y_head - local_note_h // 2,
                                      cw, local_note_h, tint)
                    self._draw_sprite("note_circle", x0,
                                      y_tail - local_note_h // 2,
                                      cw, local_note_h, tint)
            else:
                y = to_screen_y(n.y_fraction)
                if col_has_skin:
                    tap_idx = _animated_idx("note_tap", n.column, n.time_ms)
                    trail_step = max(4, local_note_h // 4)
                    for k in (2, 1):
                        ghost_y = y + k * trail_step
                        ghost_alpha = 0.20 / k
                        self._draw_sprite_idx(
                            tap_idx, x0, ghost_y - local_note_h // 2,
                            cw, local_note_h, (1, 1, 1, ghost_alpha),
                        )
                    self._draw_sprite_idx(tap_idx, x0,
                                          y - local_note_h // 2,
                                          cw, local_note_h, (1, 1, 1, 1))
                else:
                    trail_step = max(4, local_note_h // 4)
                    for k in (2, 1):
                        ghost_y = y + k * trail_step
                        ghost_alpha = 0.20 / k
                        ghost_tint = (tint[0], tint[1], tint[2], ghost_alpha)
                        self._draw_sprite(
                            "note_circle", x0, ghost_y - local_note_h // 2,
                            cw, local_note_h, ghost_tint,
                        )
                    self._draw_sprite("note_circle", x0,
                                      y - local_note_h // 2,
                                      cw, local_note_h, tint)

    # osu!mania judgment colours — used for the receptor hit-light flash,
    # the UR histogram bins, and the floating hit-error popup.
    _JUDGMENT_LIGHT: dict[str, tuple[int, int, int]] = {
        "geki": (150, 215, 255),   # 320 light blue
        "300":  ( 80, 150, 240),   # 300 blue
        "katu": (100, 220, 130),   # 200 green
        "100":  (240, 220,  90),   # 100 yellow
        "50":   (240, 160,  80),   # 50  orange
    }
    HIT_LIGHT_DURATION_MS = 320

    def _draw_receptors(self, scene: SceneState) -> None:
        if self._is_argon_default():
            self._draw_argon_receptors(scene)
            return

        rc = self.rc
        h = rc.height
        centre_y = self.receptor_centre_y_gl
        for c in range(rc.key_count):
            x0 = self.col_x[c]
            cw = self.col_w[c]
            held = scene.keys_held[c]
            kind = "receptor_on" if held else "receptor_off"
            slot_idx = self.atlas.column_slot_index(kind, c)

            # LegacyKeyArea stretches the key image across the column but
            # keeps its native DESIGN height (Texture.DisplaySize), scaled
            # from lazer's 768-unit stage. This is important for padded
            # receptor canvases: deriving height from the whole-canvas
            # aspect stretches otherwise circular visible artwork. Atlas
            # native sizes already divide @2x assets by ScaleAdjust.
            _native_w, native_h = self.atlas.column_native_size(kind, c)
            tex_scale = h / 768.0
            if native_h > 0:
                rec_h = max(1, int(round(native_h * tex_scale)))
            else:
                asp = self.atlas.column_aspect(kind, c)
                rec_h = (
                    max(1, int(cw / asp))
                    if asp > 0
                    else int(cw * RECEPTOR_HEIGHT_REL_COL)
                )
            # LegacyKeyArea is anchored to the stage edge: BottomCentre for
            # downscroll, TopCentre for upscroll. Pressing only swaps
            # KeyImage -> KeyImageD; it never resizes the authored key.
            rec_y = h - rec_h if self.upside_down else 0
            self._draw_sprite_idx(
                slot_idx, x0, rec_y, cw, rec_h, (1, 1, 1, 1),
            )
            self._draw_custom_legacy_lighting(
                scene, c=c, x0=x0, cw=cw, centre_y=centre_y, held=held,
            )

    def _legacy_lighting_rect(
        self, slot: str, *, c: int, x0: int, cw: int, centre_y: int,
    ) -> tuple[int, int, int, int] | None:
        """Native-aspect rect for a custom legacy LightingN/L sprite."""
        native_w, native_h = self.atlas.global_native_size(slot)
        if native_w <= 0 or native_h <= 0:
            return None
        kind = "n" if slot == "lighting_n" else "l"
        px_per_ref = self.rc.height / 480.0
        effective_column_width = cw / px_per_ref
        scale = legacy_lighting_scale(
            self.mania_section,
            c,
            kind,
            effective_column_width=effective_column_width,
            legacy_version=(
                self.skin_ini.legacy_version
                if self.skin_ini is not None
                else None
            ),
        )
        tex_scale = self.rc.height / 768.0
        light_w = max(1, int(round(native_w * tex_scale * scale)))
        light_h = max(1, int(round(native_h * tex_scale * scale)))
        return (
            x0 + (cw - light_w) // 2,
            centre_y - light_h // 2,
            light_w,
            light_h,
        )

    def _draw_custom_legacy_lighting(
        self,
        scene: SceneState,
        *,
        c: int,
        x0: int,
        cw: int,
        centre_y: int,
        held: bool,
    ) -> None:
        """Draw only skin-authored custom legacy hold / hit lighting."""
        if held and self.atlas.global_source("lighting_l") in ("beatmap", "user"):
            rect = self._legacy_lighting_rect(
                "lighting_l", c=c, x0=x0, cw=cw, centre_y=centre_y,
            )
            if rect is not None:
                base = self.atlas.index_of("lighting_l")
                frames = self.atlas.frame_count("lighting_l")
                frame = 0
                if frames > 1:
                    fps = self._stage_light_fps(frames)
                    age = scene.key_press_age_ms[c] if c < len(scene.key_press_age_ms) else 0
                    frame = int(age * fps / 1000.0) % frames
                press_age = (
                    scene.key_press_age_ms[c]
                    if c < len(scene.key_press_age_ms)
                    else LEGACY_HIT_EXPLOSION_FADE_IN_MS
                )
                alpha = min(1.0, max(0.0, press_age / LEGACY_HIT_EXPLOSION_FADE_IN_MS))
                self._draw_additive_sprite_idx(
                    base + frame, *rect, (1.0, 1.0, 1.0, alpha),
                )

        if c >= len(scene.hit_light_age_ms):
            return
        age = scene.hit_light_age_ms[c]
        judgment = (
            scene.hit_light_judgment[c]
            if c < len(scene.hit_light_judgment)
            else ""
        )
        alpha = legacy_hit_explosion_alpha(age)
        if alpha <= 0 or judgment not in self._JUDGMENT_LIGHT:
            return
        if self.atlas.global_source("lighting_n") not in ("beatmap", "user"):
            # A custom skin with no authored LightingN gets no fabricated R3D
            # note-circle flash. Explicit transparent assets remain authoritative.
            return
        rect = self._legacy_lighting_rect(
            "lighting_n", c=c, x0=x0, cw=cw, centre_y=centre_y,
        )
        if rect is None:
            return
        base = self.atlas.index_of("lighting_n")
        frames = self.atlas.frame_count("lighting_n")
        frame = legacy_hit_explosion_frame(age, frames)
        self._draw_additive_sprite_idx(
            base + frame, *rect, (1.0, 1.0, 1.0, alpha),
        )

    def _draw_sprite(
        self, name: str, x: int, y: int, w: int, h: int, tint: tuple,
    ) -> None:
        """Append one quad to the instance buffer by named atlas slot."""
        self._draw_sprite_idx(self.atlas.index_of(name), x, y, w, h, tint)

    def _draw_additive_sprite_idx(
        self, atlas_idx: int, x: int, y: int, w: int, h: int, tint: tuple,
    ) -> None:
        """Draw one atlas sprite additively without leaking blend state."""
        if w <= 0 or h <= 0:
            return
        # Queued instances inherit blend state at flush time. Flush the normal
        # batch first, then flush this sprite while additive blending is active.
        self._flush_sprite_batch()
        ctx = self.rc.ctx
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        try:
            self._draw_sprite_idx(atlas_idx, x, y, w, h, tint)
            self._flush_sprite_batch()
        finally:
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

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
        Called by BOTH the monolithic draw() and the wiki notes element so
        the effect is identical regardless of draw path.

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
