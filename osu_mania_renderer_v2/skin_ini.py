"""Lightweight parser for osu!'s `skin.ini`.

osu!'s skin.ini is INI-like but with quirks (case-insensitive section
names, key=value with leading whitespace, comments via `//`, multiple
`[Mania]` sections each with their own `Keys: N`).

This parser reads everything the mania renderer's atlas/layout phases
need to faithfully reproduce a skin author's intent. Anything we don't
recognise is ignored. Missing fields fall back to spec defaults so a
sparse skin.ini renders without crashing.

Coverage (Phase A — the keys that make a skin "actually take effect"):

  [Colours]
    Combo1..N                 → per-combo note colours

  [Mania] (per `Keys: N` block):
    Colour1..N                → lane background tint (1-indexed)
    ColourLight1..N           → stage-light tint when column N pressed
    ColourColumnLine          → column divider colour
    ColourBarline             → measure-bar colour
    ColourHold                → combo-counter tint during hold
    ColourBreak               → combo-counter tint on combo break
    NoteImage{N}              → per-column tap note path override
    NoteImage{N}H             → per-column hold-head path
    NoteImage{N}L             → per-column hold-body path
    NoteImage{N}T             → per-column hold-tail path
    KeyImage{N}               → per-column receptor idle path
    KeyImage{N}D              → per-column receptor pressed path
    StageLeft / StageRight    → stage-frame path overrides
    StageBottom / StageHint
    StageLight
    Hit0 / Hit50 / Hit100     → judgement-popup path overrides
    Hit200 / Hit300 / Hit300g

Both 0-indexed and 1-indexed NoteImage{N} / KeyImage{N} are accepted
(real-world skins disagree with the spec text); 0-indexed wins when
both are present (matches osu-stable behaviour).

`ColumnColour` from the previous parser was non-standard — dropped.

Note: layout/position keys (ColumnStart, ColumnWidth, HitPosition, …)
are not in this parser yet; Phase B will add them when we wire dynamic
playfield geometry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ───── Default skin.ini values ─────
#
# osu-stable's hard-coded fallbacks. Used when a skin.ini omits a key
# OR when a `Keys: N` block doesn't exist for the chart's keycount.
DEFAULT_COLOUR              = (0, 0, 0, 255)
DEFAULT_COLOUR_LIGHT        = (55, 255, 255)
DEFAULT_COLOUR_COLUMN_LINE  = (255, 255, 255, 255)
DEFAULT_COLOUR_BARLINE      = (255, 255, 255, 255)
DEFAULT_COLOUR_HOLD         = (255, 191, 51, 255)
DEFAULT_COLOUR_BREAK        = (255, 0, 0)


# ───── Output dataclasses ─────


@dataclass(frozen=True)
class ManiaSection:
    """One `[Mania]` block, scoped to a specific keycount."""
    keys: int

    # Per-column RGB(a) tints. Lookup by 0-indexed column; missing
    # entries fall back to spec defaults at consumer side.
    colour:        dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    colour_light:  dict[int, tuple[int, int, int]]      = field(default_factory=dict)

    # Block-global tints.
    colour_column_line: tuple[int, int, int, int] | None = None
    colour_barline:     tuple[int, int, int, int] | None = None
    colour_hold:        tuple[int, int, int, int] | None = None
    colour_break:       tuple[int, int, int]      | None = None

    # Per-column sprite overrides. Each maps column-index → relative path
    # (relative to the skin dir). `part` is "head"/"body"/"tail" for the
    # note variants and "down" for pressed-key.
    note_image:     dict[int, str] = field(default_factory=dict)   # tap
    note_image_h:   dict[int, str] = field(default_factory=dict)   # head
    note_image_l:   dict[int, str] = field(default_factory=dict)   # long body
    note_image_t:   dict[int, str] = field(default_factory=dict)   # tail
    key_image:      dict[int, str] = field(default_factory=dict)   # idle
    key_image_d:    dict[int, str] = field(default_factory=dict)   # pressed

    # Stage-frame asset overrides. None ⇒ use default filename.
    stage_left:   str | None = None
    stage_right:  str | None = None
    stage_bottom: str | None = None
    stage_hint:   str | None = None
    stage_light:  str | None = None

    # Judgement-popup overrides.
    hit_0:     str | None = None
    hit_50:    str | None = None
    hit_100:   str | None = None
    hit_200:   str | None = None
    hit_300:   str | None = None
    hit_300g:  str | None = None

    # ───── Phase B: playfield geometry (osu! 640×480 reference) ─────
    #
    # All position/width values are in the osu! 480-ref pixel system.
    # Consumers scale these to the render resolution via the standard
    # `target_h / 480.0` factor (or `target_w / 512.0` for X positions
    # — peppy's reference width is 512 not 640 once you subtract the
    # 64-px side margins, but the convention is 640-px X-ref. We track
    # the raw skin.ini value; conversion lives in the consumer).
    column_start:      int | None = None      # X of column 1 left edge
    column_right:      int | None = None      # right reserve
    column_width:      tuple[int, ...] = ()   # per-column widths (N entries)
    column_spacing:    tuple[int, ...] = ()   # gaps between cols (N-1 entries)
    column_line_width: tuple[int, ...] = ()   # divider thicknesses (N+1 entries)
    barline_height:    float | None = None    # measure-bar thickness

    hit_position:     int | None = None       # Y of judgement line
    light_position:   int | None = None       # Y of StageLight bottom
    score_position:   int | None = None       # Y of hitburst popups
    combo_position:   int | None = None       # Y of combo counter

    # Per-keymode behaviour flags.
    special_style:    int | None = None       # 0 / 1 / 2
    keys_under_notes: bool | None = None      # default 0
    upside_down:      bool | None = None      # default 0

    # Animation framerate for the stage-light per-press loop.
    # `None` ⇒ fall back to [General] AnimationFramerate ⇒ default 60.
    light_frame_per_second: int | None = None

    # Hold body draw style. osu!mania defines three modes:
    #   0 = stretch the head sprite (mania-noteNH.png) down the entire body
    #   1 = cascade — tile the L sprite vertically at its natural aspect
    #       (default when key is absent in skin.ini)
    #   2 = stretch the L sprite (mania-noteNL.png) over the entire body
    # Most "modern" skins (Night05, FNF, the bundled default) ship with
    # `NoteBodyStyle: 0` because their L sprite is a tall solid bar
    # designed to be stretched once; cascading it would produce visible
    # seam artefacts. `None` ⇒ caller picks the default (= 1, per peppy).
    note_body_style: int | None = None


@dataclass(frozen=True)
class SkinIni:
    combo_colours: tuple[tuple[int, int, int], ...] = ()
    mania:         tuple[ManiaSection, ...]         = field(default_factory=tuple)

    # [General] AnimationFramerate — global default fps for all
    # animated sprites that don't have a more-specific fps. `None`
    # ⇒ use the spec default (60). `-1` ⇒ derive from frame count
    # (`1000 / frame_count` ms per frame), matching danser.
    animation_framerate: int | None = None

    # [Fonts] overlaps — pixels each glyph is pulled left of the previous
    # one when composing a number. osu! defaults: ScoreOverlap 0,
    # ComboOverlap 0, HitCircleOverlap -2. The mania HUD uses the score
    # font for score/accuracy and the combo font for the combo counter.
    score_overlap: int = 0
    combo_overlap: int = 0
    # [Fonts] prefixes — the score/combo number sprites are `<prefix>-N.png`.
    # lazer: ScorePrefix ?? "score", ComboPrefix ?? "score". A skin can ship a
    # separate combo font (e.g. Night05's `combo-N.png`).
    score_prefix: str = "score"
    combo_prefix: str = "score"

    def mania_for_keycount(self, keys: int) -> ManiaSection | None:
        for m in self.mania:
            if m.keys == keys:
                return m
        return None


# ───── Parser ─────


# osu! skin.ini keys we recognise inside a [Mania] block. Names are
# matched case-insensitively; values keep their original case (paths
# are case-sensitive on Linux even though osu! is case-insensitive on
# Windows — we leave string normalisation to the consumer).
_NOTE_IMAGE_RE = re.compile(r"^NoteImage(\d+)(H|L|T)?$", re.IGNORECASE)
_KEY_IMAGE_RE  = re.compile(r"^KeyImage(\d+)(D)?$",      re.IGNORECASE)
_COLOUR_N_RE   = re.compile(r"^Colour(\d+)$",            re.IGNORECASE)
_COLOUR_LIGHT_RE = re.compile(r"^ColourLight(\d+)$",     re.IGNORECASE)
_HIT_RE        = re.compile(r"^Hit(0|50|100|200|300|300g)$", re.IGNORECASE)


def parse_skin_ini(skin_dir: Path) -> SkinIni:
    """Read `<skin_dir>/skin.ini` if it exists; return empty SkinIni
    otherwise."""
    path = skin_dir / "skin.ini"
    if not path.is_file():
        return SkinIni()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return SkinIni()

    combo_colours: list[tuple[int, int, int]] = []
    mania_blocks:  list[_ManiaBuilder]         = []
    animation_framerate: int | None = None
    score_overlap: int = 0
    combo_overlap: int = 0
    score_prefix: str = "score"
    combo_prefix: str = "score"      # lazer: ComboPrefix ?? "score"
    current_section: str | None = None
    current_mania: _ManiaBuilder | None = None

    for raw_line in text.splitlines():
        # Strip `//` comments and surrounding whitespace.
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue

        sect = re.match(r"^\[(.+?)\]$", line)
        if sect:
            current_section = sect.group(1).strip().lower()
            if current_section == "mania":
                # New [Mania] block — emit prior one (if any) and start fresh.
                current_mania = _ManiaBuilder()
                mania_blocks.append(current_mania)
            else:
                current_mania = None
            continue

        # osu! skin.ini uses `:` as the separator (not `=` like a normal
        # INI file). Some skins also use `=` so accept both — whichever
        # appears first wins.
        sep_colon = line.find(":")
        sep_equal = line.find("=")
        if sep_colon < 0 and sep_equal < 0:
            continue
        if sep_colon < 0:
            sep = sep_equal
        elif sep_equal < 0:
            sep = sep_colon
        else:
            sep = min(sep_colon, sep_equal)
        key = line[:sep].strip()
        value = line[sep + 1:].strip()

        if current_section == "colours":
            m = re.fullmatch(r"Combo(\d+)", key, re.IGNORECASE)
            if m:
                rgb = _parse_rgb(value)
                if rgb is not None:
                    combo_colours.append(rgb)
        elif current_section == "general":
            if key.strip().lower() == "animationframerate":
                animation_framerate = _parse_int(value)
        elif current_section == "fonts":
            k = key.strip().lower()
            if k == "scoreoverlap":
                v = _parse_int(value)
                if v is not None:
                    score_overlap = v
            elif k == "combooverlap":
                v = _parse_int(value)
                if v is not None:
                    combo_overlap = v
            elif k == "scoreprefix":
                if value.strip():
                    score_prefix = value.strip().replace("\\", "/")
            elif k == "comboprefix":
                if value.strip():
                    combo_prefix = value.strip().replace("\\", "/")
        elif current_section == "mania" and current_mania is not None:
            current_mania.consume(key, value)

    # Convert builders → ManiaSection. Blocks with no `Keys:` line
    # are dropped (spec requires `Keys` to be the first key).
    mania = tuple(
        b.finalize() for b in mania_blocks
        if b.keys is not None
    )
    return SkinIni(
        combo_colours=tuple(combo_colours),
        mania=mania,
        animation_framerate=animation_framerate,
        score_overlap=score_overlap,
        combo_overlap=combo_overlap,
        score_prefix=score_prefix,
        combo_prefix=combo_prefix,
    )


# ───── Internal builder (mutable while parsing) ─────


class _ManiaBuilder:
    """Mutable accumulator for one `[Mania]` block. `finalize()`
    snapshots into the frozen `ManiaSection`."""

    def __init__(self) -> None:
        self.keys: int | None = None

        self.colour:       dict[int, tuple[int, int, int, int]] = {}
        self.colour_light: dict[int, tuple[int, int, int]]      = {}

        self.colour_column_line: tuple[int, int, int, int] | None = None
        self.colour_barline:     tuple[int, int, int, int] | None = None
        self.colour_hold:        tuple[int, int, int, int] | None = None
        self.colour_break:       tuple[int, int, int]      | None = None

        self.note_image:   dict[int, str] = {}
        self.note_image_h: dict[int, str] = {}
        self.note_image_l: dict[int, str] = {}
        self.note_image_t: dict[int, str] = {}
        self.key_image:    dict[int, str] = {}
        self.key_image_d:  dict[int, str] = {}

        self.stage_left:   str | None = None
        self.stage_right:  str | None = None
        self.stage_bottom: str | None = None
        self.stage_hint:   str | None = None
        self.stage_light:  str | None = None

        self.hit_0:     str | None = None
        self.hit_50:    str | None = None
        self.hit_100:   str | None = None
        self.hit_200:   str | None = None
        self.hit_300:   str | None = None
        self.hit_300g:  str | None = None

        # Phase B — playfield geometry.
        self.column_start:      int | None = None
        self.column_right:      int | None = None
        self.column_width:      tuple[int, ...] = ()
        self.column_spacing:    tuple[int, ...] = ()
        self.column_line_width: tuple[int, ...] = ()
        self.barline_height:    float | None = None
        self.hit_position:      int | None = None
        self.light_position:    int | None = None
        self.score_position:    int | None = None
        self.combo_position:    int | None = None
        self.special_style:     int | None = None
        self.keys_under_notes:  bool | None = None
        self.upside_down:       bool | None = None
        self.light_frame_per_second: int | None = None
        self.note_body_style:    int | None = None

    def consume(self, key: str, value: str) -> None:
        lk = key.lower()
        if lk == "keys":
            try:
                self.keys = int(value)
            except ValueError:
                pass
            return

        # Per-column sprite overrides.
        m = _NOTE_IMAGE_RE.match(key)
        if m:
            col = _normalize_column_index(int(m.group(1)))
            part = (m.group(2) or "").upper()
            target = {
                "":  self.note_image,
                "H": self.note_image_h,
                "L": self.note_image_l,
                "T": self.note_image_t,
            }.get(part)
            if target is not None:
                target[col] = value
            return

        m = _KEY_IMAGE_RE.match(key)
        if m:
            col = _normalize_column_index(int(m.group(1)))
            part = (m.group(2) or "").upper()
            target = self.key_image_d if part == "D" else self.key_image
            target[col] = value
            return

        # Per-column colours.
        m = _COLOUR_N_RE.match(key)
        if m:
            col = _normalize_column_index(int(m.group(1)))
            rgba = _parse_rgba(value)
            if rgba is not None:
                self.colour[col] = rgba
            return

        m = _COLOUR_LIGHT_RE.match(key)
        if m:
            col = _normalize_column_index(int(m.group(1)))
            rgb = _parse_rgb(value)
            if rgb is not None:
                self.colour_light[col] = rgb
            return

        # Block-global colours.
        if lk == "colourcolumnline":
            self.colour_column_line = _parse_rgba(value)
            return
        if lk == "colourbarline":
            self.colour_barline = _parse_rgba(value)
            return
        if lk == "colourhold":
            self.colour_hold = _parse_rgba(value)
            return
        if lk == "colourbreak":
            self.colour_break = _parse_rgb(value)
            return

        # Judgement popups.
        m = _HIT_RE.match(key)
        if m:
            tier = m.group(1).lower()
            setattr(self, f"hit_{tier}", value)
            return

        # Stage frames.
        if lk == "stageleft":
            self.stage_left = value
            return
        if lk == "stageright":
            self.stage_right = value
            return
        if lk == "stagebottom":
            self.stage_bottom = value
            return
        if lk == "stagehint":
            self.stage_hint = value
            return
        if lk == "stagelight":
            self.stage_light = value
            return

        # Playfield geometry (osu! 480-ref pixels).
        if lk == "columnstart":
            self.column_start = _parse_int(value)
            return
        if lk == "columnright":
            self.column_right = _parse_int(value)
            return
        if lk == "columnwidth":
            self.column_width = _parse_csv_ints(value)
            return
        if lk == "columnspacing":
            self.column_spacing = _parse_csv_ints(value)
            return
        if lk == "columnlinewidth":
            self.column_line_width = _parse_csv_ints(value)
            return
        if lk == "barlineheight":
            self.barline_height = _parse_float(value)
            return
        if lk == "hitposition":
            self.hit_position = _parse_int(value)
            return
        if lk == "lightposition":
            self.light_position = _parse_int(value)
            return
        if lk == "scoreposition":
            self.score_position = _parse_int(value)
            return
        if lk == "comboposition":
            self.combo_position = _parse_int(value)
            return
        if lk == "specialstyle":
            self.special_style = _parse_int(value)
            return
        if lk == "keysundernotes":
            self.keys_under_notes = _parse_bool(value)
            return
        if lk == "upsidedown":
            self.upside_down = _parse_bool(value)
            return
        if lk == "lightframepersecond":
            self.light_frame_per_second = _parse_int(value)
            return
        if lk == "notebodystyle":
            self.note_body_style = _parse_int(value)
            return

    def finalize(self) -> ManiaSection:
        return ManiaSection(
            keys=self.keys or 0,
            colour=dict(self.colour),
            colour_light=dict(self.colour_light),
            colour_column_line=self.colour_column_line,
            colour_barline=self.colour_barline,
            colour_hold=self.colour_hold,
            colour_break=self.colour_break,
            note_image=dict(self.note_image),
            note_image_h=dict(self.note_image_h),
            note_image_l=dict(self.note_image_l),
            note_image_t=dict(self.note_image_t),
            key_image=dict(self.key_image),
            key_image_d=dict(self.key_image_d),
            stage_left=self.stage_left,
            stage_right=self.stage_right,
            stage_bottom=self.stage_bottom,
            stage_hint=self.stage_hint,
            stage_light=self.stage_light,
            hit_0=self.hit_0, hit_50=self.hit_50, hit_100=self.hit_100,
            hit_200=self.hit_200, hit_300=self.hit_300, hit_300g=self.hit_300g,
            column_start=self.column_start,
            column_right=self.column_right,
            column_width=self.column_width,
            column_spacing=self.column_spacing,
            column_line_width=self.column_line_width,
            barline_height=self.barline_height,
            hit_position=self.hit_position,
            light_position=self.light_position,
            score_position=self.score_position,
            combo_position=self.combo_position,
            special_style=self.special_style,
            keys_under_notes=self.keys_under_notes,
            upside_down=self.upside_down,
            light_frame_per_second=self.light_frame_per_second,
            note_body_style=self.note_body_style,
        )


# ───── Helpers ─────


def _normalize_column_index(raw: int) -> int:
    """osu!'s `Colour#` is documented as 1-indexed, but `KeyImage#` and
    `NoteImage#` in real skins are commonly 0-indexed. osu-stable
    accepts both. We always store 0-indexed internally.

    Per spec text: if a skin has both `Colour0` and `Colour1` we treat
    them as the same column, with the lower value (0-indexed) winning.
    The renderer's consumers should fall back to defaults for any
    missing column anyway, so the only effective difference is whether
    `Colour1` means "first column" or "second column".

    Pragmatic rule: any index `>= keys` is normalised down by 1 (treat
    as 1-indexed); anything `< keys` stays as-is (treat as 0-indexed).
    Since we don't yet know `keys` at parse time, we keep the raw value
    and let the consumer handle indexing — this function is a no-op for
    now but reserved for the consumer-side normalisation pass."""
    return int(raw)


def _parse_rgb(value: str) -> tuple[int, int, int] | None:
    """Parse `r,g,b` with optional whitespace. Returns None on bad input."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) < 3:
        return None
    try:
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if not all(0 <= c <= 255 for c in (r, g, b)):
        return None
    return r, g, b


def _parse_rgba(value: str) -> tuple[int, int, int, int] | None:
    """Parse `r,g,b` (alpha defaults to 255) or `r,g,b,a`."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) < 3:
        return None
    try:
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        a = int(parts[3]) if len(parts) >= 4 else 255
    except ValueError:
        return None
    if not all(0 <= c <= 255 for c in (r, g, b, a)):
        return None
    return r, g, b, a


def _parse_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


def _parse_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def _parse_bool(value: str) -> bool | None:
    """Parse osu!'s 0/1 booleans. Returns None on bad input so consumers
    can distinguish `unset` from `explicitly false`."""
    v = value.strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return None


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated int list. Bad entries are dropped, not
    fatal — partial skins still load."""
    out: list[int] = []
    for part in value.split(","):
        try:
            out.append(int(part.strip()))
        except ValueError:
            continue
    return tuple(out)
