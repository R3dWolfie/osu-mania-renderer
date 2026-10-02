"""Pure osu!stable legacy stage topology; columns always retain global indices.

Authority: SkinManager.LoadManiaSkin, StageMania_Calculations and
StageMania/ColumnMania in the archived stable source. This is not lazer's
DualStages architecture. The caller supplies the already-resolved total key
count (native CS; converted key count is doubled by stable's converter for KC).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from osu_mania_renderer_v2.beatmap.mods import Mod
from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection


@dataclass(frozen=True)
class LegacyStage:
    index: int
    first_column: int
    column_count: int
    special_style: int = 0
    x: float = 0.0
    width: float = 0.0

    @property
    def end_column(self) -> int:
        return self.first_column + self.column_count

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def special_column(self) -> int | None:
        n = self.column_count
        if n % 2:
            return n // 2
        if n > 4 and self.special_style in (1, 2):
            return 0 if self.special_style == 1 else n - 1
        return None

    def column_kind(self, local_column: int) -> str:
        if not 0 <= local_column < self.column_count:
            raise IndexError(local_column)
        n, c = self.column_count, local_column
        if c == self.special_column:
            return "S"
        if n > 4 and n % 2 == 0 and self.special_style in (1, 2):
            inner = c % 2 == (0 if self.special_style == 1 else 1)
        elif n % 2:
            inner = c % 2 == 1
        else:
            inner = c % 2 == (1 if c < n // 2 else 0)
        return "2" if inner else "1"


@dataclass(frozen=True)
class LegacyStageLayout:
    stages: tuple[LegacyStage, ...]
    column_x: tuple[float, ...] = ()
    column_width: tuple[float, ...] = ()
    width_scale: float = 1.0
    separation: float = 0.0

    def locate(self, global_column: int) -> tuple[LegacyStage, int]:
        for stage in self.stages:
            if stage.first_column <= global_column < stage.end_column:
                return stage, global_column - stage.first_column
        raise IndexError(global_column)

    def column_kind(self, global_column: int) -> str:
        stage, local = self.locate(global_column)
        return stage.column_kind(local)

    def column_center(self, global_column: int) -> float:
        """Stable's Column.Left + Column.Width / 2, without integer snapping."""
        return self.column_x[global_column] + self.column_width[global_column] / 2.0

    def geometry_rows(self) -> tuple[dict, ...]:
        """Diagnostic table; stage origin can precede its first column."""
        rows = []
        for c, (x, width) in enumerate(zip(self.column_x, self.column_width)):
            stage, local = self.locate(c)
            rows.append(
                dict(
                    global_column=c,
                    stage=stage.index,
                    local_column=local,
                    x=x,
                    width=width,
                    center=self.column_center(c),
                    stage_x=stage.x,
                    stage_width=stage.width,
                    stage_right=stage.right,
                )
            )
        return tuple(rows)

    def gap_geometry(self) -> dict | None:
        """Separate chrome gap from the first secondary column's spacing."""
        if len(self.stages) != 2:
            return None
        left, right = self.stages
        column_left = self.column_x[right.first_column]
        return dict(
            stage_one_right=left.right,
            stage_two_left=right.x,
            visible_column_gap=column_left - left.right,
            stage_separation=self.separation,
            boundary_column_spacing=column_left - right.x,
        )


def legacy_stage_topology(
    key_count: int,
    section: ManiaSection | None = None,
    *,
    mods: int = 0,
) -> LegacyStageLayout:
    """Skin override wins over >10/KC auto split; 1K can never split.

    Stable supports 1..18 keys, but does not clamp native CS here. Do not
    silently discard columns in malformed/extended maps either.
    """
    if key_count < 1:
        raise ValueError("key_count must be positive")
    if section is not None and section.keys != key_count:
        section = None
    override = section.split_stages if section is not None else None
    split = key_count > 1 and (
        override if override is not None else key_count > 10 or bool(mods & Mod.KC)
    )
    counts = ((key_count + 1) // 2, key_count // 2) if split else (key_count,)
    style = section.special_style if section is not None else 0
    stages = []
    offset = 0
    for index, count in enumerate(counts):
        # Stable mirrors a left/right special style on the second stage.
        local_style = (3 - style) if index % 2 and style in (1, 2) else style
        stages.append(LegacyStage(index, offset, count, local_style or 0))
        offset += count
    return LegacyStageLayout(tuple(stages))


def legacy_stage_layout(
    key_count: int,
    section: ManiaSection | None,
    *,
    render_width: float,
    render_height: float,
    mods: int = 0,
) -> LegacyStageLayout:
    """Stable's ColumnStart/Right, width fit, and global spacing in 480 units.

    CSV overrides replace only supplied indices (Skin.ReadList); they are
    NOT broadcast. Stage two retains the preceding *global* ColumnSpacing
    as a leading inset in addition to StageSeparation. The stage's X/width
    includes that inset, whereas its first column begins after it.
    """
    if section is not None and section.keys != key_count:
        section = None
    topology = legacy_stage_topology(key_count, section, mods=mods)
    scale = render_height / 480.0
    screen_width = ceil(render_width / scale)  # WindowManager.WidthScaled

    def values(authored, count, default):
        return [float(authored[i]) if i < len(authored) else default for i in range(count)]

    widths = values(section.column_width if section else (), key_count, 30.0)
    widths = [max(5.0, min(100.0, w)) for w in widths]
    spacing = values(section.column_spacing if section else (), key_count - 1, 0.0)
    spacing = [max(-widths[i + 1], s) for i, s in enumerate(spacing)]
    start = section.column_start if section and section.column_start is not None else 136.0
    reserve = section.column_right if section and section.column_right is not None else 19.0
    separation = max(
        5.0, section.stage_separation if section and section.stage_separation is not None else 40.0
    )
    if len(topology.stages) == 1:
        separation = 0.0
    left = min(start, screen_width - reserve)
    right = min(reserve, screen_width - start)
    resize_width = sum(widths) + sum(spacing) + separation
    overflow = max(0.0, left + resize_width + right - screen_width)
    fit = (resize_width - overflow) / resize_width
    x = left * scale
    column_x, column_width, stages = [], [], []
    for stage in topology.stages:
        stage_x = x
        for c in range(stage.first_column, stage.end_column):
            x += (spacing[c - 1] if c else 0.0) * fit * scale
            column_x.append(x)
            column_width.append(widths[c] * fit * scale)
            x += column_width[-1]
        stages.append(
            LegacyStage(
                stage.index,
                stage.first_column,
                stage.column_count,
                stage.special_style,
                stage_x,
                x - stage_x,
            )
        )
        x += separation * fit * scale
    return LegacyStageLayout(
        tuple(stages), tuple(column_x), tuple(column_width), fit, separation * fit * scale
    )
