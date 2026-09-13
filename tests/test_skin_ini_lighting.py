"""Focused legacy Mania lighting transport coverage."""
from __future__ import annotations

from osu_mania_renderer_v2.beatmap.skin_ini import parse_skin_ini


def test_mania_lighting_paths_and_float_widths_preserve_column_identity(tmp_path):
    skin = tmp_path / "skin"
    skin.mkdir()
    (skin / "skin.ini").write_text(
        """
[GeNeRaL]
VeRsIoN: latest

[mAnIa]
KeYs: 4
LiGhTiNgN: blank
lIgHtInGl: Effects\\Hold Light
LIGHTINGNWIDTH: 42.5,broken,70.25,
lightinglwidth: nope,30.5,0,64
""".strip(),
        encoding="utf-8",
    )

    parsed = parse_skin_ini(skin)
    section = parsed.mania_for_keycount(4)

    assert section is not None
    assert section.lighting_n == "blank"
    assert section.lighting_l == "Effects\\Hold Light"
    assert section.lighting_n_width == (42.5, 0.0, 70.25, 0.0)
    assert section.lighting_l_width == (0.0, 30.5, 0.0, 64.0)
    assert parsed.legacy_version == 2.7
