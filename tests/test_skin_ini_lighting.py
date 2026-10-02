"""Focused legacy Mania lighting transport coverage."""
from __future__ import annotations

import pytest

from osu_mania_renderer_v2.beatmap.skin_ini import parse_skin_ini
from osu_mania_renderer_v2.gpu.renderer import legacy_lighting_scale


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


@pytest.mark.parametrize("version,expected_scale", [(2.4, 1.0), (2.7, 22 / 30)])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_initial_general_header_preserves_lighting_version_with_or_without_bom(
    tmp_path, version, expected_scale, encoding,
):
    (tmp_path / "skin.ini").write_text(
        f"[General]\nVersion: {version}\nAnimationFramerate: 60\n[Mania]\n"
        "Keys: 4\nColumnWidth: 64,64,64,64\nLightingNWidth: 22,22,22,22\n"
        "LightingLWidth: 22,22,22,22\n",
        encoding=encoding,
    )
    parsed = parse_skin_ini(tmp_path)
    assert parsed.legacy_version == version
    assert parsed.animation_framerate == 60
    section = parsed.mania_for_keycount(4)
    for kind in ("n", "l"):
        assert legacy_lighting_scale(section, 0, kind, legacy_version=parsed.legacy_version) == pytest.approx(expected_scale)
