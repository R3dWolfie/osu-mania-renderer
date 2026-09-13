"""Legacy timing chrome is available only for the Argon/default skin path."""
from __future__ import annotations

import pytest

from osu_mania_renderer_v2.beatmap.models import RenderOptions
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer


def _visibility(
    *,
    is_argon_default: bool,
    show_hit_error_popup: bool,
    show_ur_bar: bool,
) -> tuple[bool, bool]:
    renderer = object.__new__(FrameRenderer)
    renderer.options = RenderOptions(
        resolution=(1280, 720),
        fps=60,
        show_hit_error_popup=show_hit_error_popup,
        show_ur_bar=show_ur_bar,
    )
    renderer._is_argon_default = lambda: is_argon_default
    return renderer._legacy_timing_overlay_visibility()


@pytest.mark.parametrize(
    ("is_argon_default", "show_hit_error_popup", "show_ur_bar", "expected"),
    [
        pytest.param(False, True, True, (False, False), id="skinned-suppresses-both"),
        pytest.param(True, True, True, (True, True), id="argon-permits-both"),
        pytest.param(True, False, False, (False, False), id="both-options-disabled"),
        pytest.param(True, False, True, (False, True), id="hit-error-disabled"),
        pytest.param(True, True, False, (True, False), id="ur-disabled"),
    ],
)
def test_legacy_timing_overlay_visibility(
    is_argon_default: bool,
    show_hit_error_popup: bool,
    show_ur_bar: bool,
    expected: tuple[bool, bool],
):
    assert _visibility(
        is_argon_default=is_argon_default,
        show_hit_error_popup=show_hit_error_popup,
        show_ur_bar=show_ur_bar,
    ) == expected
