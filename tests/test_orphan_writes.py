"""Fresh storage before every small buffer upload (gpu/renderer.py
_stream_write, the switch R3D_MANIA_ORPHAN).

What must hold:
  * with the switch on, a buffer is orphaned and THEN written, once each, with
    the bytes it was given;
  * with it off, it is only written, exactly as before;
  * the default is on on macOS (where it was measured) and off anywhere else;
    R3D_MANIA_ORPHAN forces it either way;
  * every upload the renderer makes on the frame path goes through the helper.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from osu_mania_renderer_v2.gpu import renderer as R


class _Buffer:
    def __init__(self):
        self.calls = []

    def orphan(self):
        self.calls.append(("orphan",))

    def write(self, data):
        self.calls.append(("write", bytes(data)))


def _with(flag, fn):
    old = R._ORPHAN_WRITES
    R._ORPHAN_WRITES = flag
    try:
        return fn()
    finally:
        R._ORPHAN_WRITES = old


def test_on_orphans_then_writes_the_same_bytes():
    b = _Buffer()
    _with(True, lambda: R._stream_write(b, b"\x01\x02\x03"))
    assert b.calls == [("orphan",), ("write", b"\x01\x02\x03")]


def test_off_only_writes_as_before():
    b = _Buffer()
    _with(False, lambda: R._stream_write(b, b"\x01\x02\x03"))
    assert b.calls == [("write", b"\x01\x02\x03")]


def test_the_switch(monkeypatch):
    monkeypatch.delenv("R3D_MANIA_ORPHAN", raising=False)
    monkeypatch.setattr(R.sys, "platform", "darwin")
    assert R._orphan_writes() is True
    monkeypatch.setattr(R.sys, "platform", "linux")
    assert R._orphan_writes() is False
    monkeypatch.setattr(R.sys, "platform", "win32")
    assert R._orphan_writes() is False
    for value, want in (("1", True), ("0", False), ("off", False), ("yes", True)):
        monkeypatch.setenv("R3D_MANIA_ORPHAN", value)
        for platform in ("darwin", "linux"):
            monkeypatch.setattr(R.sys, "platform", platform)
            assert R._orphan_writes() is want, (value, platform)


def test_every_upload_on_the_frame_path_goes_through_the_helper():
    src = Path(R.__file__).read_text()
    body = src.split("def _stream_write", 1)[1].split("\n\n\n", 1)[1]
    # no direct `.write(` on a vertex or instance buffer is left below the helper
    direct = [m.group(0) for m in re.finditer(r"\b(vbo|_instance_vbo)\.write\(", body)]
    assert direct == [], direct
    assert body.count("_stream_write(") == 3
