"""FrameRenderer._is_argon_default keeps its answer (gpu/renderer.py).

It is asked ~28 times a frame and each answer walked ~32 atlas lookups. What
must hold:
  * the kept answer is the worked-out answer, case by case;
  * the atlas is asked once, however often the question comes;
  * a renderer given another atlas, another [Mania] section or another key
    count works it out again;
  * the reason it is safe: nothing writes an atlas' source tables except
    SpriteAtlas while it loads.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.gpu.renderer import FrameRenderer
from osu_mania_renderer_v2.wiki_elements import _common

PKG = Path(__file__).resolve().parents[1] / "osu_mania_renderer_v2"


class _Atlas:
    def __init__(self, column=None, glob=None):
        self.column, self.glob, self.asked = column or {}, glob or {}, 0

    def column_source(self, kind, col):
        self.asked += 1
        return self.column.get((kind, col), "missing")

    def global_source(self, name):
        self.asked += 1
        return self.glob.get(name, "missing")


def _renderer(atlas, section=None, keys=4):
    fr = object.__new__(FrameRenderer)
    fr.rc = SimpleNamespace(key_count=keys)
    fr.mania_section, fr.atlas = section, atlas
    return fr


CASES = [
    ("nothing from a skin", {}, {}, None, True),
    ("bundled and classic art only", {("note_tap", 0): "bundle", ("receptor_on", 3): "classic"},
     {"stage_left": "bundle"}, None, True),
    ("a user note", {("note_tap", 2): "user"}, {}, None, False),
    ("a beatmap key", {("receptor_off", 0): "beatmap"}, {}, None, False),
    ("a user stage piece", {}, {"stage_light": "user"}, None, False),
    ("a beatmap lighting sprite", {}, {"lighting_l": "beatmap"}, None, False),
    ("a [Mania] section", {}, {}, object(), False),
]


@pytest.mark.parametrize("name,column,glob,section,want", CASES, ids=[c[0] for c in CASES])
def test_the_kept_answer_is_the_worked_out_answer(name, column, glob, section, want):
    fr = _renderer(_Atlas(column, glob), section)
    assert fr._work_out_argon_default() is want
    assert [fr._is_argon_default() for _ in range(3)] == [want] * 3
    ctx = SimpleNamespace(fr=fr)
    assert _common._skin_provides_mania(ctx) is (not want)      # the wiki elements ask the same renderer


def test_the_atlas_is_asked_once():
    atlas = _Atlas()
    fr = _renderer(atlas)
    assert fr._is_argon_default() is True
    once = atlas.asked
    assert once > 0
    for _ in range(1000):
        fr._is_argon_default()
    assert atlas.asked == once


def test_another_atlas_section_or_key_count_is_worked_out_again():
    fr = _renderer(_Atlas())
    assert fr._is_argon_default() is True
    fr.atlas = _Atlas({("note_tap", 1): "user"})
    assert fr._is_argon_default() is False
    fr.atlas = _Atlas()
    assert fr._is_argon_default() is True
    fr.mania_section = object()
    assert fr._is_argon_default() is False
    fr.mania_section = None
    assert fr._is_argon_default() is True
    # a user sprite on the sixth column only counts once there are six columns
    fr.atlas = _Atlas({("note_tap", 5): "user"})
    assert fr._is_argon_default() is True
    fr.rc = SimpleNamespace(key_count=7)
    assert fr._is_argon_default() is False


def _writes_to_source_tables():
    """(file, enclosing class.function) of everything that changes
    `_column_sources` / `_global_sources`: an assignment or delete of one of
    their items, or a call of a method that changes a dict."""
    names = {"_column_sources", "_global_sources"}
    changing = {"update", "pop", "popitem", "clear", "setdefault", "__setitem__", "__delitem__"}
    found = []

    def table(node):
        return isinstance(node, ast.Attribute) and node.attr in names

    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

        def where(node):
            chain = []
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chain.append(node.name)
            return ".".join(reversed(chain))

        for node in ast.walk(tree):
            hit = False
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
                targets = (node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target])
                hit = any((isinstance(t, ast.Subscript) and table(t.value)) or table(t) for t in targets)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                hit = node.func.attr in changing and table(node.func.value)
            if hit:
                found.append((path.relative_to(PKG).as_posix(), where(node)))
    return found


def test_only_the_atlas_loader_writes_the_source_tables():
    found = _writes_to_source_tables()
    assert found, "the scan found no writer at all: it is not looking at the right thing"
    assert {f for f in found} <= {("gpu/atlas.py", "SpriteAtlas.__init__"), ("gpu/atlas.py", "SpriteAtlas.load")}, found
