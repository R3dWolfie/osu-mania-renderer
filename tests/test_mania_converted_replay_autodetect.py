"""Replay-driven standard conversion and pinned lazer column-count semantics."""
import hashlib

import pytest

import osu_mania_renderer_v2 as api
from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
from osu_mania_renderer_v2.beatmap.models import KeyEvent, RenderOptions, ReplayInfo
from osu_mania_renderer_v2.beatmap.mods import Mod
from osu_mania_renderer_v2.beatmap.replay import parse_replay
from osu_mania_renderer_v2.cli import _build_parser
from osu_mania_renderer_v2.errors import NotAManiaError
from osu_mania_renderer_v2.render import render as rendering
from osu_mania_renderer_v2.wiki_renderer import _build_wiki_parser


def source_map(tmp_path, *, mode=0, cs=4, od=5, total=10, duration=0,
               duration_kind="mixed"):
    rows = []
    for i in range(total):
        start = 1000 + i * 1000
        if i >= duration:
            row = f"256,192,{start},1,0,0:0:0:0:"
        elif duration_kind == "spinner" or (duration_kind == "mixed" and i % 2):
            row = f"256,192,{start},8,0,{start + 500},0:0:0:0:"
        else:
            row = f"256,192,{start},2,0,L|356:192,1,100,0|0,0:0|0:0,0:0:0:0:"
        rows.append(row)
    path = tmp_path / "source.osu"
    path.write_text(
        f"osu file format v14\n[General]\nMode:{mode}\n"
        "[Metadata]\nTitle:Conversion regression\nVersion:Test\n"
        f"[Difficulty]\nCircleSize:{cs}\nOverallDifficulty:{od}\n"
        "HPDrainRate:5\nApproachRate:5\nSliderMultiplier:1.4\n"
        "[TimingPoints]\n0,500,4,1,0,100,1,0\n[HitObjects]\n"
        + "\n".join(rows) + "\n", encoding="utf-8",
    )
    return path


def mania_replay(path, *, mods=0, mask=1):
    # Synthetic supplied-fixture shape; never depend on a private replay path.
    return ReplayInfo(
        mode=3, beatmap_md5=hashlib.md5(path.read_bytes()).hexdigest(),
        player_name="Regression", replay_md5="", mods=int(mods),
        key_events=(KeyEvent(1000, mask), KeyEvent(1020, 0)),
        score=987863, accuracy=100, max_combo=235,
        count_geki=1, count_300=0, count_katu=0, count_100=0,
        count_50=0, count_miss=0, grade="SS",
    )


@pytest.fixture
def replay_plan(monkeypatch, tmp_path):
    async def encoder(*_):
        return "libx264"
    monkeypatch.setattr(rendering, "probe_encoder", encoder)

    async def build(path, *, mods=0, mask=1, **kwargs):
        replay = mania_replay(path, mods=mods, mask=mask)
        monkeypatch.setattr(rendering, "parse_replay", lambda _: replay)
        return await rendering.build_render_plan(
            osr_path=tmp_path / "synthetic.osr", beatmap_dir=tmp_path,
            output_path=tmp_path / "unused.mp4",
            options=RenderOptions((320, 240), 60, use_replay_hitsounds=False),
            **kwargs,
        )
    return build


@pytest.mark.parametrize("mask", [0, 1, 127])
async def test_default_mania_replay_over_standard_source_autoconverts(
    tmp_path, replay_plan, mask,
):
    plan = await replay_plan(source_map(tmp_path), mask=mask)
    assert plan.replay.mode == 3 and plan.replay.mods == 0
    assert plan.modded.source_mode == 0
    assert plan.modded.key_count == 7  # independent of observed replay lanes
    assert plan.modded.notes and "converted 7K" in plan.modded.difficulty


async def test_supplied_replay_source_shape_uses_automatic_7k(tmp_path, replay_plan):
    # Real source facts: 69 objects, 39 sliders/spinners, CS 4 and OD 6.5.
    # The fallback rounds OD to 6 (even) and selects 7; masks are only evidence.
    path = source_map(tmp_path, cs=4, od=6.5, total=69, duration=39)
    plan = await replay_plan(path, mask=127)
    assert plan.replay.mode == 3 and plan.replay.mods == 0
    assert plan.modded.source_mode == 0 and plan.modded.key_count == 7


async def test_native_mania_stays_native_with_key_mod_and_manual_override(
    tmp_path, monkeypatch, replay_plan,
):
    from osu_mania_renderer_v2.beatmap import converter
    def unexpected(**_):
        pytest.fail("Native Mania must not enter the standard converter")
    monkeypatch.setattr(converter, "convert_standard_to_mania", unexpected)
    path = source_map(tmp_path, mode=3, cs=4)
    native = parse_beatmap(path)
    plan = await replay_plan(path, mods=Mod.K7, convert_to_keys=5)
    assert plan.modded == native
    assert plan.modded.source_mode == 3 and plan.modded.key_count == 4


# Counts come from the actual slider/spinner/circle rows above. Boundary
# equality and midpoint cases pin Math.Round's to-even behavior and branch order.
@pytest.mark.parametrize("total,duration,cs,od,expected", [
    (10, 0, 4, 2, 7),
    (10, 1, 4, 2, 7),
    (10, 2, 4, 5, 6),
    (10, 2, 4, 6, 7),
    (10, 3, 5, 5, 6),
    (10, 3, 5, 6, 7),
    (10, 7, 4, 4, 4),
    (10, 7, 4, 5, 5),
    (10, 7, 5, 5, 6),  # CS branch precedes the high-duration branch
    (10, 3, 4, 1, 4),
    (10, 3, 4, 4, 5),
    (10, 3, 4, 5, 6),
    (10, 3, 4, 9, 7),
    (10, 6, 4, 4, 5),  # exactly .6 takes fallback, not > .6
    (10, 2, 4, 5.5, 7),
    (10, 2, 4, 4.5, 6),
    (10, 3, 4.5, 2, 4),
    (10, 3, 5.5, 2, 6),
    (10, 7, 4, 4.5, 4),
    (10, 7, 4, 5.5, 5),
    (10, 3, 4, 3.5, 5),
    (10, 3, 4, 4.5, 5),
    (0, 0, 5, 1, 4),
    (0, 0, 5, 4.5, 5),
    (0, 0, 5, 9, 7),
])
def test_pinned_lazer_automatic_column_count(tmp_path, total, duration, cs, od, expected):
    path = source_map(tmp_path, total=total, duration=duration, cs=cs, od=od)
    beatmap = parse_beatmap(path, allow_converted=True)
    assert beatmap.source_mode == 0 and beatmap.key_count == expected
    assert all(0 <= n.column < expected for n in beatmap.notes)


@pytest.mark.parametrize("duration_kind", ["slider", "spinner", "mixed"])
def test_duration_count_uses_source_objects_not_generated_mania_notes(tmp_path, duration_kind):
    path = source_map(tmp_path, duration=7, od=4, duration_kind=duration_kind)
    assert parse_beatmap(path, allow_converted=True).key_count == 4


def test_zero_duration_spinners_still_count_as_duration_bearing_objects(tmp_path):
    path = source_map(tmp_path, duration=7, od=4, duration_kind="spinner")
    text = path.read_text()
    for i in range(7):
        start = 1000 + i * 1000
        text = text.replace(f",8,0,{start + 500},", f",8,0,{start},")
    path.write_text(text)
    assert parse_beatmap(path, allow_converted=True).key_count == 4


@pytest.mark.parametrize("keys", range(1, 10))
async def test_legacy_key_mod_overrides_source_auto(tmp_path, replay_plan, keys):
    # Auto gives 6K, so the specifically requested K4/K7/K9 all differ.
    path = source_map(tmp_path, duration=2, od=5)
    plan = await replay_plan(path, mods=getattr(Mod, f"K{keys}"))
    assert plan.modded.source_mode == 0 and plan.modded.key_count == keys


@pytest.mark.parametrize("allow_converted", [False, True])
async def test_explicit_key_override_wins_over_replay_mod_and_auto(
    tmp_path, replay_plan, allow_converted,
):
    plan = await replay_plan(source_map(tmp_path), mods=Mod.K7,
                             convert_to_keys=5, allow_converted=allow_converted)
    assert plan.modded.key_count == 5 and plan.modded.source_mode == 0


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("allow_converted", [False, True])
async def test_unsupported_source_modes_are_rejected(tmp_path, replay_plan, mode, allow_converted):
    with pytest.raises(NotAManiaError, match=f"got mode={mode}"):
        await replay_plan(source_map(tmp_path, mode=mode), allow_converted=allow_converted)


def test_standalone_source_parse_still_requires_conversion_opt_in(tmp_path):
    with pytest.raises(NotAManiaError):
        parse_beatmap(source_map(tmp_path))


def test_replay_parser_still_rejects_standard_replays(fixtures_dir):
    with pytest.raises(NotAManiaError):
        parse_replay(fixtures_dir / "std_replay.osr")


@pytest.mark.parametrize("parser", [_build_parser, _build_wiki_parser])
def test_cli_omission_is_auto_and_explicit_override_is_distinct(parser):
    args = ["replay.osr", "beatmap", "-o", "unused.mp4"]
    assert parser().parse_args(args).convert_to_keys is None
    assert parser().parse_args(args + ["--convert-to-keys", "5"]).convert_to_keys == 5
    assert "taiko/ctb" not in parser().format_help()


async def test_public_render_api_preserves_automatic_count(tmp_path, monkeypatch, replay_plan):
    path = source_map(tmp_path)
    received = []
    async def gpu(**kwargs):
        received.append(await replay_plan(path, convert_to_keys=kwargs["convert_to_keys"]))
    monkeypatch.setattr(api, "USE_WIKI_RENDERER", False)
    monkeypatch.setattr(api, "_gpu_render_mania", gpu)
    await api.render_mania(osr_path=tmp_path / "synthetic.osr", beatmap_dir=tmp_path,
                           output_path=tmp_path / "unused.mp4", options=RenderOptions((320, 240), 60))
    assert received[0].modded.key_count == 7
