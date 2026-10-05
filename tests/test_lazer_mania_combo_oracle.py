"""The supplied Round 3.6b replay is an external oracle, never builder input."""
import asyncio
import hashlib
import struct
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
from osu_mania_renderer_v2.beatmap.models import HoldNote, RenderOptions
from osu_mania_renderer_v2.beatmap.replay import parse_replay
from osu_mania_renderer_v2.render.lazer_mania_combo import build_lazer_combo_timeline
from osu_mania_renderer_v2.render import render as render_module

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'lazer_combo_oracle'
REPLAY_PATH = FIXTURE_DIR / '01M45P6TFG83ZNP9G2GRSZVB2N.osr'
MAP_PATH = FIXTURE_DIR / 'Saikoro - far in the blue sky... (1nar) [42 [1.1]].osu'


@pytest.fixture(scope='module')
def oracle():
    replay = parse_replay(REPLAY_PATH)
    beatmap = parse_beatmap(MAP_PATH)
    timeline = build_lazer_combo_timeline(beatmap.notes, replay.ordered_key_events,
                                         beatmap.key_count, od=beatmap.overall_difficulty)
    return replay, beatmap, timeline


def test_exact_supplied_lazer_replay_reconstructs_1194_from_ordered_input(oracle):
    replay, beatmap, timeline = oracle
    assert struct.unpack_from('<i', REPLAY_PATH.read_bytes(), 1)[0] == 30000019
    assert hashlib.md5(MAP_PATH.read_bytes()).hexdigest() == replay.beatmap_md5
    assert replay.is_lazer_replay
    assert replay.beatmap_md5 == '37fa653eeddc8e59bfd371219b1d81f9'
    assert replay.max_combo == 1194
    assert replay.mods == 0 and replay.score == 795932
    assert (replay.count_geki, replay.count_300, replay.count_katu, replay.count_100,
            replay.count_50, replay.count_miss) == (2082, 812, 217, 39, 10, 47)
    assert sum((replay.count_geki, replay.count_300, replay.count_katu, replay.count_100,
                replay.count_50, replay.count_miss)) == 3207
    holds = sum(isinstance(note, HoldNote) for note in beatmap.notes)
    assert holds == 40 and len(beatmap.notes) - holds == 3127
    assert len(beatmap.notes) + holds == 3207
    assert timeline.reconstructed_max_combo == 1194
    assert len(timeline.facts) == 3207
    assert sum(f.result == 'miss' for f in timeline.facts) == 47


def test_oracle_streak_ends_with_consuming_an_early_press_in_the_miss_window(oracle):
    _, _, timeline = oracle
    assert timeline.at(97632).combo == 1194
    index = next(i for i, f in enumerate(timeline.facts) if f.kind == 'reset')
    fact = timeline.facts[index]
    assert (fact.time_ms, fact.column, fact.object_time_ms, fact.source,
            fact.result, fact.cause, fact.input_frame_index) == (97641, 2, 97797, 'tap', 'miss', 'press', 7260)
    assert timeline.states[index - 1].combo == 1194
    assert timeline.states[index].combo == 0
    newer = next(f for f in timeline.facts if f.time_ms == 97900 and f.column == 2)
    assert newer.object_time_ms == 97889 and newer.kind == 'increment'


def test_oracle_retains_transient_input_that_timestamp_dedup_would_lose(oracle):
    replay, beatmap, timeline = oracle
    frames = replay.ordered_key_events
    assert len(frames) == 14095
    assert sum(e.time_ms == (frames[i - 1].time_ms if i else 0) for i, e in enumerate(frames)) == 893
    grouped = {}
    for event in frames:
        grouped.setdefault(event.time_ms, []).append(event.keys_held)
    assert sum(len(set(masks)) > 1 for masks in grouped.values()) == 856
    # This transient press hits a real note; last-mask-only input misses it.
    def result_for(t):
        return next(f for f in t.facts if f.column == 3 and f.object_time_ms == 126084)
    raw = result_for(timeline)
    assert (raw.time_ms, raw.result, raw.cause, raw.input_frame_index) == (126118, '300', 'press', 10121)
    flattened = build_lazer_combo_timeline(beatmap.notes, replay.key_events,
                                          beatmap.key_count, od=beatmap.overall_difficulty)
    assert result_for(flattened).result == 'miss'
    # The maximum alone happens to survive flattening; reset positions do not.
    assert flattened.reconstructed_max_combo == 1194
    assert Counter(f.result for f in flattened.facts)['miss'] == 43


def test_actual_oracle_plan_scene_uses_facts_not_reconciled_judgments(tmp_path, monkeypatch, oracle):
    async def encoder(*_):
        return 'libx264'
    monkeypatch.setattr(render_module, 'probe_encoder', encoder)
    plan = asyncio.run(render_module.build_render_plan(osr_path=REPLAY_PATH, beatmap_dir=FIXTURE_DIR,
        output_path=tmp_path / 'unused.mp4', options=RenderOptions((320, 240), 60, encoder='libx264')))
    assert plan.replay.is_lazer_replay and plan.stable_combo_timeline is None
    assert plan.lazer_combo_timeline.facts == oracle[2].facts
    assert plan.lazer_combo_timeline.reconstructed_max_combo == 1194
    def scene(now):
        return render_module.build_frame_state(plan, now, 0, 100)[0]
    assert scene(97632).combo == 1194
    state = scene(97641)
    assert (state.combo, state.combo_break_previous_value, state.combo_break_age_ms,
            state.reconstructed_max_combo) == (0, 1194, 0, 1194)
    assert scene(97900).combo == 5
    assert scene(126118).combo == plan.lazer_combo_timeline.at(126118).combo
    # A header edit and removal of the entire reconciled timeline cannot move
    # source resets. The results maximum still presents the edited header.
    plan.replay = replace(plan.replay, max_combo=9999)
    plan.judgment_timeline = []
    del plan._tl_times
    for now in (170500, 97632, 97641, 97900, 126118, 5000):
        assert scene(now).combo == plan.lazer_combo_timeline.at(now).combo
    assert scene(170500).reconstructed_max_combo == 1194
    assert scene(plan.results_start_ms + 1000).max_combo == 9999


def test_oracle_header_max_and_all_miss_reconciliation_cannot_change_builder_facts(tmp_path, monkeypatch, oracle):
    replay, _, timeline = oracle
    async def encoder(*_):
        return 'libx264'
    monkeypatch.setattr(render_module, 'probe_encoder', encoder)
    monkeypatch.setattr(render_module, 'parse_replay', lambda _: replace(
        replay, max_combo=0, count_geki=0, count_300=0, count_katu=0,
        count_100=0, count_50=0, count_miss=3207))
    plan = asyncio.run(render_module.build_render_plan(osr_path=REPLAY_PATH, beatmap_dir=FIXTURE_DIR,
        output_path=tmp_path / 'unused.mp4', options=RenderOptions((320, 240), 60, encoder='libx264')))
    assert plan.lazer_combo_timeline.facts == timeline.facts
    assert plan.lazer_combo_timeline.reconstructed_max_combo == 1194
    assert all(j.judgment == 'miss' for j in plan.judgment_events)
    scene = render_module.build_frame_state(plan, 97632, 0, 100)[0]
    assert (scene.combo, scene.max_combo, scene.reconstructed_max_combo) == (1194, 0, 1194)
