"""Source behaviours that counts and nearest-edge matching cannot recover."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.beatmap.models import HoldNote, KeyEvent, Note
from osu_mania_renderer_v2.beatmap.mods import Mod
from osu_mania_renderer_v2.beatmap.replay import _decode_key_events, _decode_ordered_key_events
from osu_mania_renderer_v2.render.lazer_mania_combo import LazerManiaWindows, build_lazer_combo_timeline


def simulate(notes, edges=(), key_count=2, **kwargs):
    return build_lazer_combo_timeline(tuple(notes), tuple(KeyEvent(*edge) for edge in edges),
                                      key_count, **kwargs)


def presses(timeline):
    return [(f.time_ms, f.column, f.object_time_ms, f.result) for f in timeline.facts if f.cause == 'press']


@pytest.mark.parametrize('offset,result', [(-174, None), (-173.5, 'miss'), (-137, 'miss'),
    (-136.5, '50'), (-112.5, '100'), (-82.5, 'katu'), (-49.5, '300'), (-19.5, 'geki'),
    (19.5, 'geki'), (19.6, '300'), (136.5, '50'), (136.6, 'miss'), (173.5, 'miss'), (173.6, None)])
def test_lazer_windows_include_consuming_miss_band_and_half_ms_borders(offset, result):
    assert LazerManiaWindows.build(5).result_for(offset) == result


@pytest.mark.parametrize('press,result', [(97641, 'miss'), (97633, 'miss'), (97632, None)])
def test_oracle_early_press_156_ms_before_97797_consumes_note_as_miss(press, result):
    timeline = simulate([Note(0, 97797)], [(press, 1), (97700, 0), (97797, 1)], od=8)
    first = next((f for f in timeline.facts if f.time_ms == press and f.cause == 'press'), None)
    assert (first.result if first else None) == result
    assert timeline.reconstructed_max_combo == (1 if result is None else 0)


@pytest.mark.parametrize('when', [97889, 97900])
def test_real_97797_97889_97900_previous_window_cutoff_and_forced_miss(when):
    timeline = simulate([Note(0, 97797), Note(0, 97889)], [(when, 1)], od=8)
    assert presses(timeline) == [(when, 0, 97889, 'geki')]
    previous = next(f for f in timeline.facts if f.object_time_ms == 97797)
    assert (previous.time_ms, previous.result, previous.cause) == (when, 'miss', 'ordered-force-miss')
    # NewResult reaches the score processor before Column.HandleHit forces
    # earlier misses (Playfield.AddNested registers forwarding before load).
    assert [s.combo for s in timeline.states] == [1, 0]


def test_previous_note_can_still_consume_press_strictly_before_next_start():
    timeline = simulate([Note(0, 97797), Note(0, 97889)], [(97888, 1)], od=8)
    assert presses(timeline) == [(97888, 0, 97797, '100')]


def test_dense_same_column_input_uses_earliest_hittable_not_nearest_note():
    timeline = simulate([Note(0, 1000), Note(0, 1070), Note(0, 1140)],
        [(950, 1), (990, 0), (1110, 1), (1120, 0), (1140, 1)])
    assert presses(timeline) == [(950, 0, 1000, 'katu'), (1110, 0, 1070, '300'), (1140, 0, 1140, 'geki')]
    assert timeline.reconstructed_max_combo == 3


def test_a_very_early_press_is_ignored_and_does_not_consume_future_head():
    timeline = simulate([Note(0, 1000)], [(800, 1), (900, 0), (1000, 1)])
    assert presses(timeline) == [(1000, 0, 1000, 'geki')]
    assert timeline.at(1000).combo == 1


def test_chord_columns_are_independent_of_other_columns_note_lock():
    timeline = simulate([Note(0, 900), Note(1, 1000), Note(1, 1060)], [(1000, 2), (1200, 0)])
    assert presses(timeline) == [(1000, 1, 1000, 'geki')]
    assert not any(f.cause == 'ordered-force-miss' for f in timeline.facts)
    assert timeline.at(1000).combo == 1


@pytest.mark.parametrize('notes,edges,expected_max', [
    ([Note(0, 1000), Note(0, 1100)], [(1000, 1), (1100, 0), (1100, 1)], 2),
    ([Note(0, 1000)], [(900, 0), (1000, 1), (1000, 0)], 1),
    ([Note(0, 1000)], [(1000, 0), (1000, 1), (1000, 0)], 1),
    ([Note(0, 1000), Note(1, 1000)], [(1000, 0), (1000, 3), (1000, 0)], 2),
    ([HoldNote(0, 1000, 2000), Note(1, 1500)],
        [(1000, 1), (1500, 0), (1500, 2), (1500, 0), (2000, 0)], 1),
], ids=['release-then-press', 'press-then-release', 'transient-equal-endpoints',
         'different-columns', 'adjacent-body-break'])
def test_same_timestamp_raw_frames_change_results_that_dictionary_dedup_loses(
        tmp_path, notes, edges, expected_max):
    raw_frames = []
    clock = 0
    for now, mask in edges:
        raw_frames.append(SimpleNamespace(time_delta=now - clock, keys=mask))
        clock = now
    raw_frames.append(SimpleNamespace(time_delta=-12345, keys=1234))
    replay = SimpleNamespace(replay_data=raw_frames)
    path = tmp_path / 'stub.osr'
    path.write_bytes(b'stub')
    ordered = _decode_ordered_key_events(replay, path)
    flattened = _decode_key_events(replay, path)
    assert [(e.time_ms, e.keys_held) for e in ordered] == edges
    assert len(flattened) < len(ordered)
    timeline = build_lazer_combo_timeline(tuple(notes), tuple(ordered), 2)
    lossy = build_lazer_combo_timeline(tuple(notes), tuple(flattened), 2)
    assert timeline.reconstructed_max_combo == expected_max
    assert [(f.object_time_ms, f.kind, f.result) for f in timeline.facts] != [
        (f.object_time_ms, f.kind, f.result) for f in lossy.facts]


def test_replay_state_dispatches_all_releases_before_column_ordered_presses():
    timeline = simulate([HoldNote(0, 1000, 2000), Note(1, 1500)], [(1000, 1), (1500, 2)])
    at_time = [f for f in timeline.facts if f.time_ms == 1500]
    assert [(f.source, f.kind, f.column) for f in at_time] == [('body', 'reset', 0), ('tap', 'increment', 1)]
    assert timeline.at(1500).combo == 1
    assert timeline.at(1500).break_previous_value == 1


def test_clean_hold_head_and_tail_count_body_ignore_and_no_stable_ticks():
    timeline = simulate([HoldNote(0, 1000, 2000)], [(1000, 1), (1500, 1), (2000, 0)])
    assert [(f.source, f.kind, f.time_ms) for f in timeline.facts] == [
        ('head', 'increment', 1000), ('tail', 'increment', 2000)]
    assert timeline.at(1500).combo == 1
    assert timeline.reconstructed_max_combo == 2


def test_early_body_drop_breaks_combo_without_an_accuracy_result():
    timeline = simulate([HoldNote(0, 1000, 2000)], [(1000, 1), (1250, 0), (2300, 0)])
    dropped = timeline.at(1250)
    assert (dropped.combo, dropped.break_previous_value, dropped.last_break_ms) == (0, 1, 1250)
    assert [(f.result, f.source) for f in timeline.facts] == [
        ('geki', 'head'), ('combo-break', 'body'), ('miss', 'tail')]
    assert sum(f.source in ('head', 'tail') for f in timeline.facts) == 2


@pytest.mark.parametrize('release', [1796, 1800, 1990, 2000, 2204])
def test_tail_success_lenience_suppresses_body_combo_break(release):
    timeline = simulate([HoldNote(0, 1000, 2000)], [(1000, 1), (release, 0)])
    assert timeline.reconstructed_max_combo == 2
    assert [f.source for f in timeline.facts] == ['head', 'tail']
    assert all(f.kind == 'increment' for f in timeline.facts)


def test_release_in_tail_miss_band_consumes_tail_and_breaks_body():
    timeline = simulate([HoldNote(0, 1000, 2000)], [(1000, 1), (1770, 0), (1800, 1), (2000, 0)])
    assert [(f.time_ms, f.source, f.result) for f in timeline.facts] == [
        (1000, 'head', 'geki'), (1770, 'tail', 'miss'), (1770, 'body', 'combo-break')]
    assert timeline.at(2000).combo == 0


def test_broken_body_later_tail_hit_is_capped_but_increases_combo():
    timeline = simulate([HoldNote(0, 1000, 2000)], [(1000, 1), (1250, 0), (1320, 1), (2000, 0)])
    assert [(f.source, f.result) for f in timeline.facts] == [
        ('head', 'geki'), ('body', 'combo-break'), ('tail', '50')]
    assert timeline.at(2000).combo == 1
    assert timeline.at(1320).combo == 0


def test_body_result_is_emitted_once_across_multiple_drop_resume_cycles():
    timeline = simulate([HoldNote(0, 1000, 2500)],
        [(1000, 1), (1250, 0), (1320, 1), (1500, 0), (1600, 1), (2500, 0)])
    assert len([f for f in timeline.facts if f.source == 'body']) == 1
    assert timeline.facts[-1].source == 'tail' and timeline.facts[-1].result == '50'


def test_missed_head_can_resume_and_successful_tail_still_counts_as_meh():
    timeline = simulate([HoldNote(0, 1000, 2000)], [(1140, 0), (1500, 1), (2000, 0)])
    assert [(f.source, f.result) for f in timeline.facts] == [('head', 'miss'), ('tail', '50')]
    assert timeline.at(2000).combo == 1


def test_hold_cannot_begin_during_late_tail_lenience():
    timeline = simulate([HoldNote(0, 1000, 2000)], [(1140, 0), (2150, 1), (2200, 0), (2300, 0)])
    assert not any(f.kind == 'increment' for f in timeline.facts)
    assert timeline.reconstructed_max_combo == 0


def test_tail_is_not_matched_to_a_nearby_release_without_hold_ownership():
    timeline = simulate([HoldNote(0, 1000, 2000)], [(500, 1), (2000, 0), (2300, 0)])
    assert not any(f.cause == 'release' for f in timeline.facts)
    assert [(f.source, f.result) for f in timeline.facts] == [
        ('head', 'miss'), ('tail', 'miss'), ('body', 'combo-break')]


def test_hold_parent_note_lock_applies_to_head_and_blocks_resuming_after_next_start():
    timeline = simulate([HoldNote(0, 1000, 2000), Note(0, 2100)],
        [(1000, 1), (1500, 0), (2100, 1), (2150, 0), (2300, 0)])
    assert presses(timeline) == [(1000, 0, 1000, 'geki'), (2100, 0, 2100, 'geki')]
    assert not any(f.source == 'tail' and f.kind == 'increment' for f in timeline.facts)
    assert any(f.source == 'tail' and f.cause == 'ordered-force-miss' for f in timeline.facts)


def test_successful_hold_head_force_misses_prior_tap_using_parent_order():
    timeline = simulate([Note(0, 950), HoldNote(0, 1000, 2000)], [(1000, 1), (2000, 0)])
    assert [(f.source, f.result, f.cause) for f in timeline.facts[:2]] == [
        ('head', 'geki', 'press'), ('tap', 'miss', 'ordered-force-miss')]
    assert timeline.at(1000).combo == 0
    assert timeline.at(2000).combo == 1


@pytest.mark.parametrize('rate,mods', [(1, 0), (1.5, Mod.DT), (.75, Mod.HT)])
def test_audio_time_order_and_combo_survive_rate_scaling_without_integer_rounding(rate, mods):
    timeline = simulate([HoldNote(0, 1001, 2001)], [(1001, 1), (1501, 0), (1601, 1), (2001, 0)],
                         rate=rate, mods=mods)
    assert [f.time_ms for f in timeline.facts] == pytest.approx([1001 / rate, 1501 / rate, 2001 / rate])
    assert [f.kind for f in timeline.facts] == ['increment', 'reset', 'increment']
    assert timeline.reconstructed_max_combo == 1


def test_window_mods_are_applied_before_flooring_in_audio_time():
    assert LazerManiaWindows.build(8, Mod.HR, 1.5).meh == 136.5
    assert LazerManiaWindows.build(8, Mod.EZ, .75).meh == 133.5


def test_facts_and_prefix_states_are_immutable_and_direct_seek_is_frame_rate_independent():
    timeline = simulate([HoldNote(0, 1000, 2000), Note(1, 1800)],
        [(1000, 1), (1250, 0), (1320, 1), (1800, 3), (2000, 0)])
    expected = {t: timeline.at(t) for t in (0, 1000, 1250, 1800, 2000, 2500)}
    for fps in (30, 60, 120):
        for frame in range(int(2500 * fps / 1000)):
            timeline.at(frame * 1000 / fps)
        for when in (2500, 1250, 2000, 0, 1800, 1000):
            assert timeline.at(when) == expected[when]
    assert expected[2000].combo == expected[2000].running_max == 2
    with pytest.raises(FrozenInstanceError):
        timeline.facts[0].kind = 'reset'
    with pytest.raises(FrozenInstanceError):
        timeline.states[0].combo = 100


def test_no_input_produces_misses_and_body_breaks_without_combo_increases():
    timeline = simulate([Note(0, 1000), HoldNote(1, 1000, 2000)])
    assert [f.result for f in timeline.facts] == ['miss', 'miss', 'miss', 'combo-break']
    assert timeline.reconstructed_max_combo == 0


def test_sparse_two_note_automatic_miss_precedes_1300_press_and_combo_is_one():
    timeline = simulate([Note(0, 1000), Note(0, 1300)], [(1300, 1)])
    assert [(f.time_ms, f.object_time_ms, f.result, f.cause) for f in timeline.facts] == [
        (1137, 1000, 'miss', 'timeout'), (1300, 1300, 'geki', 'press')]
    assert timeline.at(1136.5).combo == 0
    assert timeline.at(1300).combo == 1
    assert timeline.at(1300).running_max == 1


def test_deadline_between_replay_frames_runs_before_later_other_column_input():
    timeline = simulate([Note(0, 1000), Note(1, 2000)], [(900, 0), (2000, 2)])
    assert [(f.time_ms, f.column, f.cause, f.input_frame_index) for f in timeline.facts] == [
        (1137, 0, 'timeout', None), (2000, 1, 'press', 1)]
    assert timeline.at(2000).combo == 1


def test_press_exactly_at_its_own_miss_deadline_is_consumed_before_non_user_expiry():
    timeline = simulate([Note(0, 1000)], [(1137, 1)])
    assert [(f.time_ms, f.result, f.cause, f.input_frame_index) for f in timeline.facts] == [
        (1137, 'miss', 'press', 0)]


def test_input_at_another_columns_deadline_precedes_expiry_and_reset():
    timeline = simulate([Note(0, 1000), Note(1, 1137)], [(1137, 2)])
    assert [(f.time_ms, f.column, f.result, f.cause) for f in timeline.facts] == [
        (1137, 1, 'geki', 'press'), (1137, 0, 'miss', 'timeout')]
    assert [s.combo for s in timeline.states] == [1, 0]


def test_equal_time_frames_keep_expiry_after_each_original_input_dispatch():
    timeline = simulate([Note(0, 1000), Note(0, 1300)], [(1137, 0), (1137, 1)])
    assert [(f.time_ms, f.object_time_ms, f.cause, f.input_frame_index) for f in timeline.facts] == [
        (1137, 1000, 'timeout', None), (1137, 1300, 'press', 1)]


def test_dense_integer_replay_updates_preserve_round36b_result_facts():
    edges = [(now, 1 if 1000 <= now < 1040 or 1300 <= now < 1340 else 0)
             for now in range(900, 1501)]
    timeline = simulate([Note(0, 1000), Note(1, 1000), Note(0, 1300)], edges)
    assert [(f.time_ms, f.column, f.object_time_ms, f.result, f.cause, f.input_frame_index)
            for f in timeline.facts] == [
        (1000, 0, 1000, 'geki', 'press', 100),
        (1137, 1, 1000, 'miss', 'timeout', None),
        (1300, 0, 1300, 'geki', 'press', 400)]
    assert [s.combo for s in timeline.states] == [1, 0, 1]


def test_ordered_hit_policy_does_not_emit_a_previously_timed_out_note_again():
    timeline = simulate([Note(0, 1000), Note(0, 1300), Note(0, 1400)],
                        [(1300, 1), (1350, 0), (1400, 1)])
    assert len([f for f in timeline.facts if f.object_time_ms == 1000]) == 1
    assert timeline.facts[0].cause == 'timeout'
    assert not any(f.cause == 'ordered-force-miss' for f in timeline.facts)
    assert timeline.at(1400).combo == 2


def test_hold_head_timeout_before_later_resume_does_not_finish_parent():
    timeline = simulate([HoldNote(0, 1000, 2000)], [(1500, 1), (2000, 0)])
    assert [(f.time_ms, f.source, f.result, f.cause) for f in timeline.facts] == [
        (1137, 'head', 'miss', 'timeout'), (2000, 'tail', '50', 'release')]
    assert timeline.at(1500).combo == 0
    assert timeline.at(2000).combo == 1


def test_hold_tail_deadline_between_frames_finalises_tail_and_body_once():
    timeline = simulate([HoldNote(0, 1000, 2000), Note(1, 2500)],
                        [(1000, 1), (2500, 3), (2600, 0)])
    assert [(f.time_ms, f.source, f.result) for f in timeline.facts] == [
        (1000, 'head', 'geki'), (2205, 'tail', 'miss'),
        (2205, 'body', 'combo-break'), (2500, 'tap', 'geki')]
    assert timeline.at(2204.75).combo == 1
    assert timeline.at(2205).combo == 0
    assert timeline.at(2500).combo == 1
    assert len([f for f in timeline.facts if f.source == 'body']) == 1


def test_tail_timeout_does_not_repeat_an_already_broken_body_result():
    timeline = simulate([HoldNote(0, 1000, 2000)], [(1000, 1), (1250, 0), (2500, 0)])
    assert [(f.time_ms, f.source, f.result) for f in timeline.facts] == [
        (1000, 'head', 'geki'), (1250, 'body', 'combo-break'), (2205, 'tail', 'miss')]


def test_logical_timeout_timeline_direct_and_backward_seeks_are_immutable():
    timeline = simulate([Note(0, 1000), Note(0, 1300), Note(0, 1600)],
                        [(1300, 1), (1400, 0), (1900, 0)])
    facts, states = timeline.facts, timeline.states
    for fps in (30, 60, 120):
        for frame in range(60 * fps):
            timeline.at(frame * 1000 / fps)
        assert [timeline.at(t).combo for t in (1900, 1300, 1736.5, 1137, 1737, 0)] == [0, 1, 1, 0, 0, 0]
        assert timeline.at(1900).last_break_ms == 1737
    assert timeline.facts is facts and timeline.states is states


@pytest.mark.parametrize('rate,mods,deadline', [(1, 0, 1138), (1.5, Mod.DT, 1206), (.75, Mod.HT, 1104)])
def test_rate_mod_head_deadlines_are_ordered_in_audio_time_before_scaling(rate, mods, deadline):
    timeline = simulate([Note(0, 1001), Note(0, 1302)], [(1302, 1)], rate=rate, mods=mods)
    assert [(f.source, f.result, f.cause) for f in timeline.facts] == [
        ('tap', 'miss', 'timeout'), ('tap', 'geki', 'press')]
    assert [f.time_ms for f in timeline.facts] == pytest.approx([deadline / rate, 1302 / rate])
    assert timeline.at(1302 / rate).combo == 1


@pytest.mark.parametrize('rate,mods,deadline', [(1, 0, 2205), (1.5, Mod.DT, 2307), (.75, Mod.HT, 2154)])
def test_rate_mod_tail_deadlines_keep_release_lenience_and_body_order(rate, mods, deadline):
    timeline = simulate([HoldNote(0, 1000, 2000), Note(1, 2500)],
                        [(1000, 1), (2500, 3)], rate=rate, mods=mods)
    assert [(f.source, f.result) for f in timeline.facts] == [
        ('head', 'geki'), ('tail', 'miss'), ('body', 'combo-break'), ('tap', 'geki')]
    assert [f.time_ms for f in timeline.facts] == pytest.approx([
        1000 / rate, deadline / rate, deadline / rate, 2500 / rate])
    assert timeline.at(2500 / rate).combo == 1
