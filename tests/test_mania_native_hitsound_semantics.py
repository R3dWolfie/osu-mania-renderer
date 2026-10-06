"""Native flag/carrier rules and legacy sample clocks from the pinned clients."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
from osu_mania_renderer_v2.beatmap.models import HitSample, HoldNote, Note, TimingPoint
from osu_mania_renderer_v2.render import hitsounds as hs
from tests.test_mania_hitsound_pipeline import MAP, fact, wave


@pytest.fixture
def sources(tmp_path, monkeypatch):
    roots = {name: tmp_path / name for name in ('map', 'skin', 'default')}
    for root in roots.values():
        root.mkdir()
        for bank in ('normal', 'soft', 'drum'):
            for kind in ('normal', 'whistle', 'finish', 'clap'):
                wave(root / f'{bank}-hit{kind}.wav', .125)
    monkeypatch.setattr(hs, '_DEFAULT_HITSOUND_DIR', roots['default'])
    return roots


def chart(note, *, source_mode=3, timing=None):
    return replace(parse_beatmap(MAP), notes=(note,), source_mode=source_mode,
                   timing_points=timing or (TimingPoint(0, 1, 1, 100),))


def resolve(sources, beatmap, *, lazer=False):
    cache = hs._SampleCache(44100, beatmap_dir=sources['map'], skin_dirs=(sources['skin'],))
    return hs._resolve_samples_for_note(beatmap.notes[0], beatmap, cache, is_lazer_replay=lazer)


@pytest.mark.parametrize('lazer', [False, True])
@pytest.mark.parametrize('bits,expected', [
    (0, ['normal']), (1, ['normal']), (2, ['whistle']), (4, ['finish']), (8, ['clap']),
    (10, ['whistle', 'clap']), (12, ['finish', 'clap']),
    (14, ['whistle', 'finish', 'clap']), (9, ['normal', 'clap']),
])
def test_native_mania_plays_only_requested_layers(sources, lazer, bits, expected):
    layers = resolve(sources, chart(Note(0, 100, bits)), lazer=lazer)
    assert [layer.type_name for layer in layers] == expected
    assert [layer.path.name for layer in layers] == [f'normal-hit{name}.wav' for name in expected]


@pytest.mark.parametrize('lazer', [False, True])
def test_converted_carrier_keeps_layered_normal(sources, lazer):
    layers = resolve(sources, chart(Note(0, 100, 8), source_mode=0), lazer=lazer)
    assert [layer.type_name for layer in layers] == ['normal', 'clap']


@pytest.mark.parametrize('valid', [False, True])
def test_stable_custom_replaces_banked_path_only_when_valid(sources, valid):
    custom = sources['map'] / 'custom.wav'
    if valid:
        wave(custom, .25)
    else:
        custom.write_bytes(b'corrupt' * 300)  # not stable's tiny-file silence
    layers = resolve(sources, chart(Note(0, 100, 8, HitSample(filename='custom.wav'))))
    assert [layer.type_name for layer in layers] == (['custom'] if valid else ['clap'])
    assert layers[0].path == (custom if valid else sources['map'] / 'normal-hitclap.wav')


@pytest.mark.parametrize('bits,expected', [
    (0, ['custom']), (8, ['custom', 'clap']),
    (10, ['custom', 'whistle', 'clap']), (14, ['custom', 'whistle', 'finish', 'clap']),
])
def test_lazer_custom_keeps_separately_flagged_additions(sources, bits, expected):
    custom = sources['map'] / 'custom.wav'
    wave(custom, .25)
    sample = HitSample(normal_set=1, addition_set=2, index=2, volume=50, filename='custom.wav')
    for kind in ('whistle', 'finish', 'clap'):
        wave(sources['map'] / f'soft-hit{kind}2.wav', .125)
    layers = resolve(sources, chart(Note(0, 100, bits, sample)), lazer=True)
    assert [layer.type_name for layer in layers] == expected
    assert layers[0].path == custom
    assert all(layer.path == sources['map'] / f'soft-hit{layer.type_name}2.wav' for layer in layers[1:])
    assert all(layer.gain == .5 for layer in layers)


def test_lazer_unresolved_custom_primary_does_not_discard_valid_clap(sources):
    for root in sources.values():
        for path in root.glob('*hitnormal*'):
            path.unlink()
    layers = resolve(sources, chart(Note(0, 100, 8, HitSample(filename='missing.wav'))), lazer=True)
    assert [layer.type_name for layer in layers] == ['clap']
    assert layers[0].path == sources['map'] / 'normal-hitclap.wav'


@pytest.mark.parametrize('lazer', [False, True])
def test_valid_silent_custom_obeys_client_layering(sources, lazer):
    wave(sources['map'] / 'custom.wav', 0)
    layers = resolve(sources, chart(Note(0, 100, 8, HitSample(filename='custom.wav'))), lazer=lazer)
    assert [layer.type_name for layer in layers] == (['custom', 'clap'] if lazer else ['custom'])
    assert not np.any(layers[0].samples)


@pytest.mark.parametrize('lazer', [False, True])
@pytest.mark.parametrize('new_point', [103, 105])
def test_native_tap_sample_point_is_stable_plus2_lazer_plus5(sources, lazer, new_point):
    # Include a third point just outside lazer's leniency to prove its boundary.
    timing = (TimingPoint(0, 1, 1, 25), TimingPoint(new_point, 2, 2, 80), TimingPoint(106, 3, 1, 100))
    wave(sources['map'] / 'soft-hitnormal2.wav', .5)
    layer, = resolve(sources, chart(Note(0, 100), timing=timing), lazer=lazer)
    assert layer.path.name == ('soft-hitnormal2.wav' if lazer else 'normal-hitnormal.wav')
    assert layer.gain == (.8 if lazer else .2)


@pytest.mark.parametrize('lazer', [False, True])
@pytest.mark.parametrize('new_point', [150, 205])
def test_native_hold_head_sample_point_uses_source_end_only_for_lazer(sources, lazer, new_point):
    timing = (TimingPoint(0, 1, 1, 25), TimingPoint(new_point, 2, 2, 80), TimingPoint(206, 3, 1, 100))
    wave(sources['map'] / 'soft-hitnormal2.wav', .5)
    layer, = resolve(sources, chart(HoldNote(0, 100, 200), timing=timing), lazer=lazer)
    assert layer.path.name == ('soft-hitnormal2.wav' if lazer else 'normal-hitnormal.wav')
    assert layer.gain == (.8 if lazer else .2)


@pytest.mark.parametrize('lazer', [False, True])
@pytest.mark.parametrize('rate', [1, 1.5, .75])
def test_mixed_hold_head_keeps_raw_parent_clock_and_native_tail_silence(sources, lazer, rate, caplog):
    wave(sources['map'] / 'soft-hitnormal2.wav', .5)
    raw = HoldNote(3, 100, 200)
    display = HoldNote(3, int(100 / rate), int(200 / rate))
    beatmap = chart(display, timing=(TimingPoint(0, 1, 1, 25), TimingPoint(150, 2, 2, 80)))
    facts = (fact(100 / rate, 100 / rate, column=3, source='head'),
             fact(200 / rate, 200 / rate, column=3, source='tail'))
    if lazer:
        # Heads retain the original source HoldNote; only explicit tails are synthesized.
        events = list(hs._lazer_sound_events(facts, (raw,), rate))
        assert events[0].note is raw and events[1].silent_node
    caplog.set_level('INFO', logger=hs.log.name)
    output = hs.build_hitsound_track(beatmap=beatmap, beatmap_dir=sources['map'],
        output_wav=sources['map'] / 'out.wav', duration_ms=500, sample_notes=(raw,), audio_rate=rate,
        is_lazer_replay=lazer, lazer_facts=facts,
        stable_sound_facts=(SimpleNamespace(time_ms=100 / rate, object_time_ms=100 / rate, column=3, source='head'),),
        combo_break_sound=False)
    pcm, sample_rate = sf.read(output, always_2d=True)
    start = int(100 / rate / 1000 * sample_rate)
    np.testing.assert_allclose(pcm[start + 10:start + 100], .4 if lazer else .025, atol=1 / 32768)
    assert not np.any(pcm[int(200 / rate / 1000 * sample_rate):])
    diagnostic = next(r for r in caplog.records if r.msg == 'hitsound_track_built')
    assert diagnostic.resolved_sample_layers == 1
    assert diagnostic.silent_node_events == int(lazer)


def test_converted_lazer_clock_is_not_inferred_from_generated_hold_end(sources):
    beatmap = chart(HoldNote(0, 100, 200), source_mode=0,
        timing=(TimingPoint(0, 1, 1, 25), TimingPoint(103, 2, 1, 80)))
    layer, = resolve(sources, beatmap, lazer=True)
    assert layer.path.name == 'normal-hitnormal.wav' and layer.gain == .25
