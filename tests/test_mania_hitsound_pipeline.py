"""Round 3.7 audio oracle, source gains, dependency failures and final MP4 mix.

The replay/map are the unchanged Round 3.6b fixtures (MD5 37fa653e...).
audio110-first4s.flac is a lossless PCM excerpt decoded from the supplied
/home/theaussie/Downloads/test/beatmap3/audio110.mp3 with ffmpeg, 0..4 seconds,
stereo 44100 Hz. Keeping only the bounded excerpt makes the mix test portable.
"""
import asyncio
import importlib
import shutil
import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from osu_mania_renderer_v2.beatmap.beatmap import parse_beatmap
from osu_mania_renderer_v2.beatmap.judgments import JudgmentEvent
from osu_mania_renderer_v2.beatmap.models import HitSample, HoldNote, KeyEvent, Note, RenderOptions, TimingPoint
from osu_mania_renderer_v2.beatmap.replay import parse_replay
from osu_mania_renderer_v2.errors import RendererError
from osu_mania_renderer_v2.render import hitsounds as hs
from osu_mania_renderer_v2.render import render as rendering
from osu_mania_renderer_v2.render.encode import build_ffmpeg_cmd
from osu_mania_renderer_v2.render.lazer_mania_combo import LazerComboFact, build_lazer_combo_timeline

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / 'tests/fixtures/lazer_combo_oracle'
REPLAY = ORACLE / '01M45P6TFG83ZNP9G2GRSZVB2N.osr'
MAP = ORACLE / 'Saikoro - far in the blue sky... (1nar) [42 [1.1]].osu'
SONG = ROOT / 'tests/fixtures/mania_hitsound_oracle/audio110-first4s.flac'


@pytest.fixture(scope='module')
def oracle():
    replay, beatmap = parse_replay(REPLAY), parse_beatmap(MAP)
    timeline = build_lazer_combo_timeline(beatmap.notes, replay.ordered_key_events,
                                         beatmap.key_count, od=beatmap.overall_difficulty)
    return replay, beatmap, timeline


def build(tmp_path, beatmap, facts, **kwargs):
    return hs.build_hitsound_track(beatmap=beatmap, beatmap_dir=tmp_path,
        output_wav=tmp_path / 'hits.wav', duration_ms=kwargs.pop('duration_ms', 1000),
        lazer_facts=facts, is_lazer_replay=True, miss_hitsound=False, **kwargs)


def fact(time, object_time, *, kind='increment', source='tap', column=0, result='geki'):
    return LazerComboFact(time_ms=time, object_time_ms=object_time, kind=kind,
        source=source, column=column, result=result, cause='press')


def wave(path, value=.5, frames=441):
    sf.write(path, np.full((frames, 2), value, dtype=np.float32), 44100, subtype='FLOAT')


def diagnostic(caplog):
    return next(r for r in reversed(caplog.records) if r.msg in ('hitsound_track_built','hitsound_track_unusable'))


def test_runtime_dependency_and_decodable_bundled_normal():
    metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert 'soundfile>=0.12' in metadata['project']['dependencies']
    assert importlib.util.find_spec('soundfile') is not None
    source, rate = sf.read(hs._DEFAULT_HITSOUND_DIR / 'normal-hitnormal.wav', always_2d=True)
    assert rate > 0 and source.size > 0 and np.max(np.abs(source)) > 0


def test_full_oracle_stereo_pcm_non_silent_and_truthful_counts(tmp_path, caplog, oracle):
    _, beatmap, timeline = oracle
    caplog.set_level('INFO', logger=hs.log.name)
    output = build(tmp_path, beatmap, timeline.facts, duration_ms=beatmap.total_duration_ms + 500)
    assert output == tmp_path / 'hits.wav' and output.is_file()
    info = sf.info(output)
    assert (info.channels, info.samplerate, info.subtype) == (2, 44100, 'PCM_16')
    data, _ = sf.read(output, dtype='float32', always_2d=True)
    peak, rms = float(np.max(np.abs(data))), float(np.sqrt(np.mean(data * data, dtype=np.float64)))
    assert peak > 0 and rms > 0
    log = diagnostic(caplog)
    assert log.eligible_hit_events == 3120
    assert log.resolved_hit_events == log.eligible_hit_events
    assert log.unresolved_hit_events == 0
    assert log.resolved_sample_layers == log.mixed_sample_layers == 3120
    assert log.silent_node_events == 40
    assert log.combo_break_layers == 0
    assert log.peak == pytest.approx(peak) and log.rms == pytest.approx(rms)
    assert timeline.reconstructed_max_combo == 1194
    assert timeline.at(97632).combo == 1194 and timeline.at(97641).combo == 0
    # Before the second hit, contribution must be the source waveform * .10.
    cache = hs._SampleCache(44100, beatmap_dir=tmp_path)
    first = next(n for n in beatmap.notes if n.column == 0 and n.time_ms == 2988)
    layer, = hs._resolve_samples_for_note(first, beatmap, cache, is_lazer_replay=True)
    assert layer.path == hs._DEFAULT_HITSOUND_DIR / 'normal-hitnormal.wav'
    assert layer.gain == .10
    start = int(2987 / 1000 * 44100)
    length = min(len(layer.samples), int((3169 - 2987) / 1000 * 44100))
    np.testing.assert_allclose(data[start:start + length], layer.samples[:length] * .10,
                               atol=1 / 32768)
    assert np.max(np.abs(data[start:start + length])) > np.max(np.abs(layer.samples[:length])) * .022 * 4


def test_consuming_miss_silent_but_next_real_accepted_object_sounds(tmp_path, oracle, caplog):
    _, beatmap, timeline = oracle
    missed = next(f for f in timeline.facts if f.column == 2 and f.object_time_ms == 97797)
    accepted = next(f for f in timeline.facts if f.column == 2 and f.object_time_ms == 97889)
    assert (missed.time_ms, missed.result, missed.kind) == (97641, 'miss', 'reset')
    assert accepted.time_ms == 97900 and accepted.kind == 'increment'
    caplog.set_level('INFO', logger=hs.log.name)
    output = build(tmp_path, beatmap, (missed, accepted), duration_ms=98200, target_sample_rate=8000)
    data, rate = sf.read(output, always_2d=True)
    assert not np.any(data[int(97641 / 1000 * rate):int(97900 / 1000 * rate)])
    assert np.max(np.abs(data[int(97900 / 1000 * rate):])) > 0
    assert diagnostic(caplog).eligible_hit_events == diagnostic(caplog).mixed_sample_layers == 1


@pytest.mark.parametrize('volume,gain', [(0,.05),(1,.05),(4,.05),(5,.05),(10,.10),(100,1)])
def test_lazer_control_point_volume_floor_including_zero_inheritance(oracle, tmp_path, volume, gain):
    _, beatmap, _ = oracle
    beatmap = replace(beatmap, timing_points=(TimingPoint(0, 0, 0, volume),))
    cache = hs._SampleCache(44100, beatmap_dir=tmp_path)
    layer, = hs._resolve_samples_for_note(Note(0, 100), beatmap, cache, is_lazer_replay=True)
    assert layer.gain == gain
    assert layer.path.name == 'normal-hitnormal.wav'  # auto follows [General] Normal


@pytest.mark.parametrize('volume', [0,1,4,5,8,10,100])
def test_stable_floor_layer_coefficients_and_integer_percentages(oracle, tmp_path, volume):
    _, beatmap, _ = oracle
    beatmap = replace(beatmap, timing_points=(TimingPoint(0, 1, 0, volume),))
    cache = hs._SampleCache(44100, beatmap_dir=tmp_path)
    layers = hs._resolve_samples_for_note(Note(0,100,15), beatmap, cache)
    gains = {s.type_name: s.gain for s in layers}
    assert gains == {'normal': int(max(volume,8)*.8)/100,
                     'finish': max(volume,8)/100,
                     'whistle': int(max(volume,8)*.85)/100,
                     'clap': int(max(volume,8)*.85)/100}


@pytest.mark.parametrize('volume', [0,1,10,100])
def test_exact_custom_sample_direct_stable_path_and_lazer_floor(oracle, tmp_path, volume):
    _, beatmap, _ = oracle
    wave(tmp_path / 'custom.wav')
    beatmap = replace(beatmap, timing_points=(TimingPoint(0, 1, 0, volume),))
    cache = hs._SampleCache(44100, beatmap_dir=tmp_path)
    note = Note(0, 100, 14, HitSample(filename='custom.wav'))
    stable, = hs._resolve_samples_for_note(note, beatmap, cache)
    lazer_layers = hs._resolve_samples_for_note(note, beatmap, cache, is_lazer_replay=True)
    assert [layer.type_name for layer in lazer_layers] == ['custom', 'whistle', 'finish', 'clap']
    lazer = lazer_layers[0]
    assert stable.path.name == 'custom.wav' and stable.gain == volume/100
    assert lazer.gain == max(volume,5)/100


def test_fallback_order_index_note_override_corrupt_sample_and_beatmap_toggle(tmp_path, oracle):
    _, beatmap, _ = oracle
    skin = tmp_path / 'skin'; skin.mkdir()
    wave(skin / 'normal-hitnormal.wav', .25)
    wave(tmp_path / 'normal-hitnormal2.wav', .75)
    note = Note(0, 100, hit_sample=HitSample(index=2, volume=37))
    def resolved(enabled=True):
        return hs._resolve_samples_for_note(note, beatmap, hs._SampleCache(44100,
            beatmap_dir=tmp_path, skin_dirs=(skin,), beatmap_hitsounds=enabled),
            is_lazer_replay=True)[0]
    layer = resolved()
    assert layer.path == tmp_path / 'normal-hitnormal2.wav' and layer.gain == .37
    assert resolved(False).path == skin / 'normal-hitnormal.wav'
    (tmp_path / 'normal-hitnormal2.wav').write_bytes(b'not audio')
    assert resolved().path == skin / 'normal-hitnormal.wav'
    (skin / 'normal-hitnormal.wav').unlink()
    assert resolved().path == hs._DEFAULT_HITSOUND_DIR / 'normal-hitnormal.wav'
    wave(tmp_path / 'custom.wav')
    custom = replace(note, hit_sample=HitSample(filename='custom.wav', volume=23))
    disabled = hs._SampleCache(44100, beatmap_dir=tmp_path, beatmap_hitsounds=False)
    layer, = hs._resolve_samples_for_note(custom, beatmap, disabled, is_lazer_replay=True)
    assert layer.path == hs._DEFAULT_HITSOUND_DIR / 'normal-hitnormal.wav' and layer.gain == .23


@pytest.mark.parametrize('explicit', [False,True])
def test_lazer_head_and_intentional_tail_node_samples(tmp_path, oracle, caplog, explicit):
    _, beatmap, _ = oracle
    wave(tmp_path / 'normal-hitnormal.wav')
    wave(tmp_path / 'tail.wav', .25)
    hold = HoldNote(0,100,500, tail_hit_sample=HitSample(filename='tail.wav') if explicit else None)
    beatmap = replace(beatmap, notes=(hold,), key_count=4, timing_points=(TimingPoint(0,1,1,10),))
    timeline = build_lazer_combo_timeline((hold,), (KeyEvent(100,1),KeyEvent(500,0)), 4, od=8)
    caplog.set_level('INFO', logger=hs.log.name)
    data, rate = sf.read(build(tmp_path, beatmap, timeline.facts), always_2d=True)
    assert np.max(data[int(.1*rate):int(.12*rate)]) == pytest.approx(.05, abs=1/32768)
    tail = data[int(.5*rate):int(.52*rate)]
    assert np.max(tail) == pytest.approx(.025 if explicit else 0, abs=1/32768)
    log = diagnostic(caplog)
    assert log.eligible_hit_events == log.resolved_hit_events == log.mixed_sample_layers == 1 + explicit
    assert log.silent_node_events == (0 if explicit else 1)
    assert log.unresolved_hit_events == 0


def test_rate_and_mirror_metadata_uses_raw_control_point_clock(tmp_path, oracle):
    _, beatmap, _ = oracle
    wave(tmp_path / 'normal-hitnormal.wav')
    raw = Note(3, 1001, hit_sample=HitSample(volume=37))
    beatmap = replace(beatmap, notes=(Note(3,667),), timing_points=(TimingPoint(0,1,1,10),
        TimingPoint(900,1,1,80)))
    output = build(tmp_path, beatmap, (fact(1002/1.5, 1001/1.5, column=3),),
                   sample_notes=(raw,), audio_rate=1.5)
    data, rate = sf.read(output, always_2d=True)
    assert np.max(data[int(1002/1.5/1000*rate):]) == pytest.approx(.185, abs=1/32768)
    # Without a note override, look up 1001ms, not rounded video time 667ms.
    output = build(tmp_path, beatmap, (fact(1002/1.5, 1001/1.5, column=3),),
                   sample_notes=(replace(raw,hit_sample=HitSample()),), audio_rate=1.5)
    assert np.max(sf.read(output)[0]) == pytest.approx(.4, abs=1/32768)


def test_body_reset_only_combo_break_audio_and_truthful_layer_count(tmp_path, oracle, caplog):
    _, beatmap, _ = oracle
    wave(tmp_path / 'normal-hitnormal.wav')
    wave(tmp_path / 'combobreak.wav', .25)
    notes = tuple(Note(0,100+i*10) for i in range(20))
    facts = tuple(fact(n.time_ms,n.time_ms) for n in notes) + (
        fact(500,500,kind='reset',source='body',result='combo_break'),)
    beatmap = replace(beatmap, notes=notes, timing_points=tuple(replace(tp,custom_index=1) for tp in beatmap.timing_points))
    caplog.set_level('INFO', logger=hs.log.name)
    output = hs.build_hitsound_track(beatmap=beatmap, beatmap_dir=tmp_path, lazer_facts=facts,
        is_lazer_replay=True, output_wav=tmp_path/'hits.wav',duration_ms=1000)
    data, rate = sf.read(output, always_2d=True)
    assert np.max(data[int(.5*rate):]) == pytest.approx(.25*hs.COMBO_BREAK_GAIN, abs=1/32768)
    assert diagnostic(caplog).combo_break_layers == 1
    assert diagnostic(caplog).mixed_sample_layers == 20


@pytest.mark.parametrize('failure', ['unresolved','silent'])
def test_resolution_failure_is_distinct_from_authored_silence_with_nc_overlay(tmp_path, oracle, caplog, monkeypatch, failure):
    _, beatmap, _ = oracle
    beatmap = replace(beatmap, notes=(Note(0,100),), timing_points=tuple(replace(tp,custom_index=1) for tp in beatmap.timing_points))
    if failure == 'unresolved':
        monkeypatch.setattr(hs._SampleCache,'get',lambda *_,**__: None)
    else:
        wave(tmp_path / 'normal-hitnormal.wav', 0)
    def overlay(*args,**kwargs):
        yield hs.SamplePlacement(0, np.full((10,2),.5,dtype=np.float32), 1, 'nightcore')
    monkeypatch.setattr(hs,'_nightcore_layers',overlay)
    caplog.set_level('INFO', logger=hs.log.name)
    if failure == 'unresolved':
        with pytest.raises(RendererError, match='no valid gameplay samples'):
            build(tmp_path, beatmap, (fact(100,100),), nightcore=True)
    else:
        output=build(tmp_path, beatmap, (fact(100,100),), nightcore=True)
        data,_=sf.read(output)
        assert np.max(data)==.5 and not np.any(data[10:])  # only the overlay sounds
    log = diagnostic(caplog)
    assert log.eligible_hit_events == log.resolved_hit_events + log.unresolved_hit_events == 1
    assert log.resolved_hit_events == (0 if failure == 'unresolved' else 1)
    assert (tmp_path/'hits.wav').exists() is (failure == 'silent')


def test_stable_trigger_timing_and_gain_remain_separate(tmp_path, oracle):
    _, beatmap, _ = oracle
    wave(tmp_path / 'normal-hitnormal.wav')
    hold = HoldNote(0,100,500,tail_hit_sample=HitSample(filename='tail.wav'))
    beatmap = replace(beatmap,notes=(hold,), timing_points=tuple(replace(tp,custom_index=1) for tp in beatmap.timing_points))
    output = hs.build_hitsound_track(beatmap=beatmap,beatmap_dir=tmp_path,output_wav=tmp_path/'stable.wav',
        duration_ms=1000, judgments_events=(JudgmentEvent(100,0,'geki',11),
            JudgmentEvent(500,0,'geki',0,is_tail=True)),miss_hitsound=False)
    data, rate = sf.read(output)
    assert not np.any(data[:int(.111*rate)])
    assert np.max(data) == pytest.approx(.5*.08,abs=1/32768)
    assert not np.any(data[int(.5*rate):])  # preserve existing stable tail silence


def test_stable_direct_custom_volume_zero_is_intentional_silence(tmp_path, oracle, caplog):
    _, beatmap, _ = oracle
    wave(tmp_path/'custom.wav')
    beatmap = replace(beatmap, notes=(Note(0,100,hit_sample=HitSample(filename='custom.wav')),),
                      timing_points=(TimingPoint(0,1,0,0),))
    caplog.set_level('INFO',logger=hs.log.name)
    output = hs.build_hitsound_track(beatmap=beatmap,beatmap_dir=tmp_path,
        output_wav=tmp_path/'stable.wav',duration_ms=1000,miss_hitsound=False,
        judgments_events=(JudgmentEvent(100,0,'geki',0),))
    assert not np.any(sf.read(output)[0])
    log = diagnostic(caplog)
    assert log.zero_gain_hit_events == log.resolved_hit_events == 1
    assert log.unresolved_hit_events == 0


def unavailable(monkeypatch):
    original = hs.importlib.import_module
    def missing(name,*args,**kwargs):
        if name == 'soundfile':
            raise ModuleNotFoundError('simulated missing soundfile')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(hs.importlib,'import_module',missing)


def plan_setup(monkeypatch, oracle):
    replay, beatmap, _ = oracle
    async def encoder(*_):
        return 'libx264'
    monkeypatch.setattr(rendering,'probe_encoder',encoder)
    monkeypatch.setattr(rendering,'parse_beatmap',lambda *a,**kw: replace(beatmap,audio_filename=str(SONG)))
    return replay


def plan(tmp_path, options):
    return asyncio.run(rendering.build_render_plan(osr_path=REPLAY,beatmap_dir=ORACLE,
        output_path=tmp_path/'oracle.mp4',options=options))


@pytest.mark.parametrize('mode',['replay','nightcore','nc_mod'])
def test_missing_decoder_fails_actual_plan_instead_of_song_only(tmp_path, monkeypatch, oracle, mode):
    replay = plan_setup(monkeypatch,oracle)
    if mode == 'nc_mod':
        monkeypatch.setattr(rendering,'parse_replay',lambda _:replace(replay,mods=512))
    unavailable(monkeypatch)
    with pytest.raises(RendererError, match="pip install 'soundfile>=0.12'"):
        plan(tmp_path,RenderOptions((16,16),10,encoder='libx264',normalize_loudness=False,
            use_replay_hitsounds=mode=='replay',nightcore_hitsounds=mode=='nightcore'))


def test_requested_audio_checks_dependency_even_when_song_is_absent(tmp_path, monkeypatch, oracle):
    plan_setup(monkeypatch,oracle)
    monkeypatch.setattr(rendering,'parse_beatmap',lambda *a,**kw: oracle[1])
    unavailable(monkeypatch)
    with pytest.raises(RendererError, match="pip install 'soundfile>=0.12'"):
        plan(tmp_path,RenderOptions((16,16),10,encoder='libx264',normalize_loudness=False))


def test_all_audio_overlays_explicitly_disabled_ignore_missing_decoder_and_use_song_only(tmp_path, monkeypatch, oracle):
    plan_setup(monkeypatch,oracle)
    unavailable(monkeypatch)
    result = plan(tmp_path,RenderOptions((16,16),10,encoder='libx264',normalize_loudness=False,
        use_replay_hitsounds=False,nightcore_hitsounds=False))
    assert not any(str(arg).endswith('.hits.wav') for arg in result.ffmpeg_cmd)
    graph = result.ffmpeg_cmd[result.ffmpeg_cmd.index('-filter_complex')+1]
    assert 'loudnorm=' in graph and 'amix=' not in graph and '[hits]' not in graph


def test_actual_plan_audio_facts_survive_header_reconciliation(tmp_path, monkeypatch, oracle):
    replay = plan_setup(monkeypatch,oracle)
    monkeypatch.setattr(rendering,'parse_replay',lambda _:replace(replay,max_combo=0,
        count_geki=0,count_300=0,count_katu=0,count_100=0,count_50=0,count_miss=3207))
    received = {}
    def capture(**kwargs):
        received.update(kwargs)
        return kwargs['output_wav']
    monkeypatch.setattr(rendering,'build_hitsound_track',capture)
    result = plan(tmp_path,RenderOptions((16,16),10,encoder='libx264',normalize_loudness=False))
    assert all(j.judgment=='miss' for j in result.judgment_events)
    assert received['is_lazer_replay'] is True
    assert received['lazer_facts'] == oracle[2].facts
    events = list(hs._lazer_sound_events(received['lazer_facts'],received['sample_notes'],1))
    assert any(e.time_ms==2987 and e.kind=='increment' and e.note is not None for e in events)
    assert next(e for e in events if e.time_ms==97641).kind=='reset'


def command(song, hits, output, volume=.8):
    return build_ffmpeg_cmd(encoder='libx264',encoder_device=None,resolution=(16,16),fps=10,
        audio_path=song,audio_rate=1,audio_lead_in_ms=0,video_bitrate='100k',audio_bitrate='160k',
        output_path=output,total_duration_ms=4000,hitsound_path=hits,hitsound_volume=volume)


def test_ffmpeg_distinct_hit_input_volume_song_only_loudnorm_and_final_mapping(tmp_path):
    cmd = command(SONG,tmp_path/'hits.wav',tmp_path/'out.mp4',.8)
    assert [cmd[i+1] for i,arg in enumerate(cmd[:-1]) if arg=='-i'] == ['pipe:0',str(SONG),str(tmp_path/'hits.wav')]
    graph = cmd[cmd.index('-filter_complex')+1]
    assert '[2:a]volume=0.800[hits]' in graph
    assert '[song][hits]amix=inputs=2' in graph
    assert graph.index('loudnorm=') < graph.index('[song]') < graph.index('amix=')
    assert graph.count('loudnorm=')==1
    assert [cmd[i+1] for i,arg in enumerate(cmd[:-1]) if arg=='-map'] == ['0:v','[aout]']
    song_only = command(SONG,None,tmp_path/'song.mp4')
    assert str(tmp_path/'hits.wav') not in song_only
    assert 'amix' not in song_only[song_only.index('-filter_complex')+1]


@pytest.mark.slow
def test_final_encoded_oracle_song_plus_hits_has_energy_at_source_time(tmp_path, oracle):
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg unavailable')
    _, beatmap, timeline = oracle
    hits = build(tmp_path,beatmap,timeline.facts,duration_ms=4000)
    # A silent hit input gives the song-only reference the same resampling,
    # limiter and AAC path. Also encode the actual disabled-input command.
    silence = tmp_path / 'silence.wav'
    sf.write(silence, np.zeros((4*44100,2),dtype=np.float32),44100,subtype='PCM_16')
    outputs = []
    for name,hit_path in [('disabled',None),('song',silence),('mixed',hits)]:
        output = tmp_path / f'{name}.mp4'
        cmd = command(SONG,hit_path,output,1)
        result = subprocess.run(cmd,input=bytes(16*16*3*40),capture_output=True,timeout=30)
        assert result.returncode==0,result.stderr.decode()
        decoded = subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(output),
            '-vn','-ac','2','-ar','44100','-f','f32le','pipe:1'],capture_output=True,timeout=30,check=True)
        outputs.append(np.frombuffer(decoded.stdout,dtype='<f4').reshape(-1,2))
    disabled, song, mixed = outputs
    assert not np.array_equal(disabled,mixed)
    length = min(len(song),len(mixed))
    # The mix limiter delays 1ms; compensate before comparing AAC outputs.
    candidates = [(np.mean((mixed[shift:int(2.5*44100)+shift] - song[:int(2.5*44100)])**2), shift)
                  for shift in range(0,90)]
    _, shift = min(candidates)
    diff = mixed[shift:length] - song[:length-shift]
    before = diff[int(2.0*44100):int(2.5*44100)]
    hit = diff[int(2.987*44100):int(3.1*44100)]
    assert np.sqrt(np.mean(hit**2)) > .0001
    assert np.sqrt(np.mean(hit**2)) > 3*np.sqrt(np.mean(before**2))
    assert np.sqrt(np.mean(song**2)) > .001 and np.sqrt(np.mean(mixed**2)) > .001
