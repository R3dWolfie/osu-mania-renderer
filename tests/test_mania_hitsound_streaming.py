"""Chunk boundaries, overlays and long-duration allocation guards."""
from itertools import islice
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from osu_mania_renderer_v2.beatmap.models import Note, TimingPoint
from osu_mania_renderer_v2.errors import RendererError
from osu_mania_renderer_v2.render import hitsounds as hs
from tests.test_mania_hitsound_pipeline import oracle, fact


def test_chunk_mixer_matches_full_reference_with_overlap_clipping_and_overlay_tails():
    rng = np.random.default_rng(481)
    placements = [hs.SamplePlacement(start, rng.uniform(-.8,.8,(length,2)).astype('f4'), gain, kind)
        for start,length,gain,kind in [(2,300,1,'gameplay'),(71,180,.7,'gameplay'),
            (90,200,1,'combo_break'),(127,320,.3,'nightcore'),(230,120,.8,'nc_mod')]]
    expected = np.zeros((500,2),dtype='f4')
    for p in placements:
        expected[p.start:p.start+len(p.samples)] += p.samples*p.gain
    np.clip(expected,-1,1,out=expected)
    for size in (1,31,128,512):
        actual = np.concatenate([b for b,_ in hs._mix_blocks(iter(placements),500,size)])
        np.testing.assert_array_equal(actual,expected)


def test_six_hour_pcm_memory_is_bounded_by_chunk_not_duration(monkeypatch):
    original = hs.np.zeros
    allocations = []
    def bounded(shape,*args,**kwargs):
        allocations.append(shape)
        assert shape[0] <= 16384
        return original(shape,*args,**kwargs)
    monkeypatch.setattr(hs.np,'zeros',bounded)
    blocks = list(islice(hs._mix_blocks(iter(()),6*3600*44100,16384),3))
    assert len(blocks)==3 and max(b.nbytes for b,_ in blocks)==16384*2*4
    assert allocations == [(16384,2)]*3


def test_long_actual_builder_streams_and_cleans_partial_output(tmp_path, oracle, monkeypatch):
    _,beatmap,_ = oracle
    beatmap = replace(beatmap,notes=(Note(0,100),))
    original = hs.np.zeros
    def bounded(shape,*args,**kwargs):
        assert shape[0] <= 16384
        return original(shape,*args,**kwargs)
    class Sink:
        def __init__(self,*args,**kwargs): self.writes=0
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def write(self,block):
            self.writes+=1
            if self.writes==3: raise OSError('stress sink stop')
    # Decode cache is primed before replacing the writer/reader class.
    cache=hs._SampleCache(44100,beatmap_dir=tmp_path)
    hs._resolve_samples_for_note(beatmap.notes[0],beatmap,cache,is_lazer_replay=True)
    monkeypatch.setattr(hs,'_SampleCache',lambda *args,**kwargs:cache)
    monkeypatch.setattr(hs.np,'zeros',bounded)
    monkeypatch.setattr(sf,'SoundFile',Sink)
    with pytest.raises(RendererError,match='stress sink stop'):
        hs.build_hitsound_track(beatmap=beatmap,beatmap_dir=tmp_path,lazer_facts=(fact(100,100),),
            is_lazer_replay=True,output_wav=tmp_path/'long.wav',duration_ms=6*3600*1000,miss_hitsound=False)
    assert not list(tmp_path.glob('long.wav*'))


@pytest.mark.parametrize('nc_mod',[False,True])
def test_real_nc_overlay_pcm_identical_across_chunk_sizes(tmp_path, oracle, nc_mod):
    _,beatmap,_ = oracle
    # One sample lasts past several chunk boundaries, with a timing change.
    sf.write(tmp_path/'normal-hitclap.wav',np.full((3000,2),.2,dtype='f4'),44100)
    sf.write(tmp_path/'normal-hitfinish.wav',np.full((5000,2),.3,dtype='f4'),44100)
    beatmap=replace(beatmap,notes=(Note(0,120),),timing_points=(
        TimingPoint(0,1,0,10,beat_length_ms=150),TimingPoint(510,1,0,10,beat_length_ms=180)))
    rate=1.5 if nc_mod else 1
    outputs=[]
    for chunk in (127,16384):
        path=hs.build_hitsound_track(beatmap=beatmap,beatmap_dir=tmp_path,
            is_lazer_replay=True,lazer_facts=(fact(120/rate,120/rate),),skin_dirs=(tmp_path,),
            output_wav=tmp_path/f'{chunk}.wav',duration_ms=1200,chunk_frames=chunk,
            nightcore=not nc_mod,nc_mod=nc_mod,audio_rate=rate)
        outputs.append(sf.read(path)[0])
    np.testing.assert_array_equal(*outputs)
    assert np.max(np.abs(outputs[0]))>0
