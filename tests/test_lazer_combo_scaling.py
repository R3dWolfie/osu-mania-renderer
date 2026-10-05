"""Source-fact equivalence to 703a4ac and structural preprocessing bounds."""
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import pytest

from osu_mania_renderer_v2.beatmap.models import HoldNote, KeyEvent, Note
from osu_mania_renderer_v2.beatmap.mods import Mod
from osu_mania_renderer_v2.render import lazer_mania_combo as combo


def scenario(seed):
    rng=random.Random(seed)
    notes=[]
    for _ in range(90):
        time=rng.randrange(100,2500)
        column=rng.randrange(4)
        notes.append(HoldNote(column,time,time+rng.randrange(1,700))
                     if rng.randrange(3)==0 else Note(column,time))
    events=tuple(KeyEvent(t,rng.randrange(16)) for t in sorted(rng.randrange(-100,3300) for _ in range(350)))
    rate=(1,1.5,.75)[seed%3]
    mods=(0,int(Mod.HR),int(Mod.EZ))[seed%3]
    return tuple(notes),events,rate,mods


def fact_digest(timeline):
    return hashlib.sha256(json.dumps([asdict(f) for f in timeline.facts],sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize('seed',range(30))
def test_ordered_mixed_hold_facts_match_pre_optimization_source_checkpoint(seed):
    expected=json.loads((Path(__file__).parent/'fixtures/lazer_combo_facts_sha256.json').read_text())[str(seed)]
    notes,events,rate,mods=scenario(seed)
    timeline=combo.build_lazer_combo_timeline(notes,events,4,od=8,mods=mods,rate=rate)
    assert fact_digest(timeline)==expected['sha256']
    assert timeline.reconstructed_max_combo==expected['max_combo']
    assert len(timeline.facts)==expected['facts']


def test_completed_prefixes_are_not_rescanned(monkeypatch):
    visits=0
    end=combo._Object.end.fget
    def counted_end(self):
        nonlocal visits
        visits+=1
        return end(self)
    monkeypatch.setattr(combo._Object,'end',property(counted_end))
    notes=tuple(Note(i%4,1000+i*50) for i in range(4000))
    events=tuple(e for note in notes for e in (KeyEvent(note.time_ms,1<<note.column),KeyEvent(note.time_ms+1,0)))
    timeline=combo.build_lazer_combo_timeline(notes,events,4,od=8)
    assert timeline.reconstructed_max_combo==len(notes)
    assert visits < 5*len(notes)  # old force_earlier alone performs ~N² / (2*columns)
