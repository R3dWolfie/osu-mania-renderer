from osu_mania_renderer_v2.models import HoldNote, KeyEvent, Note, VisualMods
from osu_mania_renderer_v2.scene import snapshot

# Playfield occupies the bottom 1000ms of approach time at the receptors.
APPROACH_MS = 600


def test_no_notes_visible_before_any_appear():
    s = snapshot(
        notes=(Note(0, 1000),),
        key_events=(),
        t_ms=0,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    # 1000ms - 600ms approach = note becomes visible at t=400ms.
    assert s.visible_notes == ()


def test_note_becomes_visible_within_approach_window():
    s = snapshot(
        notes=(Note(0, 1000),),
        key_events=(),
        t_ms=500,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    assert len(s.visible_notes) == 1


def test_note_hits_receptor_at_exact_time():
    s = snapshot(
        notes=(Note(0, 1000),),
        key_events=(),
        t_ms=1000,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    n = s.visible_notes[0]
    # Y-position 1.0 means at the receptor.
    assert abs(n.y_fraction - 1.0) < 1e-3


def test_held_keys_reflected():
    s = snapshot(
        notes=(),
        key_events=(KeyEvent(time_ms=500, keys_held=0b0011),),
        t_ms=600,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    assert s.keys_held == (True, True, False, False)


def test_hold_note_renders_as_segment_when_active():
    s = snapshot(
        notes=(HoldNote(0, 1000, 2000),),
        key_events=(),
        t_ms=1500,
        key_count=4,
        approach_ms=APPROACH_MS,
        visual_mods=VisualMods(),
    )
    visible_holds = [n for n in s.visible_notes if n.is_hold]
    assert len(visible_holds) == 1
    # Head has passed the receptor (y > 1), tail is still in the playfield.
    h = visible_holds[0]
    assert h.head_y_fraction > 1.0
    assert h.tail_y_fraction < 1.0
