from osu_mania_renderer_v2.beatmap.judgments import compute_judgments
from osu_mania_renderer_v2.beatmap.models import KeyEvent, Note


def test_perfect_tap_within_window():
    notes = (Note(0, 1000),)
    events = (
        KeyEvent(time_ms=999, keys_held=0),
        KeyEvent(time_ms=1000, keys_held=0b0001),
        KeyEvent(time_ms=1020, keys_held=0),
    )
    result = compute_judgments(notes, events, key_count=4)
    # Sum of all judgments should equal #notes.
    total = (result.count_geki + result.count_300 + result.count_katu
             + result.count_100 + result.count_50 + result.count_miss)
    assert total == 1
    assert result.count_geki == 1  # tapped within ~17ms = perfect (Rainbow 300)


def test_miss_when_key_never_pressed():
    notes = (Note(0, 1000),)
    events = ()
    result = compute_judgments(notes, events, key_count=4)
    assert result.count_miss == 1


def test_combo_breaks_on_miss():
    notes = (Note(0, 1000), Note(0, 2000), Note(0, 3000))
    events = (
        KeyEvent(time_ms=1000, keys_held=0b0001),
        KeyEvent(time_ms=1020, keys_held=0),
        # Miss at 2000
        KeyEvent(time_ms=3000, keys_held=0b0001),
        KeyEvent(time_ms=3020, keys_held=0),
    )
    result = compute_judgments(notes, events, key_count=4)
    assert result.max_combo == 1
