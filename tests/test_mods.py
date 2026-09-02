from osu_mania_renderer_v2.models import BeatmapInfo, HoldNote, Note, ReplayInfo
from osu_mania_renderer_v2.mods import Mod, apply_mods


def _mk_beatmap(notes=None, key_count=4) -> BeatmapInfo:
    return BeatmapInfo(
        key_count=key_count,
        notes=tuple(notes or []),
        audio_filename="audio.mp3",
        background_filename=None,
        total_duration_ms=10_000,
        audio_lead_in_ms=0,
        artist="A",
        title="T",
        difficulty="V",
        creator="C",
        beatmap_id=None,
        beatmapset_id=None,
    )


def _mk_replay(mods: int = 0) -> ReplayInfo:
    return ReplayInfo(
        mode=3,
        beatmap_md5="",
        player_name="P",
        replay_md5="",
        mods=mods,
        key_events=(),
        score=0,
        accuracy=100.0,
        max_combo=0,
        count_geki=0,
        count_300=0,
        count_katu=0,
        count_100=0,
        count_50=0,
        count_miss=0,
        grade="SS",
    )


def test_no_mods_passthrough():
    bm = _mk_beatmap(notes=[Note(0, 1000), Note(1, 2000)])
    rp = _mk_replay(mods=0)
    res = apply_mods(bm, rp)
    assert res.audio_rate == 1.0
    assert res.beatmap.notes == bm.notes
    assert res.warnings == ()
    assert res.visual_mods.hidden is False


def test_dt_halves_note_times():
    bm = _mk_beatmap(notes=[Note(0, 3000)])
    rp = _mk_replay(mods=Mod.DT.value)
    res = apply_mods(bm, rp)
    assert res.audio_rate == 1.5
    assert res.beatmap.notes[0].time_ms == 2000  # 3000 / 1.5


def test_rate_override_beats_dt_bitmask():
    """Lazer rate-adjusted DT (--rate): the TRUE clock multiplier must
    override the bitmask-implied 1.5 for gameplay/audio timing."""
    bm = _mk_beatmap(notes=[Note(0, 2900)])
    rp = _mk_replay(mods=Mod.DT.value)
    res = apply_mods(bm, rp, rate_override=1.16)
    assert res.audio_rate == 1.16
    assert res.beatmap.notes[0].time_ms == int(2900 / 1.16)


def test_rate_override_none_keeps_legacy_dt():
    """--rate absent -> byte-identical legacy behaviour (DT stays 1.5)."""
    bm = _mk_beatmap(notes=[Note(0, 3000)])
    rp = _mk_replay(mods=Mod.DT.value)
    res = apply_mods(bm, rp, rate_override=None)
    assert res.audio_rate == 1.5
    assert res.beatmap.notes[0].time_ms == 2000


def test_ht_extends_note_times():
    bm = _mk_beatmap(notes=[Note(0, 3000)])
    rp = _mk_replay(mods=Mod.HT.value)
    res = apply_mods(bm, rp)
    assert res.audio_rate == 0.75
    assert res.beatmap.notes[0].time_ms == 4000  # 3000 / 0.75


def test_nc_treated_as_dt():
    bm = _mk_beatmap(notes=[Note(0, 3000)])
    rp = _mk_replay(mods=Mod.NC.value | Mod.DT.value)
    res = apply_mods(bm, rp)
    assert res.audio_rate == 1.5


def test_mirror_flips_columns():
    bm = _mk_beatmap(notes=[Note(0, 1000), Note(3, 2000)], key_count=4)
    rp = _mk_replay(mods=Mod.MR.value)
    res = apply_mods(bm, rp)
    cols = [n.column for n in res.beatmap.notes]
    assert cols == [3, 0]


def test_hidden_sets_visual_flag():
    bm = _mk_beatmap()
    rp = _mk_replay(mods=Mod.HD.value)
    res = apply_mods(bm, rp)
    assert res.visual_mods.hidden is True


def test_flashlight_sets_visual_flag():
    bm = _mk_beatmap()
    rp = _mk_replay(mods=Mod.FL.value)
    res = apply_mods(bm, rp)
    assert res.visual_mods.flashlight is True


def test_random_warns_falls_back_to_nm():
    bm = _mk_beatmap(notes=[Note(0, 1000), Note(2, 2000)])
    rp = _mk_replay(mods=Mod.RD.value)
    res = apply_mods(bm, rp)
    cols = [n.column for n in res.beatmap.notes]
    assert cols == [0, 2]  # NOT shuffled
    assert any("Random" in w for w in res.warnings)


def test_key_coop_warns():
    bm = _mk_beatmap()
    rp = _mk_replay(mods=Mod.KC.value)
    res = apply_mods(bm, rp)
    assert any("Key Coop" in w for w in res.warnings)


def test_hold_note_end_time_is_also_rescaled():
    bm = _mk_beatmap(notes=[HoldNote(0, 3000, 6000)])
    rp = _mk_replay(mods=Mod.DT.value)
    res = apply_mods(bm, rp)
    h = res.beatmap.notes[0]
    assert h.time_ms == 2000
    assert h.end_time_ms == 4000
