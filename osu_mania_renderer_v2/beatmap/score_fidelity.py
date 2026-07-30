"""Lazer-standardised (ScoreV3) total for a STABLE osu!mania replay (#115).

Red's decision (#115): the in-video score must be the lazer-standardised
ScoreV3 EVERYWHERE. A LAZER .osr (game_version >= 30_000_000, or the ScoreV2
mod) already stores a 1,000,000-scale standardised total in its header, so it
is used as-is -- the caller GATES on game_version and only calls this module
for STABLE replays. A STABLE .osr stores the legacy ScoreV1 total, which uses
a different scoring model; this module converts it to the standardised total
the player recognises from lazer / the osu! website, using osu!(lazer)'s own
`StandardisedScoreMigrationTools.convertFromLegacyTotalScore`, ruleset 3
(mania) -- the SAME math osu-web runs server-side. (Catch already does this in
its own `beatmap/score_fidelity.py`; this is the mania analogue.)

Why mania is a closed form: `ManiaLegacyScoreSimulator.Simulate` returns a
CONSTANT `LegacyScoreAttributes` -- ComboScore = 1_000_000, and AccuracyScore
/ BonusScore / BonusScoreRatio / MaxCombo all 0 -- so the generic conversion
collapses (accuracyScore and bonusProportion vanish) to:

    legacyMult   = mania legacy mod multiplier (NF/EZ/HT/DC only for a NATIVE
                   mania map; ppy/osu ManiaLegacyScoreSimulator)
    maxComboV1   = round(1_000_000 * legacyMult)
    comboProp    = max(header, 0) / maxComboV1           # accScore/bonus are 0
    withoutMods  = round(150000 * comboProp + 850000 * acc^(2 + 2*acc))
    total        = round(withoutMods * modMultiplier)    # ScoreV3 mod mult

where `acc` is the play's LAZER accuracy (MAX/geki weighted 305) in [0,1] and
`modMultiplier` is the lazer ScoreV3 mod multiplier (render.mods_score_multiplier).
A stable SS (header == maxComboV1, acc == 1) converts to exactly
1_000_000 * modMultiplier -- identical to a lazer SS, which is the invariant
this port is verified against.

Source (ppy/osu master):
  osu.Game/Database/StandardisedScoreMigrationTools.cs
    convertFromLegacyTotalScore (setup + case 3) + the ScoreV2 short-circuit
  osu.Game.Rulesets.Mania/Difficulty/ManiaLegacyScoreSimulator.cs
    Simulate (constant attrs) + GetLegacyScoreMultiplier

CAVEAT: the mania legacy mod multiplier's key-mod term (column-count change)
applies ONLY to std->mania CONVERTED maps; for the native mania maps that are
the overwhelming majority it is skipped (the ppy source early-returns), so
this port implements the native-map multiplier exactly and omits the
converted-map key-mod term (a <=~2% error confined to converted-map + key-mod
plays). Fail-soft: the caller keeps the raw header on any exception.
"""
from __future__ import annotations

# .osr game_version boundary: stable writes 8-digit YYYYMMDD (~2025xxxx),
# lazer local exports write >= 30_000_000. Mirrors replay.py's is_v2 gate.
LAZER_GAME_VERSION_BOUNDARY = 30_000_000

# osu! legacy mod bit flags.
_NF = 1 << 0
_EZ = 1 << 1
_HT = 1 << 8       # HalfTime (Daycore shares this legacy bit)
_SCORE_V2 = 1 << 29

_MAX_SCORE = 1_000_000.0


def mania_legacy_mod_multiplier(mods: int) -> float:
    """ManiaLegacyScoreSimulator.GetLegacyScoreMultiplier, native-map path.
    Only NF/EZ/HT/DC carry a mania legacy multiplier (HD/HR/DT/NC/FL do NOT).
    NF is 1.0 under the ScoreV2 mod, else 0.5."""
    mods = int(mods or 0)
    v2 = bool(mods & _SCORE_V2)
    m = 1.0
    if mods & _NF:
        m *= 1.0 if v2 else 0.5
    if mods & _EZ:
        m *= 0.5
    if mods & _HT:
        m *= 0.5
    return m


def mania_lazer_accuracy(count_geki: int, count_300: int, count_katu: int,
                         count_100: int, count_50: int, count_miss: int) -> float:
    """The play's LAZER accuracy in [0,1] -- the value osu-web feeds the
    conversion. Mania lazer weights: MAX(geki)=305, GREAT(300)=300,
    GOOD(katu)=200, OK(100)=100, MEH(50)=50, MISS=0; max per object = 305."""
    total = (int(count_geki) + int(count_300) + int(count_katu)
             + int(count_100) + int(count_50) + int(count_miss))
    if total <= 0:
        return 1.0
    num = (305 * int(count_geki) + 300 * int(count_300) + 200 * int(count_katu)
           + 100 * int(count_100) + 50 * int(count_50))
    return num / (305.0 * total)


def stable_to_standardised(header_score: int, mods: int, accuracy: float,
                           mod_multiplier: float) -> int:
    """Convert a STABLE mania ScoreV1 header total to the lazer standardised
    (ScoreV3) total. `accuracy` is the LAZER accuracy in [0,1];
    `mod_multiplier` is the ScoreV3 mania mod multiplier
    (render.mods_score_multiplier). See module docstring for the derivation."""
    header = int(header_score)
    if header <= 0:
        return 0

    mods = int(mods or 0)
    legacy_mult = mania_legacy_mod_multiplier(mods)

    # ScoreV2-mod stable scores are already 1M-standardised: divide out the
    # legacy multiplier to get the without-mods total, re-apply the ScoreV3
    # multiplier (StandardisedScoreMigrationTools first overload, l.118-119).
    if mods & _SCORE_V2:
        without_mods = header / legacy_mult if legacy_mult else float(header)
        return int(round(without_mods * float(mod_multiplier)))

    max_combo_v1 = int(round(_MAX_SCORE * legacy_mult))
    # comboProportion = max(header - accScore, 0) / (comboScore + bonusScore);
    # for mania accScore == 0 and bonusScore == 0.
    combo_prop = (max(header, 0) / max_combo_v1) if max_combo_v1 > 0 else 1.0

    acc = max(0.0, min(1.0, float(accuracy)))
    without_mods = round(150000.0 * combo_prop
                         + 850000.0 * (acc ** (2.0 + 2.0 * acc)))
    return int(round(without_mods * float(mod_multiplier)))
