# Notes for V3

Running list of issues / design intents to address in the V3 mania renderer
rewrite. Not v2 TODOs — things deliberately deferred to V3.

## Rendering fidelity

### Judgement / hitburst sprites stretched for custom skins (aspect ratio not preserved)
- **Symptom:** hit-judgement popups (`mania-hit300` / `hit100` / `hit50` / `hit0`)
  and the hitburst/effect layer render **stretched and distorted** for some custom
  skins — smeared across the playfield instead of a proportional popup. On a normal
  skin it looks fine, so it is skin-dependent.
- **Reported:** forum topic `/forum/bugs/15` (Shiro253, "That Skin looking like
  playing a Horror game"). Render `01M1EK0ZVYDVGTCHN207YAQ1H0` — Camellia -
  Parallel Universe Shifter [Homeworld], mode 3 mania 4K, skin **"OT!skin collab (v1.0.1)"**.
- **Root-cause hypothesis (v2):** the judgement/hitburst popup is scaled to a fixed
  box (column-width or a fixed target) **without preserving the sprite's native
  aspect ratio**, so a skin whose `mania-hit300.png` has a non-standard aspect gets
  smeared to fit.
- **V3 requirement:** scale judgement + hitburst popups **aspect-preserving** — pick
  a target height, derive width proportionally (match osu!stable / lazer behaviour).
  Honour `@2x` assets and `score_position` (Y of hitburst popups) from `skin.ini`.
- **Where (v2 refs):** `gpu/atlas.py` (`judgment_300` → `mania-hit300.png`, etc.);
  `skin_ini.py` `score_position` / `hit_position`.
- **Verify:** re-render the map+skin above and diff against a lazer reference; also
  cross-check std/catch/taiko judgement scaling for parity.
- **Severity:** low (cosmetic, custom skin only).
