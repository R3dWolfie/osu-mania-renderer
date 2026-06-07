 # Render Pipeline — Wiki-Driven Design

Your original description was correct. Below is the formalised version:

> 1. gather per frame what needs rendering, back-to-front order
> 2. loop each element, look in uskin for its image file
> 3. if not found, look in default skin
> 4. if not found in either, skip rendering entirely
> 5. if found, call the render function
> 6. render function refers to wiki, notes every variable that modifies rendering
> 7. look for those variables in skin.ini in uskin
> 8. if not found, use wiki's default value
> 9. then draw

## Refinements

### Note A: per-frame wiki lookup is O(N×M) — parse once at startup

The wiki is read **once** at init and parsed into an `ElementSpec` table:

```python
elements: dict[str, ElementSpec] = {}
for md_file in glob("osu-wiki/wiki/Skinning/*.md"):
    spec = parse_skinning_page(md_file)
    elements[spec.element_id] = spec
```

Each `ElementSpec` bundles:
- skin.ini keys to check for image filenames (e.g. `NoteImage1`, `KeyImage1`)
- skin.ini variables that affect rendering, with per-variable defaults
- z-order (draw layer)
- per-instance layout rules (how many, where, sizing/anchor)
- background beatmap image is loaded separately from beatmap dir

### Note B: persistent element array

Elements that persist across frames (hold bodies, judgment popups, stage
lights, hit lighting) live in a mutable array. Each frame, update their
state in place (position, opacity, animation frame, scale) rather than
rebuilding from scratch. Add new elements when they spawn, remove when
they expire.

### Note C: image paths from skin.ini

The wiki says image filenames are defined per-column in skin.ini keys.
Resolution order for a given column index:

1. `NoteImage{col+1}` / `KeyImage{col+1}` in uskin skin.ini
2. conventional filename (e.g. `mania-note1.png`) in uskin directory
3. `NoteImage{col+1}` / `KeyImage{col+1}` in default skin skin.ini
4. conventional filename in default skin directory
5. skip — if even the default skin doesn't have it, osu! doesn't render it

### Note D: image resource fallback

```
uskin/{path} → default-skin/{path} → skip (if neither exists)
```

The default skin is the legacy baked-in osu skin extracted from the
bootstrapped `.osk`. Its skin.ini values *are* the wiki defaults — there
is no separate tier.

### Note E: skin.ini variable fallback

```
uskin skin.ini → wiki default (= default skin value)
```

### Note F: score, combo, accuracy, PP are sprites

These use digit-by-digit sprite composition, just like in-game. The
remaining HUD elements (HP bar, progress bar, UR bar) will be custom-rendered
later to match the in-game look.

### Note G: procedural elements (do later)

- HP bar
- progress bar
- UR bar

---

## Final Pipeline

```
STARTUP (once)
  ├── parse wiki skinning pages → dict[element_id, ElementSpec]
  ├── load uskin skin.ini
  ├── index uskin image files by relative path
  ├── index default-skin image files by relative path
  └── init empty persistent render-element array

PER FRAME
  │
  ├── 1. COMPUTE FRAME STATE
  │     For each element type, determine which instances exist this frame.
  │     Update the persistent array: spawn new, update existing, expire old.
  │     Each instance carries: element_id, column (if columnar), resolved
  │     texture, src_rect, position, opacity, scale, animation_frame, etc.
  │
  ├── 2. SORT INSTANCES
  │     Sort by z-order (back-to-front). Z-order is per-element-type,
  │     defined in the wiki spec (background=0, hold bodies=50, HUD=100…)
  │
  └── 3. FOR EACH INSTANCE:
        │
        ├── 3a. RESOLVE IMAGE PATH
        │     Look up skin.ini for this element + column → image filename.
        │     If skin.ini has no entry, use the wiki's default filename.
        │
        ├── 3b. FIND IMAGE FILE
        │     uskin/{filename} → default-skin/{filename} → skip
        │     (skipping is safe — default skin has everything osu! renders)
        │
        ├── 3c. RESOLVE VARIABLES
        │     For each skin.ini variable the wiki says affects this element:
        │       uskin skin.ini → wiki default
        │
        └── 3d. DRAW
              Upload resolved texture data to GPU if not already resident.
              Draw quad/instance with all resolved parameters.
```
