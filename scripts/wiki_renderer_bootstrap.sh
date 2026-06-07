#!/usr/bin/env bash
# Bootstraps assets/ for wiki_renderer.py:
#   assets/default_skin/   — extracted stable default skin (268 sprite/wav files)
#   assets/wiki_cache/     — sparse clone of ppy/osu-wiki (Skinning + File_formats),
#                            with symlinks + INDEX.md for per-mode lookup
#
# Re-run anytime to refresh the wiki snapshot (idempotent: skips
# extraction if default_skin/ is already populated, runs `git pull` on
# the existing sparse clone).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS="$ROOT/assets"
DEFAULT_SKIN_OSK="${DEFAULT_SKIN_OSK:-}"   # caller can override

mkdir -p "$ASSETS"

# ---- default skin --------------------------------------------------------
if [ ! -d "$ASSETS/default_skin" ] || [ -z "$(ls -A "$ASSETS/default_skin" 2>/dev/null)" ]; then
    if [ -z "$DEFAULT_SKIN_OSK" ] || [ ! -f "$DEFAULT_SKIN_OSK" ]; then
        echo "ERR: need DEFAULT_SKIN_OSK=<path-to-default-skin.osk>" >&2
        echo "     (the stable client's bundled default skin, ~16 MB)" >&2
        exit 2
    fi
    mkdir -p "$ASSETS/default_skin"
    unzip -q -o "$DEFAULT_SKIN_OSK" -d "$ASSETS/default_skin"
    echo "extracted default skin: $(ls "$ASSETS/default_skin" | wc -l) files"
else
    echo "default_skin/ already populated, skipping extract"
fi

# ---- wiki sparse clone --------------------------------------------------
WIKI="$ASSETS/wiki_cache/osu-wiki"
if [ ! -d "$WIKI/.git" ]; then
    mkdir -p "$ASSETS/wiki_cache"
    git clone --depth=1 --filter=blob:none --sparse \
        https://github.com/ppy/osu-wiki.git "$WIKI"
    git -C "$WIKI" sparse-checkout set \
        wiki/Skinning wiki/Beatmap wiki/Client/File_formats \
        wiki/Game_modifier wiki/Storyboard wiki/Gameplay
else
    echo "wiki_cache/osu-wiki exists, pulling latest..."
    git -C "$WIKI" pull --quiet
fi

# ---- symlink layout -----------------------------------------------------
CACHE="$ASSETS/wiki_cache"
SRC="$WIKI/wiki"
mkdir -p "$CACHE/shared" "$CACHE/std" "$CACHE/mania" "$CACHE/taiko" "$CACHE/catch"

link() {
    # ln -sf with explicit cleanup so stale links from prior layouts don't linger
    local src=$1 dst=$2
    rm -f "$dst"
    ln -s "$src" "$dst"
}

link "$SRC/Skinning/en.md"                                  "$CACHE/shared/skinning_overview.md"
link "$SRC/Skinning/skin.ini/en.md"                         "$CACHE/shared/skin_ini_reference.md"
link "$SRC/Skinning/Interface/en.md"                        "$CACHE/shared/interface_hud.md"
link "$SRC/Skinning/Sounds/en.md"                           "$CACHE/shared/sounds.md"
link "$SRC/Skinning/History/en.md"                          "$CACHE/shared/skin_history.md"
link "$SRC/Client/File_formats/osk_(file_format)/en.md"     "$CACHE/shared/osk_format.md"
link "$SRC/Client/File_formats/osr_(file_format)/en.md"     "$CACHE/shared/osr_format.md"
link "$SRC/Client/File_formats/osu_(file_format)/en.md"     "$CACHE/shared/osu_format.md"
link "$SRC/Client/File_formats/osb_(file_format)/en.md"     "$CACHE/shared/osb_format.md"
link "$SRC/Skinning/osu!/en.md"                             "$CACHE/std/skinning.md"
link "$SRC/Skinning/osu!mania/en.md"                        "$CACHE/mania/skinning.md"
link "$SRC/Skinning/osu!taiko/en.md"                        "$CACHE/taiko/skinning.md"
link "$SRC/Skinning/osu!catch/en.md"                        "$CACHE/catch/skinning.md"

echo "wiki cache ready at $CACHE — see INDEX.md for the per-mode read order"
