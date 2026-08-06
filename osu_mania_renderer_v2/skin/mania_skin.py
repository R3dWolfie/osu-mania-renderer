"""Mania skin provider + skin.ini resolution.

The single object every asset/variable lookup goes through (parity with
taiko's ``skin/taiko_skin.py::TaikoSkin`` and catch's ``skin/skin.py``).
Relocated verbatim from the compositor; ``SkinPair`` kept as a back-compat
alias for ``ManiaSkin``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Sentinel: a variable spec WITHOUT a wiki_default forces a RenderError
# when the skin doesn't supply it. Compare with `is MISSING` only.
MISSING: Any = object()


class RenderError(RuntimeError):
    """Raised when a required value cannot be resolved anywhere.

    Carries `element`, `variable`, and the ordered `searched` list so a
    log line is enough to diagnose without re-running the render."""

    def __init__(
        self,
        *,
        element: str,
        variable: str,
        searched: list[str],
        hint: str | None = None,
    ) -> None:
        self.element = element
        self.variable = variable
        self.searched = list(searched)
        self.hint = hint
        msg = (
            f"render error: cannot resolve '{variable}' for element "
            f"'{element}'. Looked in: " + " | ".join(searched)
        )
        if hint:
            msg += f" — {hint}"
        super().__init__(msg)


class SkinIni:
    """Tolerant skin.ini reader. Sections + 'Key: value' lines, with
    repeated [Mania] blocks bucketed by `Keys:` count so per-keymode
    overrides resolve cleanly."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._sections: dict[str, dict[str, str]] = {}
        self._mania_by_keys: dict[int, dict[str, str]] = {}
        if path is not None and path.is_file():
            self._parse(path.read_text(encoding="utf-8", errors="replace"))

    def _parse(self, text: str) -> None:
        section = "_root"
        cur = self._sections.setdefault(section, {})
        mania_blocks: list[dict[str, str]] = []
        for raw in text.splitlines():
            line = raw.split("//", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                name = line[1:-1]
                if name == "Mania":
                    cur = {}
                    mania_blocks.append(cur)
                    section = name
                else:
                    section = name
                    cur = self._sections.setdefault(name, {})
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                cur[k.strip()] = v.strip()
        for block in mania_blocks:
            try:
                k = int(block.get("Keys", "0"))
            except ValueError:
                continue
            if k > 0:
                self._mania_by_keys[k] = block

    def get(self, key: str, *, section: str | None = None) -> str | None:
        target = "_root" if section is None else section
        return self._sections.get(target, {}).get(key)

    def get_mania(self, keys: int, key: str) -> str | None:
        return self._mania_by_keys.get(keys, {}).get(key)


@dataclass
class ManiaSkin:
    """User skin + default skin, with skin.ini files pre-parsed.
    Every asset and variable lookup in the pipeline goes through this
    object so the fall-back order is consistent."""

    user_dir: Path
    default_dir: Path
    user_ini: SkinIni = field(init=False)
    default_ini: SkinIni = field(init=False)

    def __post_init__(self) -> None:
        self.user_ini = SkinIni(self.user_dir / "skin.ini")
        self.default_ini = SkinIni(self.default_dir / "skin.ini")

    def resolve_asset(
        self, basename: str, *, extensions: tuple[str, ...] = (".png", ".jpg"),
    ) -> Path | None:
        """user skin → default skin → None. Honors the @2x high-DPI suffix
        because the wiki treats `name@2x.png` as the preferred variant
        when present."""
        for root in (self.user_dir, self.default_dir):
            for stem in (f"{basename}@2x", basename):
                for ext in extensions:
                    p = root / f"{stem}{ext}"
                    if p.is_file():
                        return p
        return None

    def resolve_variable(
        self,
        *,
        element: str,
        key: str,
        section: str | None = None,
        mania_keys: int | None = None,
        wiki_default: Any = MISSING,
    ) -> Any:
        """user skin.ini → default skin.ini → wiki_default → RenderError.
        Pass `mania_keys=N` to read from the [Mania] block for that key
        count instead of a named section."""
        searched: list[str] = []
        for ini, label in (
            (self.user_ini, "user"),
            (self.default_ini, "default"),
        ):
            if mania_keys is not None:
                searched.append(f"{label}.ini [Mania Keys={mania_keys}].{key}")
                val = ini.get_mania(mania_keys, key)
            else:
                where = section or "root"
                searched.append(f"{label}.ini [{where}].{key}")
                val = ini.get(key, section=section)
            if val is not None:
                return val
        searched.append("wiki default")
        if wiki_default is not MISSING:
            return wiki_default
        raise RenderError(
            element=element, variable=key, searched=searched,
            hint="add a wiki_default to the VariableSpec if this is intended",
        )


# Back-compat alias — the provider was formerly named SkinPair.
SkinPair = ManiaSkin
