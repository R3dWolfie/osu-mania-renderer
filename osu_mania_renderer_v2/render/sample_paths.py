"""Windows-authored audio names resolved case-insensitively within one root."""
from pathlib import Path, PureWindowsPath


def is_relative_sample_name(relative: str) -> bool:
    relative = relative.replace("\\", "/")
    return (bool(relative) and not relative.startswith("/") and not PureWindowsPath(relative).drive
            and "\x00" not in relative and ":" not in relative and ".." not in relative.split("/")
            and any(part not in ("", ".") for part in relative.split("/")))


def sample_file(root: Path, relative: str) -> Path | None:
    relative = relative.replace("\\", "/")
    if not is_relative_sample_name(relative):
        return None
    try:
        base = root.resolve(strict=True)
        current = base
        for part in relative.split("/"):
            if part in ("", "."):
                continue
            exact = current / part
            if exact.exists():
                current = exact
            else:
                # Deterministic if an ext4 directory contains case collisions.
                current = next((p for p in sorted(current.iterdir(), key=lambda p: p.name)
                                if p.name.casefold() == part.casefold()), None)
                if current is None:
                    return None
            current = current.resolve(strict=True)
            if not current.is_relative_to(base):
                return None
        return current if current.is_file() else None
    except (OSError, ValueError, RuntimeError):
        return None
