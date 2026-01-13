from __future__ import annotations

from pathlib import Path


def load_kb_text(kb_path: str) -> str:
    return Path(kb_path).read_text(encoding="utf-8")
