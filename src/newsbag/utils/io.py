from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_manifest(path: Path) -> List[Path]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    out: List[Path] = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        out.append(Path(ln).expanduser())
    return out


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write `text` to `path` atomically (best-effort).

    Torch runs can execute multiple jobs/stages against the same RUN_DIR (e.g. split GPU jobs).
    Atomic writes avoid partial/corrupt files when two processes overwrite metadata concurrently.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_lines(path: Path, rows: Iterable[str]) -> None:
    text = "\n".join(rows)
    if text and not text.endswith("\n"):
        text += "\n"
    _atomic_write_text(path, text, encoding="utf-8")
