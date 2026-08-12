from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def add_sys_paths(paths: Iterable[str | Path]) -> None:
    for raw_path in paths:
        path = str(Path(raw_path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)

