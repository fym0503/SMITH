from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "SMITH"
author = "Yimin Fan"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]
source_suffix = {".md": "markdown"}
master_doc = "index"
root_doc = "index"
exclude_patterns = ["_build/**"]
html_theme = "furo"
html_title = "SMITH"
myst_enable_extensions = ["colon_fence", "deflist"]
