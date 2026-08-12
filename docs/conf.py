from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
html_theme = "furo"
html_title = "SMITH"
master_doc = "docs/source/index" if os.environ.get("READTHEDOCS") else "index"
root_doc = master_doc
exclude_patterns = [
    "_build/**",
    ".pytest_cache/**",
    "build/**",
    "data/**",
    "manifests/**",
    "outputs/**",
    "reproducibility/**",
    "scripts/**",
    "src/**",
    "tests/**",
    "README.md",
]
myst_enable_extensions = ["colon_fence", "deflist"]
