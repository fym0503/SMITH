from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "SMITH"
author = "Yimin Fan"
release = "0.1.0"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
]
master_doc = "index"
root_doc = "index"
exclude_patterns = [
    "_build/**",
    "**/*_source.ipynb",
    "tutorials/notebooks/wmb_section/**",
    "**/.ipynb_checkpoints",
]
html_theme = "furo"
html_title = "SMITH"
myst_enable_extensions = ["colon_fence", "deflist"]
nb_execution_mode = "off"
nb_merge_streams = True
nb_output_stderr = "remove"
