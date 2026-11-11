"""Run the packaged utility from a source checkout."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ROOT / "src/smith_agent/resources/scripts"
os.environ.setdefault("SMITH_PACKAGE_ROOT", str(ROOT))
sys.path.insert(0, str(RUNNERS))

if __name__ == "__main__":
    runpy.run_path(str(RUNNERS / Path(__file__).name), run_name="__main__")
