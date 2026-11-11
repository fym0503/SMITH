#!/usr/bin/env python3
"""Validate tutorial sources or execute them to refresh the published notebooks."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "docs/source/tutorials/notebooks"
TUTORIALS = {
    "02_regulatory_activity": "regulatory_section/02_SMITH_Regulatory_Activity",
    "03_ribomap_transfer": "ribomap_section/03_SMITH_RIBOMap_Transfer",
    "05_agent": "agent_section/05_SMITH_Agent_Evaluation",
}


def validate_pair(source, executed):
    """Check that the rendered notebook corresponds to its editable source."""
    nbformat.validate(source)
    nbformat.validate(executed)
    source_cells = [(cell.cell_type, cell.source) for cell in source.cells]
    executed_cells = [(cell.cell_type, cell.source) for cell in executed.cells]
    if source_cells != executed_cells:
        raise ValueError("Source and executed notebooks differ; rerun the source with --execute.")
    for cell in executed.cells:
        if any(output.output_type == "error" for output in cell.get("outputs", [])):
            raise ValueError("The executed notebook contains a failed cell.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run source notebooks and save their outputs.")
    parser.add_argument("--only", choices=sorted(TUTORIALS))
    parser.add_argument("--data-root", default="data/tutorials")
    parser.add_argument("--output-root", default="outputs/tutorials")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse completed training where supported by the tutorial.")
    args = parser.parse_args()
    selected = {args.only: TUTORIALS[args.only]} if args.only else TUTORIALS
    old_environment = os.environ.copy()
    try:
        if args.execute:
            os.environ.update({
                "SMITH_TUTORIAL_DATA": str(Path(args.data_root).expanduser().resolve()),
                "SMITH_TUTORIAL_OUTPUT": str(Path(args.output_root).expanduser().resolve()),
                "SMITH_TUTORIAL_DEVICE": args.device,
                "SMITH_TUTORIAL_REUSE_EXISTING": "1" if args.reuse_existing else "0",
            })
        for stem in selected.values():
            source_path = NOTEBOOK_ROOT / f"{stem}_source.ipynb"
            executed_path = NOTEBOOK_ROOT / f"{stem}_executed.ipynb"
            source = nbformat.read(source_path, as_version=4)
            if args.execute:
                executed = NotebookClient(source, timeout=None, kernel_name="python3", allow_errors=False).execute(cwd=str(ROOT))
                nbformat.write(executed, executed_path)
            else:
                executed = nbformat.read(executed_path, as_version=4)
            validate_pair(source, executed)
            print(f"Validated {source_path.relative_to(ROOT)}")
    finally:
        os.environ.clear()
        os.environ.update(old_environment)


if __name__ == "__main__":
    main()
