from __future__ import annotations

import subprocess
import sys
from pathlib import Path


METHOD_DIRS = {
    "PERSIST-class": ("persist", "run_sup.py"),
    "PERSIST": ("persist", "run_unsup.py"),
    "ActiveSVM": ("activeSVM", "run.py"),
    "scGIST": ("scGIST", "run.py"),
    "scGeneFit": ("scGeneFit", "run.py"),
    "Spapros": ("spapros", "run.py"),
}


def run_baseline(
    method: str,
    train_file: str | Path,
    output_dir: str | Path,
    panel_size: int,
    label_column: str,
    baseline_root: str | Path,
    epochs: int,
    force: bool,
    python_executable: str | Path | None = None,
) -> Path:
    if method not in METHOD_DIRS:
        raise ValueError(f"Unsupported manuscript baseline: {method}")
    folder, script_name = METHOD_DIRS[method]
    script = Path(baseline_root).resolve() / folder / script_name
    if not script.is_file():
        raise FileNotFoundError(
            f"Missing {method} backend at {script}. Clone/install the manuscript baseline bundle and pass --baseline-root."
        )
    output_dir = Path(output_dir)
    panel = output_dir / f"marker_{panel_size}.csv"
    if panel.is_file() and not force:
        return panel
    output_dir.mkdir(parents=True, exist_ok=True)
    interpreter = str(Path(python_executable).expanduser().resolve()) if python_executable else sys.executable
    if not Path(interpreter).is_file():
        raise FileNotFoundError(f"Python interpreter for {method} does not exist: {interpreter}")
    command = [interpreter, str(script), "--adata", str(Path(train_file).resolve())]
    if method != "PERSIST":
        command += ["--label", label_column]
    command += ["--num_markers", str(panel_size), "--output", str(output_dir)]
    if method in {"PERSIST-class", "PERSIST"}:
        command += ["--max_epochs", str(epochs)]
    elif method == "scGIST":
        command += ["--epochs", str(epochs)]
    elif method == "ActiveSVM":
        command += ["--num_samples", "3600", "--max_iter", str(epochs)]
    with (output_dir / "training.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not panel.is_file():
        raise FileNotFoundError(f"{method} finished without writing {panel}")
    return panel
