from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def run_persist_reconstruction(
    train_file: str | Path,
    test_file: str | Path,
    output_dir: str | Path,
    panel_size: int,
    baseline_root: str | Path,
    epochs: int,
    force: bool,
    *,
    python_executable: str | Path | None = None,
    device: str = "cpu",
    seed: int = 1,
) -> tuple[Path, Path]:
    """Run the actual unsupervised PERSIST model and retain its reconstruction."""
    output_dir = Path(output_dir)
    panel = output_dir / f"marker_{panel_size}.csv"
    reconstruction = output_dir / "test_reconstruction.npz"
    if panel.is_file() and reconstruction.is_file() and not force:
        return panel, reconstruction
    output_dir.mkdir(parents=True, exist_ok=True)
    interpreter = str(Path(python_executable).expanduser().resolve()) if python_executable else sys.executable
    if not Path(interpreter).is_file():
        raise FileNotFoundError(f"Python interpreter for PERSIST does not exist: {interpreter}")
    command = [
        interpreter,
        str(REPO_ROOT / "reproducibility" / "workflows" / "regulatory_activity" / "run_persist_reconstruction_backend.py"),
        "--train-adata", str(Path(train_file).resolve()),
        "--test-adata", str(Path(test_file).resolve()),
        "--baseline-root", str(Path(baseline_root).resolve()),
        "--output", str(output_dir.resolve()),
        "--panel-size", str(panel_size),
        "--max-epochs", str(epochs),
        "--seed", str(seed),
        "--device", device,
    ]
    with (output_dir / "training.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not panel.is_file() or not reconstruction.is_file():
        raise FileNotFoundError("PERSIST finished without a panel and held-out reconstruction")
    return panel, reconstruction
