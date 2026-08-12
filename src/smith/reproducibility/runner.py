from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from importlib import resources

from .registry import ReproducibilityCase, default_reproducibility_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_case(case: ReproducibilityCase, root: str | Path | None = None) -> dict[str, Any]:
    repro_root = Path(root).resolve() if root else default_reproducibility_root()
    checks = []
    for spec in case.inputs:
        path = repro_root / str(spec["path"])
        expected = str(spec.get("sha256", ""))
        actual = _sha256(path) if path.exists() and path.is_file() else ""
        checks.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256_ok": not expected or actual == expected,
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )
    return {"case": case.id, "ready": all(item["exists"] and item["sha256_ok"] for item in checks), "inputs": checks}


def _write_summary(case: ReproducibilityCase, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "case": case.id,
        "title": case.title,
        "manuscript_section": case.manuscript_section,
        "figure": case.figure,
        "claim": case.claim,
        **payload,
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["summary_json"] = str(output_path)
    return result


def _run_wmb(case: ReproducibilityCase, repro_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    package_root = repro_root.parent
    scripts_root = package_root / "scripts"
    if not scripts_root.exists():
        scripts_root = Path(str(resources.files("smith_agent.resources"))) / "scripts"
    env = os.environ.copy()
    source_root = package_root / "src"
    if source_root.exists():
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(source_root) + ((os.pathsep + current) if current else "")
    adata = output_dir / "example_panel.h5ad"
    saving = output_dir / "selection"
    logs = output_dir / "logs"
    subprocess.run(
        [sys.executable, str(scripts_root / "make_smoke_h5ad.py"), "--output", str(adata)],
        check=True,
        cwd=output_dir,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts_root / "main.py"),
            "--adata_file", str(adata),
            "--saving_dir", str(saving),
            "--log_dir", str(logs),
            "--tasks", "recon,cls,region,pathology,coordination",
            "--task_name", "celltype",
            "--panel_size", "8",
            "--epoch", "1",
            "--record", "1",
            "--batch_size", "16",
            "--rep_dim", "8",
            "--rep_hidden_dims", "8",
            "--head_hidden_dims", "8",
            "--dim", "8",
            "--device", "cpu",
            "--seed", "7",
            "--balance_mode", "off",
        ],
        check=True,
        cwd=output_dir,
        env=env,
    )
    ranking = pd.read_csv(saving / "epoch_0.csv")
    selected = ranking.head(8)
    return _write_summary(
        case,
        output_dir,
        {
            "selected_targets": selected.iloc[:, 0].astype(str).tolist(),
            "selected_panel_size": int(len(selected)),
            "ranking_size": int(len(ranking)),
        },
    )


def _run_regulatory(case: ReproducibilityCase, repro_root: Path, output_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(repro_root / case.inputs[0]["path"])
    smith = df[df["method"].str.startswith("SMITH")].copy()
    rows = []
    for (dataset, panel_size), group in smith.groupby(["dataset", "num_markers"]):
        best = group.sort_values(["celltype_knn_accuracy_mean", "time_knn_pearson_mean"], ascending=False).iloc[0]
        rows.append({
            "dataset": dataset,
            "panel_size": int(panel_size),
            "method": best["method"],
            "celltype_accuracy": float(best["celltype_knn_accuracy_mean"]),
            "time_pearson": float(best["time_knn_pearson_mean"]),
        })
    return _write_summary(case, output_dir, {"best_smith_by_dataset_and_panel": rows})


def _run_ribomap(case: ReproducibilityCase, repro_root: Path, output_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(repro_root / case.inputs[0]["path"])
    subset = df[(df["method"] == "SMITH") & (df["metric"] == "accuracy")].copy()
    rows = subset.sort_values(["dataset", "label"])[["dataset", "label", "panel_size", "value_mean", "value_std", "rank"]]
    return _write_summary(case, output_dir, {"smith_transfer_metrics": rows.to_dict(orient="records")})


def _run_inhouse(case: ReproducibilityCase, repro_root: Path, output_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(repro_root / case.inputs[0]["path"])
    cols = ["comparison", "n_seeds", "delta_spearman_mean", "delta_top64_mean", "spearman_improved_seeds", "top64_improved_seeds"]
    return _write_summary(case, output_dir, {"transfer_robustness": df[cols].to_dict(orient="records")})


def _run_agent(case: ReproducibilityCase, repro_root: Path, output_dir: Path) -> dict[str, Any]:
    metrics = pd.read_csv(repro_root / case.inputs[0]["path"], sep="\t")
    accuracy = metrics[metrics["metric"] == "cell_type_accuracy"]
    grouped = accuracy.groupby(["panel_size", "panel"], as_index=False)["value"].agg(["mean", "std"]).reset_index()
    feasibility = pd.read_csv(repro_root / case.inputs[1]["path"], sep="\t")
    pass_columns = [name for name in ["pass_property", "pass_specificity", "pass_deployment", "overall_pass"] if name in feasibility]
    pass_rates = {name: float(feasibility[name].fillna(False).astype(bool).mean()) for name in pass_columns}
    return _write_summary(
        case,
        output_dir,
        {"multi_reference_accuracy": grouped.to_dict(orient="records"), "feasibility_pass_rates": pass_rates},
    )


RUNNERS = {
    "01_wmb": _run_wmb,
    "02_regulatory_activity": _run_regulatory,
    "03_ribomap_transfer": _run_ribomap,
    "04_inhouse_disease": _run_inhouse,
    "05_agent": _run_agent,
}


def run_case(case: ReproducibilityCase, output_dir: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    repro_root = Path(root).resolve() if root else default_reproducibility_root()
    status = check_case(case, repro_root)
    if not status["ready"]:
        raise FileNotFoundError(f"Inputs are missing or invalid for reproducibility case `{case.id}`.")
    runner = RUNNERS.get(case.id)
    if runner is None:
        raise KeyError(f"No runner registered for reproducibility case `{case.id}`.")
    return runner(case, repro_root, Path(output_dir).resolve())
