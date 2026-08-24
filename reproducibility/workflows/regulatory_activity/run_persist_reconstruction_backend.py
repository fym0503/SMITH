#!/usr/bin/env python3
"""Train the manuscript PERSIST backend and export held-out reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-adata", required=True)
    parser.add_argument("--test-adata", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel-size", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    persist_root = Path(args.baseline_root).resolve() / "persist" / "persist"
    if not persist_root.is_dir():
        raise FileNotFoundError(f"PERSIST Python package is missing: {persist_root}")
    sys.path.insert(0, str(persist_root))
    from persist.data import ExpressionDataset
    from persist.selection import PERSIST

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, test = ad.read_h5ad(args.train_adata), ad.read_h5ad(args.test_adata)
    try:
        train_genes = train.var_names.astype(str).str.upper().tolist()
        test_genes = test.var_names.astype(str).str.upper().tolist()
        if train_genes != test_genes:
            raise ValueError("PERSIST reconstruction requires aligned train/test features")
        x_train = dense(train.X).astype(np.float32)
        x_test = dense(test.X).astype(np.float32)
    finally:
        train.file.close()
        test.file.close()

    fit_x, validation_x = train_test_split(x_train, test_size=0.25, random_state=args.seed)
    train_dataset = ExpressionDataset(fit_x, fit_x)
    validation_dataset = ExpressionDataset(validation_x, validation_x)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable in the PERSIST environment")
    selector = PERSIST(
        train_dataset,
        validation_dataset,
        loss_fn=torch.nn.MSELoss(),
        device=device,
    )
    indices, model = selector.select(
        num_genes=args.panel_size,
        max_nepochs=args.max_epochs,
        mbsize=512,
        bar=False,
    )
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().numpy()
    indices = np.asarray(indices, dtype=int).reshape(-1)
    model.eval()
    with torch.no_grad():
        reconstruction, _, _ = model(torch.from_numpy(x_test).to(device))
    reconstruction = reconstruction.detach().cpu().numpy().astype(np.float32)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_file = output_dir / f"marker_{args.panel_size}.csv"
    reconstruction_file = output_dir / "test_reconstruction.npz"
    pd.DataFrame({"marker": [train_genes[int(index)] for index in indices]}).to_csv(panel_file, index=False)
    np.savez_compressed(
        reconstruction_file,
        reconstruction=reconstruction,
        genes=np.asarray(train_genes),
    )
    (output_dir / "reconstruction_manifest.json").write_text(
        json.dumps(
            {
                "method": "PERSIST",
                "panel_size": args.panel_size,
                "seed": args.seed,
                "train_adata": str(Path(args.train_adata).resolve()),
                "test_adata": str(Path(args.test_adata).resolve()),
                "panel_file": str(panel_file),
                "reconstruction_file": str(reconstruction_file),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
