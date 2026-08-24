"""Manuscript-defined C. elegans analyses for Figure 3g-k."""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.cluster import SpectralBiclustering
from sklearn.linear_model import Ridge

from reproducibility.workflows.common import gene_symbols, read_panel


LINEAGE_NAMES = ("muscle", "neuron", "pharynx", "skin")
TISSUE_ALIASES = {
    "muscle": "muscle",
    "neuronal system": "neuron",
    "neuron": "neuron",
    "pharynx": "pharynx",
    "skin": "skin",
    "intestine": "intestine",
}
ISOFORM_SUFFIX_RE = re.compile(r"\([A-Za-z0-9_.-]+\)$")


def canonical_tf_name(value: object) -> str:
    return ISOFORM_SUFFIX_RE.sub("", str(value).strip().upper())


def _matrix(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _close(adata: ad.AnnData) -> None:
    if getattr(adata, "file", None) is not None:
        adata.file.close()


def _collapse_columns(matrix: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(name, []).append(index)
    ordered = sorted(name for name in groups if name)
    collapsed = np.empty((matrix.shape[0], len(ordered)), dtype=np.float32)
    for output_index, name in enumerate(ordered):
        collapsed[:, output_index] = matrix[:, groups[name]].mean(axis=1)
    return collapsed, ordered


def _aggregate_rows(
    matrix: np.ndarray,
    groups: list[str],
    keep_groups: list[str],
) -> tuple[np.ndarray, list[str]]:
    frame = pd.DataFrame(matrix)
    frame.insert(0, "_group", pd.Series(groups, dtype="string"))
    frame = frame[frame["_group"].isin(set(keep_groups))]
    grouped = frame.groupby("_group", sort=True, observed=False).mean(numeric_only=True)
    return grouped.to_numpy(dtype=np.float32), grouped.index.astype(str).tolist()


def _correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    standard_deviation = centered.std(axis=0, ddof=1, keepdims=True)
    standard_deviation[standard_deviation == 0] = 1.0
    correlation = ((centered / standard_deviation).T @ (centered / standard_deviation)) / max(
        matrix.shape[0] - 1, 1
    )
    correlation = np.clip(correlation, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _module_table(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    aliases = {str(column).strip().lower(): column for column in frame.columns}
    gene = next((aliases[name] for name in ("gene_symbol", "gene", "tf", "target") if name in aliases), None)
    module = next((aliases[name] for name in ("module_id", "module", "module_name") if name in aliases), None)
    if module is None and {
        "tissue", "progenitor_lineage", "temporal_module"
    }.issubset(aliases):
        frame["_module_id"] = frame[
            [aliases["tissue"], aliases["progenitor_lineage"], aliases["temporal_module"]]
        ].astype(str).agg("|".join, axis=1)
        module = "_module_id"
    if module is None or gene is None:
        raise ValueError(f"Module annotation {path} lacks module and TF columns")
    result = frame[[module, gene]].rename(columns={module: "module_id", gene: "gene_symbol"}).copy()
    result["module_id"] = result["module_id"].astype(str).str.strip()
    result["gene_symbol"] = result["gene_symbol"].map(canonical_tf_name)
    result = result[(result["module_id"] != "") & (result["gene_symbol"] != "")]
    if result.empty:
        raise ValueError(f"Module annotation {path} contains no usable rows")
    return result.drop_duplicates()


def module_miss_rate(panel: list[str] | set[str], modules: pd.DataFrame) -> float:
    selected = {canonical_tf_name(gene) for gene in panel}
    normalized = modules.assign(gene_symbol=modules["gene_symbol"].map(canonical_tf_name))
    grouped = normalized.groupby("module_id")["gene_symbol"].apply(set)
    return float(sum(not (genes & selected) for genes in grouped) / len(grouped))


def write_module_coverage(
    panels: pd.DataFrame,
    module_file: str | Path,
    output_file: str | Path,
) -> Path:
    modules = _module_table(module_file)
    rows = []
    for row in panels.to_dict("records"):
        genes = read_panel(Path(str(row["panel_file"])), int(row["panel_size"]))
        rows.append(
            {
                "dataset": row.get("dataset", "elegans_tf"),
                "split": row.get("split", "split_1"),
                "training_seed": row.get("training_seed", row.get("seed", 0)),
                "method": row.get("method", "SMITH"),
                "panel_size": int(row["panel_size"]),
                "n_modules": int(modules["module_id"].nunique()),
                "module_miss_rate": module_miss_rate(genes, modules),
            }
        )
    if not rows:
        raise ValueError("No generated TF panels were supplied for module coverage")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_file, sep="\t", index=False)
    return output_file


def _pair_score(matrix: np.ndarray, left: int, right: int) -> float:
    a, b = matrix[:, left].astype(float), matrix[:, right].astype(float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _smith_reconstruction(
    test_matrix: np.ndarray,
    panel_positions: list[int],
    checkpoint_file: str | Path,
) -> np.ndarray:
    import torch
    from smith.model_selector import get_model

    args = SimpleNamespace(
        device="cpu",
        rep_dim=32,
        rep_hidden_dims="32",
        dim=32,
        head_hidden_dims="",
        lam=0.5,
        sigma=0.5,
        dropout_rate=0.2,
        activation="tanh",
        hurdle=False,
    )
    selected_input = np.zeros_like(test_matrix, dtype=np.float32)
    selected_input[:, panel_positions] = test_matrix[:, panel_positions]
    model = get_model(["input", "recon"], [selected_input, test_matrix], args)
    checkpoint = Path(checkpoint_file)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SMITH reconstruction checkpoint is missing: {checkpoint}")
    reconstruction_checkpoint = checkpoint.with_name(checkpoint.name.replace("-rep-", "-recon-"))
    if not reconstruction_checkpoint.is_file():
        raise FileNotFoundError(f"SMITH reconstruction head is missing: {reconstruction_checkpoint}")
    model["rep"].load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model["recon"].load_state_dict(torch.load(reconstruction_checkpoint, map_location="cpu"))
    model["rep"].eval()
    model["recon"].eval()
    with torch.no_grad():
        representation, _ = model["rep"](torch.from_numpy(selected_input), None)
        reconstructed, _ = model["recon"](representation, None)
    return reconstructed.detach().cpu().numpy()


def _read_pairs(path: str | Path | None, genes: list[str]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(
            [("all", "", left, right) for left, right in itertools.combinations(genes, 2)],
            columns=["tissue", "progenitor_lineage", "gene_a", "gene_b"],
        )
    frame = pd.read_csv(path, sep="\t")
    columns = {str(column).strip().lower(): column for column in frame.columns}
    left = next((columns[name] for name in ("gene_a", "tf_a", "tf1", "gene1", "left") if name in columns), None)
    right = next((columns[name] for name in ("gene_b", "tf_b", "tf2", "gene2", "right") if name in columns), None)
    if left is None or right is None:
        raise ValueError(f"TF-pair annotation {path} lacks TF1/TF2 columns")
    tissue = columns.get("tissue")
    progenitor = next(
        (columns[name] for name in ("progenitor_lineage", "progenitor cell lineage (spatial module)") if name in columns),
        None,
    )
    result = pd.DataFrame(
        {
            "tissue": frame[tissue].astype(str) if tissue else "all",
            "progenitor_lineage": frame[progenitor].astype(str) if progenitor else "",
            "gene_a": frame[left].map(canonical_tf_name),
            "gene_b": frame[right].map(canonical_tf_name),
        }
    )
    result["tissue"] = result["tissue"].str.strip().str.lower().map(TISSUE_ALIASES).fillna(
        result["tissue"].str.strip().str.lower()
    )
    result["progenitor_lineage"] = result["progenitor_lineage"].str.strip()
    available = set(genes)
    result = result[
        result["gene_a"].isin(available)
        & result["gene_b"].isin(available)
        & (result["gene_a"] != result["gene_b"])
    ]
    if result.empty:
        raise ValueError(f"No TF pairs in {path} overlap the activity input")
    return result.drop_duplicates()


def _load_reconstruction(path: str | Path, genes: list[str]) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.ndarray):
        return np.asarray(payload)
    stored_genes = [canonical_tf_name(gene) for gene in payload["genes"].astype(str)]
    reconstruction, collapsed_genes = _collapse_columns(payload["reconstruction"], stored_genes)
    positions = {gene: index for index, gene in enumerate(collapsed_genes)}
    missing = [gene for gene in genes if gene not in positions]
    if missing:
        raise ValueError(f"Reconstruction {path} lacks {len(missing)} activity TFs")
    return reconstruction[:, [positions[gene] for gene in genes]]


def coactivity_reconstruction(
    train_file: str | Path,
    test_file: str | Path,
    panel_file: str | Path,
    output_file: str | Path,
    *,
    pair_file: str | Path | None = None,
    lineage_column: str | None = None,
    max_pairs: int | None = None,
    seed: int = 1,
    method: str = "SMITH",
    checkpoint_file: str | Path | None = None,
    reconstruction_file: str | Path | None = None,
) -> Path:
    """Compare reconstructed and observed TF-pair cosine profiles in held-out lineages."""
    train, test = ad.read_h5ad(train_file), ad.read_h5ad(test_file)
    try:
        train_names = [canonical_tf_name(gene) for gene in gene_symbols(train)]
        test_names = [canonical_tf_name(gene) for gene in gene_symbols(test)]
        raw_train = _matrix(train).astype(np.float32)
        raw_test = _matrix(test).astype(np.float32)
        x_train, train_genes = _collapse_columns(raw_train, train_names)
        x_test, test_genes = _collapse_columns(raw_test, test_names)
        common = sorted(set(train_genes) & set(test_genes))
        if len(common) < 3:
            raise ValueError("Training and test activity inputs have fewer than three shared TFs")
        train_index = {gene: index for index, gene in enumerate(train_genes)}
        test_index = {gene: index for index, gene in enumerate(test_genes)}
        x_train = x_train[:, [train_index[gene] for gene in common]]
        x_test = x_test[:, [test_index[gene] for gene in common]]
        panel = [canonical_tf_name(gene) for gene in read_panel(panel_file) if canonical_tf_name(gene) in common]
        panel = list(dict.fromkeys(panel))
        if len(panel) < 2:
            raise ValueError("Generated panel has fewer than two TFs in the activity universe")
        if reconstruction_file:
            reconstructed = _load_reconstruction(reconstruction_file, common)
            backend = "persist_reconstruction_head"
        elif checkpoint_file and method == "SMITH":
            panel_positions = [index for index, gene in enumerate(test_names) if gene in set(panel)]
            reconstructed_raw = _smith_reconstruction(raw_test, panel_positions, checkpoint_file)
            reconstructed_collapsed, reconstructed_genes = _collapse_columns(
                reconstructed_raw, test_names
            )
            reconstructed_index = {
                gene: index for index, gene in enumerate(reconstructed_genes)
            }
            reconstructed = reconstructed_collapsed[
                :, [reconstructed_index[gene] for gene in common]
            ]
            backend = "smith_reconstruction_head"
        elif method == "ridge":
            panel_positions = [common.index(gene) for gene in panel]
            decoder = Ridge(alpha=1.0).fit(x_train[:, panel_positions], x_train)
            reconstructed = decoder.predict(x_test[:, panel_positions])
            backend = "ridge_panel_decoder"
        else:
            raise FileNotFoundError(
                f"{method} co-activity analysis requires its held-out reconstruction output; "
                "a substitute decoder is not used"
            )

        if lineage_column:
            if lineage_column not in test.obs:
                raise KeyError(f"Required lineage column {lineage_column!r} is absent")
            cell_names = test.obs[lineage_column].astype(str).to_numpy()
        elif "cell_name" in test.obs:
            cell_names = test.obs["cell_name"].astype(str).to_numpy()
        else:
            raise KeyError("TF activity input lacks obs['cell_name'] for progenitor-lineage matching")

        pairs = _read_pairs(pair_file, common)
        rows = []
        rng = np.random.default_rng(seed)
        for lineage in LINEAGE_NAMES:
            tissue_pairs = pairs[pairs["tissue"].isin([lineage, "all"])].copy()
            progenitors = sorted(set(tissue_pairs["progenitor_lineage"]) - {""})
            if progenitors:
                mask = np.array([any(name.startswith(prefix) for prefix in progenitors) for name in cell_names])
            else:
                labels = test.obs.get("cell_type", pd.Series("", index=test.obs_names)).astype(str).str.lower()
                aliases = (lineage, "hypodermis") if lineage == "skin" else (lineage,)
                mask = np.array([any(alias in label for alias in aliases) for label in labels])
            if mask.sum() < 3 or tissue_pairs.empty:
                continue
            if max_pairs and len(tissue_pairs) > max_pairs:
                tissue_pairs = tissue_pairs.iloc[
                    np.sort(rng.choice(len(tissue_pairs), size=max_pairs, replace=False))
                ]
            truth_scores, predicted_scores = [], []
            for pair in tissue_pairs.itertuples(index=False):
                left, right = common.index(pair.gene_a), common.index(pair.gene_b)
                truth_scores.append(_pair_score(x_test[mask], left, right))
                predicted_scores.append(_pair_score(reconstructed[mask], left, right))
            correlation = (
                float(pearsonr(truth_scores, predicted_scores).statistic)
                if len(truth_scores) > 1 and np.std(truth_scores) and np.std(predicted_scores)
                else np.nan
            )
            rows.append(
                {
                    "split": "test",
                    "lineage": lineage,
                    "method": method,
                    "panel_size": len(panel),
                    "pair_scope": "annotated" if pair_file else "all",
                    "backend": backend,
                    "n_cells": int(mask.sum()),
                    "n_pairs": int(len(tissue_pairs)),
                    "pearson": correlation,
                }
            )
        if not rows:
            raise ValueError("No held-out observations matched the annotated tissue progenitor lineages")
    finally:
        _close(train)
        _close(test)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_file, sep="\t", index=False)
    return output_file


def _matrix_pair_correlation(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    triangle = np.triu_indices_from(matrix_a, k=1)
    return float(pearsonr(matrix_a[triangle], matrix_b[triangle]).statistic)


def _mean_rowwise_correlation(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    values = []
    for row_a, row_b in zip(matrix_a, matrix_b):
        if np.std(row_a) and np.std(row_b):
            values.append(float(pearsonr(row_a, row_b).statistic))
    return float(np.mean(values)) if values else float("nan")


def tf_scrna_correlation(
    scrna_file: str | Path,
    tf_file: str | Path,
    output_file: str | Path,
    *,
    tf_lineage_column: str = "cell_name",
    scrna_lineage_column: str = "random_precise_lineage",
    n_clusters: int = 6,
    random_state: int = 0,
) -> Path:
    """Reproduce Figure 3j from shared TFs aggregated over shared lineages."""
    scrna, tf = ad.read_h5ad(scrna_file), ad.read_h5ad(tf_file)
    try:
        scrna_source_names = (
            scrna.var["gene_short_name"].astype(str).tolist()
            if "gene_short_name" in scrna.var
            else scrna.var_names.astype(str).tolist()
        )
        scrna_matrix, scrna_genes = _collapse_columns(
            _matrix(scrna), [str(name).strip().upper() for name in scrna_source_names]
        )
        tf_matrix, tf_genes = _collapse_columns(
            _matrix(tf), [canonical_tf_name(name) for name in tf.var_names.astype(str)]
        )
        shared_genes = sorted(set(scrna_genes) & set(tf_genes))
        if len(shared_genes) < 3:
            raise ValueError("scRNA and TF activity inputs share fewer than three TFs")
        if tf_lineage_column not in tf.obs or scrna_lineage_column not in scrna.obs:
            raise KeyError(
                f"Expected TF obs[{tf_lineage_column!r}] and scRNA obs[{scrna_lineage_column!r}]"
            )
        tf_lineages = tf.obs[tf_lineage_column].astype(str).tolist()
        scrna_lineages = scrna.obs[scrna_lineage_column].astype(str).tolist()
        shared_lineages = sorted(set(tf_lineages) & set(scrna_lineages))
        if not shared_lineages:
            raise ValueError("scRNA and TF activity inputs have no shared lineage identifiers")
        tf_index = {gene: index for index, gene in enumerate(tf_genes)}
        scrna_index = {gene: index for index, gene in enumerate(scrna_genes)}
        tf_shared = tf_matrix[:, [tf_index[gene] for gene in shared_genes]]
        scrna_shared = scrna_matrix[:, [scrna_index[gene] for gene in shared_genes]]
        tf_aggregated, tf_order = _aggregate_rows(tf_shared, tf_lineages, shared_lineages)
        scrna_aggregated, scrna_order = _aggregate_rows(scrna_shared, scrna_lineages, shared_lineages)
        if tf_order != scrna_order:
            raise RuntimeError("Lineage ordering differs after TF/scRNA aggregation")
        tf_correlation = np.abs(_correlation_matrix(tf_aggregated))
        scrna_correlation = np.abs(_correlation_matrix(scrna_aggregated))
        upper_triangle = _matrix_pair_correlation(tf_correlation, scrna_correlation)
        mean_rowwise = _mean_rowwise_correlation(tf_correlation, scrna_correlation)

        clustering = SpectralBiclustering(
            n_clusters=(n_clusters, n_clusters), method="log", random_state=random_state
        ).fit(tf_correlation)
        order = np.lexsort((np.asarray(shared_genes), clustering.column_labels_, clustering.row_labels_))
        ordered_genes = np.asarray(shared_genes)[order]
        matrix_file = Path(output_file).with_name("figure3_j_correlation_matrices.npz")
        matrix_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            matrix_file,
            genes=ordered_genes,
            tf=np.asarray(tf_correlation[order][:, order], dtype=np.float32),
            scrna=np.asarray(scrna_correlation[order][:, order], dtype=np.float32),
        )
    finally:
        _close(scrna)
        _close(tf)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "shared_genes": len(shared_genes),
                "shared_lineages": len(shared_lineages),
                "pearson": mean_rowwise,
                "mean_rowwise_pearson": mean_rowwise,
                "upper_triangle_pearson": upper_triangle,
                "matrix_file": matrix_file.name,
            }
        ]
    ).to_csv(output_file, sep="\t", index=False)
    return output_file
