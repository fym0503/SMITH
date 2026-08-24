"""Paper-specific C. elegans analyses for Figure 3g-k.

These helpers deliberately consume raw/processed biological inputs and panels
created by the current workflow.  They do not read the historical aggregate
tables shipped as reference outputs.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

from reproducibility.workflows.common import gene_symbols, read_panel


LINEAGE_NAMES = ("muscle", "neuron", "pharynx", "skin")


def _matrix(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _find_column(adata: ad.AnnData, requested: str | None, candidates: tuple[str, ...]) -> str:
    if requested:
        if requested not in adata.obs:
            raise KeyError(f"Required observation column {requested!r} is absent from {adata.filename}")
        return requested
    for candidate in candidates:
        if candidate in adata.obs:
            return candidate
    raise KeyError(f"Could not find any of {candidates} in {adata.filename}")


def _module_table(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    aliases = {str(column).lower(): column for column in frame.columns}
    module = next((aliases[name] for name in ("module", "module_id", "module_name") if name in aliases), None)
    gene = next((aliases[name] for name in ("gene", "gene_symbol", "tf", "target") if name in aliases), None)
    if module is None or gene is None:
        raise ValueError(f"Module annotation {path} must contain module_id and gene_symbol columns")
    result = frame[[module, gene]].rename(columns={module: "module_id", gene: "gene_symbol"}).copy()
    result["module_id"] = result["module_id"].astype(str)
    result["gene_symbol"] = result["gene_symbol"].astype(str).str.upper()
    result = result[(result["module_id"] != "") & (result["gene_symbol"] != "")]
    if result.empty:
        raise ValueError(f"Module annotation {path} contains no usable rows")
    return result.drop_duplicates()


def module_miss_rate(panel: list[str] | set[str], modules: pd.DataFrame) -> float:
    selected = {str(gene).upper() for gene in panel}
    normalized = modules.assign(gene_symbol=modules["gene_symbol"].astype(str).str.upper())
    grouped = normalized.groupby("module_id")["gene_symbol"].apply(set)
    return float(sum(not (genes & selected) for genes in grouped) / len(grouped))


def write_module_coverage(
    panels: pd.DataFrame,
    module_file: str | Path,
    output_file: str | Path,
) -> Path:
    """Write Figure 3h values from generated panel files."""
    modules = _module_table(module_file)
    rows = []
    for row in panels.to_dict("records"):
        panel_path = Path(str(row["panel_file"]))
        genes = read_panel(panel_path, int(row["panel_size"]))
        rows.append({
            "dataset": row.get("dataset", "elegans_tf"),
            "split": row.get("split", "split_1"),
            "training_seed": row.get("training_seed", row.get("seed", 0)),
            "method": row.get("method", "SMITH"),
            "panel_size": int(row["panel_size"]),
            "module_miss_rate": module_miss_rate(genes, modules),
        })
    if not rows:
        raise ValueError("No generated TF panel files were supplied for module coverage")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_file, sep="\t", index=False)
    return output_file


def _pair_score(matrix: np.ndarray, left: int, right: int) -> float:
    a, b = matrix[:, left].astype(float), matrix[:, right].astype(float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _smith_reconstruction(test_matrix: np.ndarray, panel_positions: list[int], checkpoint_file: str | Path) -> np.ndarray:
    """Run the reconstruction head saved by the current SMITH training run."""
    import torch
    from smith.model_selector import get_model

    args = SimpleNamespace(
        device="cpu", rep_dim=32, rep_hidden_dims="32", dim=32, head_hidden_dims="",
        lam=0.5, sigma=0.5, dropout_rate=0.2, activation="tanh", hurdle=False,
    )
    matrix = np.zeros_like(test_matrix, dtype=np.float32)
    matrix[:, panel_positions] = test_matrix[:, panel_positions]
    tensor = torch.from_numpy(matrix.astype(np.float32))
    full = torch.from_numpy(test_matrix.astype(np.float32))
    model = get_model(["input", "recon"], [tensor.numpy(), full.numpy()], args)
    checkpoint = Path(checkpoint_file)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SMITH reconstruction checkpoint is missing: {checkpoint}")
    saving_dir = checkpoint.parent
    model["rep"].load_state_dict(torch.load(saving_dir / checkpoint.name.replace("-rep-", "-rep-"), map_location="cpu"))
    recon_name = checkpoint.name.replace("-rep-", "-recon-")
    model["recon"].load_state_dict(torch.load(saving_dir / recon_name, map_location="cpu"))
    model["rep"].eval()
    model["recon"].eval()
    with torch.no_grad():
        representation, _ = model["rep"](tensor, None)
        reconstructed, _ = model["recon"](representation, None)
    return reconstructed.detach().cpu().numpy()


def _read_pairs(path: str | Path | None, genes: list[str]) -> list[tuple[str, str]]:
    if path is None:
        return list(itertools.combinations(genes, 2))
    frame = pd.read_csv(path, sep="\t")
    columns = {str(column).lower(): column for column in frame.columns}
    left = next((columns[name] for name in ("gene_a", "tf_a", "gene1", "left") if name in columns), None)
    right = next((columns[name] for name in ("gene_b", "tf_b", "gene2", "right") if name in columns), None)
    if left is None or right is None:
        raise ValueError(f"TF pair annotation {path} must contain gene_a and gene_b columns")
    available = set(genes)
    pairs = []
    for a, b in frame[[left, right]].itertuples(index=False):
        pair = (str(a).upper(), str(b).upper())
        if pair[0] in available and pair[1] in available and pair[0] != pair[1]:
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"No TF pairs in {path} overlap the activity input")
    return list(dict.fromkeys(pairs))


def coactivity_reconstruction(
    train_file: str | Path,
    test_file: str | Path,
    panel_file: str | Path,
    output_file: str | Path,
    *,
    pair_file: str | Path | None = None,
    lineage_column: str | None = None,
    max_pairs: int | None = 5000,
    seed: int = 1,
    method: str = "SMITH",
    checkpoint_file: str | Path | None = None,
) -> Path:
    """Reconstruct full TF profiles from a generated panel and score coactivity.

    The reconstruction head saved by SMITH is used when ``checkpoint_file`` is
    available. A transparent ridge panel decoder is used for external panels
    without a model checkpoint. In both cases the score is Pearson agreement of
    TF-pair cosine profiles on held-out cells.
    """
    train, test = ad.read_h5ad(train_file), ad.read_h5ad(test_file)
    try:
        train_genes, test_genes = gene_symbols(train), gene_symbols(test)
        common = [gene for gene in train_genes if gene in set(test_genes)]
        if len(common) < 3:
            raise ValueError("Training and test activity inputs have fewer than three shared TFs")
        train_pos = [train_genes.index(gene) for gene in common]
        test_pos = [test_genes.index(gene) for gene in common]
        x_train, x_test = _matrix(train[:, train_pos]), _matrix(test[:, test_pos])
        panel = [gene for gene in read_panel(panel_file) if gene in common]
        if len(panel) < 2:
            raise ValueError("Generated panel has fewer than two TFs in the activity universe")
        panel_train = [common.index(gene) for gene in panel]
        backend = "panel_decoder_ridge"
        if checkpoint_file and method == "SMITH":
            reconstructed = _smith_reconstruction(x_test, panel_train, checkpoint_file)
            backend = "smith_reconstruction_head"
        else:
            decoder = Ridge(alpha=1.0)
            decoder.fit(x_train[:, panel_train], x_train)
            reconstructed = decoder.predict(x_test[:, panel_train])
        column = _find_column(test, lineage_column, ("lineage", "lineage_name", "cell_type", "celltype"))
        labels = test.obs[column].astype(str).to_numpy()
        pairs = _read_pairs(pair_file, common)
        if max_pairs:
            pairs = pairs[:max_pairs]
        rows = []
        for lineage in LINEAGE_NAMES:
            mask = np.char.find(np.char.lower(labels.astype(str)), lineage) >= 0
            if mask.sum() < 3:
                continue
            truth_scores, predicted_scores = [], []
            for left, right in pairs:
                truth_scores.append(_pair_score(x_test[mask], common.index(left), common.index(right)))
                predicted_scores.append(_pair_score(reconstructed[mask], common.index(left), common.index(right)))
            correlation = float(pearsonr(truth_scores, predicted_scores).statistic) if len(truth_scores) > 1 else np.nan
            rows.append({"split": "test", "lineage": lineage, "method": method, "panel_size": len(panel), "pair_scope": "annotated" if pair_file else "all", "backend": backend, "pearson": correlation})
        if not rows:
            raise ValueError(f"No test observations matched the expected lineages {LINEAGE_NAMES}; check --lineage-column")
    finally:
        train.file.close()
        test.file.close()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_file, sep="\t", index=False)
    return output_file


def tf_scrna_correlation(
    scrna_file: str | Path,
    tf_file: str | Path,
    output_file: str | Path,
    *,
    max_genes: int = 128,
) -> Path:
    """Compare TF-TF correlation structure between scRNA and activity data."""
    scrna, tf = ad.read_h5ad(scrna_file), ad.read_h5ad(tf_file)
    try:
        scrna_genes, tf_genes = gene_symbols(scrna), gene_symbols(tf)
        shared = [gene for gene in tf_genes if gene in set(scrna_genes)]
        if len(shared) < 3:
            raise ValueError("scRNA and TF activity inputs share fewer than three genes")
        shared = shared[:max_genes]
        s = _matrix(scrna[:, [scrna_genes.index(gene) for gene in shared]])
        t = _matrix(tf[:, [tf_genes.index(gene) for gene in shared]])
        s_corr, t_corr = np.corrcoef(s, rowvar=False), np.corrcoef(t, rowvar=False)
        tri = np.triu_indices(len(shared), k=1)
        agreement = float(pearsonr(s_corr[tri], t_corr[tri]).statistic)
    finally:
        scrna.file.close()
        tf.file.close()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"shared_genes": len(shared), "pearson": agreement}]).to_csv(output_file, sep="\t", index=False)
    return output_file
