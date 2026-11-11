import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    explained_variance_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

try:
    import torch
except ImportError:
    torch = None

def downsample_adata(adata, target_n=50000, random_state=42):
    if target_n >= adata.n_obs:
        return adata.copy()
    np.random.seed(random_state)
    sampled_indices = np.random.choice(adata.n_obs, size=target_n, replace=False)
    return adata[sampled_indices].copy()


def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _sample_indices(n_obs, max_cells, random_state=42):
    if max_cells is None or n_obs <= max_cells:
        return np.arange(n_obs)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_obs, size=max_cells, replace=False))


def _dedupe_keep_order(items):
    deduped = []
    seen = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def load_panel_genes(panel_path, panel_size=None):
    panel_path = Path(panel_path)
    text = panel_path.read_text().strip()

    if panel_path.suffix.lower() == ".csv":
        raw_genes = pd.read_csv(panel_path).iloc[:, 0].astype(str).tolist()
    else:
        raw_genes = None
        if text:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple, np.ndarray, pd.Series)):
                raw_genes = [str(gene) for gene in parsed]
        if raw_genes is None:
            raw_genes = [line.strip().split(",")[0] for line in text.splitlines() if line.strip()]

    genes = _dedupe_keep_order([str(gene).upper() for gene in raw_genes if str(gene).strip()])
    if panel_size is not None:
        genes = genes[:panel_size]
    return genes


def _panel_positions(adata, selected_genes):
    adata_genes = pd.Index([str(gene).upper() for gene in adata.var_names.tolist()])
    selected_genes = _dedupe_keep_order([str(gene).upper() for gene in selected_genes])
    positions = adata_genes.get_indexer(selected_genes)
    missing = [gene for gene, pos in zip(selected_genes, positions) if pos < 0]
    if missing:
        raise KeyError(f"Missing panel genes in AnnData: {missing}")
    return selected_genes, positions.tolist()


def _shared_panel_positions(train_adata, test_adata, selected_genes):
    train_genes = pd.Index([str(gene).upper() for gene in train_adata.var_names.tolist()])
    test_genes = pd.Index([str(gene).upper() for gene in test_adata.var_names.tolist()])
    panel_genes = _dedupe_keep_order([str(gene).upper() for gene in selected_genes])

    train_positions = train_genes.get_indexer(panel_genes)
    test_positions = test_genes.get_indexer(panel_genes)

    shared = []
    train_shared_positions = []
    test_shared_positions = []
    missing = []
    for gene, train_pos, test_pos in zip(panel_genes, train_positions, test_positions):
        if train_pos >= 0 and test_pos >= 0:
            shared.append(gene)
            train_shared_positions.append(int(train_pos))
            test_shared_positions.append(int(test_pos))
        else:
            missing.append(gene)

    if not shared:
        raise KeyError("No panel genes were shared between train and test AnnData objects")

    return shared, train_shared_positions, test_shared_positions, missing


def _subset_matrices(adata, selected_genes, max_cells=None, random_state=42):
    _, positions = _panel_positions(adata, selected_genes)
    cell_indices = _sample_indices(adata.n_obs, max_cells, random_state=random_state)
    x_full = to_dense(adata.X[cell_indices]).astype(np.float32)
    x_panel = x_full[:, positions]
    return x_full, x_panel, cell_indices


def _knn_indices(x, k):
    n_neighbors = min(k + 1, x.shape[0])
    if n_neighbors <= 1:
        raise ValueError("Need at least 2 observations to compute neighborhood metrics")
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    model.fit(x)
    indices = model.kneighbors(x, return_distance=False)
    return indices[:, 1:]

def calculate_knn_accuracy(embedding, labels, label_name, n_neighbors=5, test_size=0.2, random_state=42):
    y = labels[label_name].values
    X_train, X_test, y_train, y_test = train_test_split(embedding, y, test_size=test_size, random_state=random_state)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return accuracy, report


def evaluate_split_knn_classification(train_adata, test_adata, selected_genes, label_name, n_neighbors=5):
    if label_name not in train_adata.obs.columns or label_name not in test_adata.obs.columns:
        raise KeyError(f"Label column {label_name!r} missing in train/test AnnData")

    _, train_positions, test_positions, _ = _shared_panel_positions(train_adata, test_adata, selected_genes)
    X_train = to_dense(train_adata.X).astype(np.float32)[:, train_positions]
    X_test = to_dense(test_adata.X).astype(np.float32)[:, test_positions]
    y_train = train_adata.obs[label_name].astype(str).to_numpy()
    y_test = test_adata.obs[label_name].astype(str).to_numpy()

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return float(accuracy), report

def evaluate_reconstruction(adata, selected_genes, test_size=0.2, random_state=42):
    selected_genes, selected_positions = _panel_positions(adata, selected_genes)
    all_genes = [str(g).upper() for g in adata.var_names.tolist()]
    selected_set = set(selected_genes)
    unselected_positions = [idx for idx, gene in enumerate(all_genes) if gene not in selected_set]
    
    if len(unselected_positions) == 0:
        return None, None
    
    x_all = to_dense(adata.X).astype(np.float32)
    X = x_all[:, selected_positions]
    y = x_all[:, unselected_positions]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    ev = explained_variance_score(y_test, y_pred)
    corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())
    
    return ev, corr

def eval_regression(adata, selected_genes, obsm_key, test_size=0.2, random_state=42):
    _, selected_positions = _panel_positions(adata, selected_genes)
    
    x_all = to_dense(adata.X).astype(np.float32)
    X = x_all[:, selected_positions]
    y = adata.obsm[obsm_key]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    ev = explained_variance_score(y_test, y_pred)
    corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())
    
    return ev, corr


def evaluate_split_reconstruction(train_adata, test_adata, selected_genes):
    train_genes = pd.Index([str(gene).upper() for gene in train_adata.var_names.tolist()])
    test_genes = pd.Index([str(gene).upper() for gene in test_adata.var_names.tolist()])
    shared_gene_set = set(train_genes).intersection(set(test_genes))
    if not shared_gene_set:
        raise KeyError("No shared genes were found between train and test AnnData objects")

    panel_genes = _dedupe_keep_order([str(gene).upper() for gene in selected_genes if str(gene).upper() in shared_gene_set])
    if not panel_genes:
        raise KeyError("No panel genes were shared between train and test AnnData objects")

    shared_genes = [gene for gene in train_genes.tolist() if gene in shared_gene_set]
    panel_gene_set = set(panel_genes)
    unselected_shared_genes = [gene for gene in shared_genes if gene not in panel_gene_set]
    if not unselected_shared_genes:
        return None, None

    train_panel_positions = train_genes.get_indexer(panel_genes).tolist()
    test_panel_positions = test_genes.get_indexer(panel_genes).tolist()
    train_unselected_positions = train_genes.get_indexer(unselected_shared_genes).tolist()
    test_unselected_positions = test_genes.get_indexer(unselected_shared_genes).tolist()

    train_matrix = to_dense(train_adata.X).astype(np.float32)
    test_matrix = to_dense(test_adata.X).astype(np.float32)
    X_train = train_matrix[:, train_panel_positions]
    X_test = test_matrix[:, test_panel_positions]
    y_train = train_matrix[:, train_unselected_positions]
    y_test = test_matrix[:, test_unselected_positions]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ev = float(explained_variance_score(y_test, y_pred))
    if np.std(y_test.flatten()) == 0 or np.std(y_pred.flatten()) == 0:
        corr = np.nan
    else:
        corr = float(pearsonr(y_test.flatten(), y_pred.flatten())[0])
    return ev, corr


def evaluate_split_obsm_regression(train_adata, test_adata, selected_genes, obsm_key):
    if obsm_key not in train_adata.obsm or obsm_key not in test_adata.obsm:
        raise KeyError(f"obsm key {obsm_key!r} missing in train/test AnnData")

    _, train_positions, test_positions, _ = _shared_panel_positions(train_adata, test_adata, selected_genes)
    X_train = to_dense(train_adata.X).astype(np.float32)[:, train_positions]
    X_test = to_dense(test_adata.X).astype(np.float32)[:, test_positions]
    y_train = np.asarray(train_adata.obsm[obsm_key], dtype=np.float32)
    y_test = np.asarray(test_adata.obsm[obsm_key], dtype=np.float32)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if np.std(y_test.flatten()) == 0 or np.std(y_pred.flatten()) == 0:
        corr = np.nan
    else:
        corr = float(pearsonr(y_test.flatten(), y_pred.flatten())[0])
    ev = float(explained_variance_score(y_test, y_pred))
    return ev, corr


def infer_time_label(adata, candidates=("absolute_time", "consensus_time", "time")):
    for column in candidates:
        if column in adata.obs.columns:
            values = pd.to_numeric(adata.obs[column], errors="coerce")
            if values.notna().sum() >= 3:
                return column
    return None


def infer_group_label(adata, candidates=("cell_name",)):
    for column in candidates:
        if column in adata.obs.columns:
            values = adata.obs[column].astype(str)
            if values.nunique() >= 2:
                return column
    return None


def evaluate_time_knn_regression(
    adata,
    selected_genes,
    time_label,
    ks=(5, 10, 20),
    test_size=0.2,
    random_state=42,
    group_label=None,
):
    if time_label not in adata.obs.columns:
        raise KeyError(f"Time label column {time_label!r} not found in obs")

    _, selected_positions = _panel_positions(adata, selected_genes)
    x_all = to_dense(adata.X).astype(np.float32)
    y_all = pd.to_numeric(adata.obs[time_label], errors="coerce").to_numpy(dtype=np.float32)

    valid_mask = np.isfinite(y_all)
    if valid_mask.sum() < 3:
        raise ValueError(f"Need at least 3 valid observations in obs[{time_label!r}] for KNN regression")

    X = x_all[valid_mask][:, selected_positions]
    y = y_all[valid_mask]
    groups = None
    if group_label and group_label in adata.obs.columns:
        groups = adata.obs[group_label].astype(str).to_numpy()[valid_mask]

    if groups is not None and len(np.unique(groups)) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train, g_test = groups[train_idx], groups[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        g_train = g_test = None

    rows = []
    for k in ks:
        effective_k = min(int(k), max(1, X_train.shape[0]))
        model = KNeighborsRegressor(n_neighbors=effective_k, weights="distance")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if np.std(y_test) == 0 or np.std(y_pred) == 0:
            corr = np.nan
        else:
            corr = float(pearsonr(y_test, y_pred)[0])

        rows.append(
            {
                "metric": "time_knn_regression",
                "label": time_label,
                "group_label": group_label,
                "k": effective_k,
                "pearson_correlation": corr,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "n_train_groups": None if g_train is None else int(pd.Series(g_train).nunique()),
                "n_test_groups": None if g_test is None else int(pd.Series(g_test).nunique()),
            }
        )

    if rows:
        rows.append(
            {
                "metric": "time_knn_regression",
                "label": time_label,
                "group_label": group_label,
                "k": -1,
                "pearson_correlation": float(np.nanmean([row["pearson_correlation"] for row in rows])),
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "n_train_groups": None if g_train is None else int(pd.Series(g_train).nunique()),
                "n_test_groups": None if g_test is None else int(pd.Series(g_test).nunique()),
            }
        )

    return pd.DataFrame(rows)


def evaluate_split_time_knn_regression(
    train_adata,
    test_adata,
    selected_genes,
    time_label,
    n_neighbors=5,
):
    if time_label not in train_adata.obs.columns or time_label not in test_adata.obs.columns:
        raise KeyError(f"Time label column {time_label!r} missing in train/test AnnData")

    _, train_positions, test_positions, _ = _shared_panel_positions(train_adata, test_adata, selected_genes)
    X_train = to_dense(train_adata.X).astype(np.float32)[:, train_positions]
    X_test = to_dense(test_adata.X).astype(np.float32)[:, test_positions]
    y_train = pd.to_numeric(train_adata.obs[time_label], errors="coerce").to_numpy(dtype=np.float32)
    y_test = pd.to_numeric(test_adata.obs[time_label], errors="coerce").to_numpy(dtype=np.float32)

    train_mask = np.isfinite(y_train)
    test_mask = np.isfinite(y_test)
    if train_mask.sum() < 2 or test_mask.sum() < 2:
        raise ValueError(f"Need at least 2 valid train/test observations in obs[{time_label!r}]")

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    effective_k = min(int(n_neighbors), max(1, X_train.shape[0]))
    model = KNeighborsRegressor(n_neighbors=effective_k, weights="distance")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if np.std(y_test) == 0 or np.std(y_pred) == 0:
        corr = np.nan
    else:
        corr = float(pearsonr(y_test, y_pred)[0])
    return corr


def evaluate_knn_overlap(adata, selected_genes, ks=(5, 10, 20), max_cells=5000, random_state=42):
    x_full, x_panel, _ = _subset_matrices(adata, selected_genes, max_cells=max_cells, random_state=random_state)
    rows = []
    for k in ks:
        if x_full.shape[0] <= 1:
            continue
        effective_k = min(k, x_full.shape[0] - 1)
        full_knn = _knn_indices(x_full, effective_k)
        panel_knn = _knn_indices(x_panel, effective_k)
        overlap_fraction = []
        for idx in range(full_knn.shape[0]):
            full_set = set(map(int, full_knn[idx]))
            panel_set = set(map(int, panel_knn[idx]))
            intersection = len(full_set & panel_set)
            overlap_fraction.append(intersection / effective_k if effective_k else np.nan)
        rows.append(
            {
                "metric": "knn_overlap",
                "k": effective_k,
                "mean_overlap_fraction": float(np.mean(overlap_fraction)),
                "median_overlap_fraction": float(np.median(overlap_fraction)),
                "n_cells": int(x_full.shape[0]),
            }
        )
    if rows:
        rows.append(
            {
                "metric": "knn_overlap",
                "k": -1,
                "mean_overlap_fraction": float(np.mean([row["mean_overlap_fraction"] for row in rows])),
                "median_overlap_fraction": float(np.mean([row["median_overlap_fraction"] for row in rows])),
                "n_cells": int(x_full.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def save_results(epoch, var, models, args, tasks, train_loss, val_loss):
    array = models['rep'].mu.detach().cpu().numpy()
    index = np.argsort(array)[::-1]
    var_index = [var[i] for i in index]
    
    saving_name = args.tasks.replace(",", "-") + '-seed' + str(args.seed) + '-' + args.task_name
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.saving_dir, exist_ok=True)
    
    log_file = os.path.join(args.log_dir, f'{saving_name}.csv')
    var_file = os.path.join(args.saving_dir, f'epoch_{epoch}.csv')
    
    log_data = {'epoch': [epoch]}
    for t_index, t in enumerate(tasks[1:]):
        log_data[f'{t}_train_loss'] = [train_loss[t_index]]
        if args.val:
            log_data[f'{t}_val_loss'] = [val_loss[t_index]]
    
    df_log = pd.DataFrame(log_data)
    df_log.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    
    
    df_var = pd.DataFrame({'marker': var_index})
    df_var.to_csv(var_file, index=False)
    
    
    if args.save:
        if torch is None:
            raise ImportError("torch is required to save model checkpoints during training result export")
        for model_key in models.keys():
            torch.save(models[model_key].state_dict(), 
                      os.path.join(args.saving_dir, f'{saving_name}-{model_key}-epoch{epoch}.pt'))
    
    print(f"Results saved: {log_file}, {var_file}")
