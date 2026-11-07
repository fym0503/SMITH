import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, explained_variance_score
from scipy.stats import pearsonr
import argparse
import os


def load_panel_genes(panel_path, panel_size=32):
    """Load the ordered marker CSV exported by SMITH."""
    if panel_size <= 0:
        raise ValueError('panel_size must be positive')
    table = pd.read_csv(panel_path)
    genes = table.iloc[:panel_size, 0].dropna().astype(str).str.upper().tolist()
    if len(genes) != panel_size or len(set(genes)) != panel_size:
        raise ValueError('Panel must contain the requested number of distinct genes')
    return genes


def _split_panel_matrices(train_adata, test_adata, selected_genes):
    matrices = []
    for adata in (train_adata, test_adata):
        if not adata.var_names.is_unique:
            raise ValueError('Gene names must be unique')
        missing = [g for g in selected_genes if g not in adata.var_names]
        if missing:
            raise ValueError(f'Selected genes absent from evaluation input: {missing}')
        values = adata[:, selected_genes].X
        matrices.append(values.toarray() if sparse.issparse(values) else np.asarray(values))
    return matrices


def evaluate_split_knn_classification(train_adata, test_adata, selected_genes,
                                      label_name, n_neighbors=5):
    """Fit only on source cells and score the supplied held-out target cells."""
    train_x, test_x = _split_panel_matrices(train_adata, test_adata, selected_genes)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(train_x, train_adata.obs[label_name].to_numpy())
    predicted = model.predict(test_x)
    observed = test_adata.obs[label_name].to_numpy()
    return accuracy_score(observed, predicted), classification_report(
        observed, predicted, output_dict=True, zero_division=0,
    )


def evaluate_split_obsm_regression(train_adata, test_adata, selected_genes, obsm_key):
    train_x, test_x = _split_panel_matrices(train_adata, test_adata, selected_genes)
    train_y = np.asarray(train_adata.obsm[obsm_key])
    test_y = np.asarray(test_adata.obsm[obsm_key])
    model = LinearRegression().fit(train_x, train_y)
    predicted = model.predict(test_x)
    # Match the existing eval_regression convention: pooled output Pearson r.
    return explained_variance_score(test_y, predicted), pearsonr(
        test_y.flatten(), predicted.flatten(),
    )[0]


def evaluate_split_time_knn_regression(train_adata, test_adata, selected_genes,
                                      time_label, n_neighbors=5):
    train_x, test_x = _split_panel_matrices(train_adata, test_adata, selected_genes)
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(train_x, train_adata.obs[time_label].to_numpy(dtype=float))
    return pearsonr(
        test_adata.obs[time_label].to_numpy(dtype=float), model.predict(test_x),
    )[0]

def downsample_adata(adata, target_n=50000, random_state=42):
    if target_n >= adata.n_obs:
        return adata.copy()
    np.random.seed(random_state)
    sampled_indices = np.random.choice(adata.n_obs, size=target_n, replace=False)
    return adata[sampled_indices].copy()

def calculate_knn_accuracy(embedding, labels, label_name, n_neighbors=5, test_size=0.2, random_state=42):
    y = labels[label_name].values
    X_train, X_test, y_train, y_test = train_test_split(embedding, y, test_size=test_size, random_state=random_state)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

def evaluate_reconstruction(adata, selected_genes, test_size=0.2, random_state=42):
    selected_genes = [g.upper() for g in selected_genes]
    all_genes = adata.var_names.tolist()
    unselected_genes = [g for g in all_genes if g not in selected_genes]
    
    if len(unselected_genes) == 0:
        return None, None
    
    X = adata[:, selected_genes].X
    y = adata[:, unselected_genes].X
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    ev = explained_variance_score(y_test, y_pred)
    corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())
    
    return ev, corr

def eval_regression(adata, selected_genes, obsm_key, test_size=0.2, random_state=42):
    selected_genes = [g.upper() for g in selected_genes]
    
    X = adata[:, selected_genes].X
    y = adata.obsm[obsm_key]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    ev = explained_variance_score(y_test, y_pred)
    corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())
    
    return ev, corr


def save_results(epoch, var, models, args, tasks, train_loss, val_loss):
    array = models['rep'].mu.detach().cpu().numpy()
    index = np.argsort(array)[::-1]
    if getattr(args, 'prior_file', '0') != '0':
        from smith.datasets import get_prior_idx
        prior_indices = get_prior_idx(args, var)
        if len(prior_indices) > args.panel_size:
            raise ValueError('Required prior genes exceed the panel budget')
        prior_set = set(prior_indices)
        index = np.asarray(prior_indices + [i for i in index if i not in prior_set])
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
        for model_key in models.keys():
            torch.save(models[model_key].state_dict(), 
                      os.path.join(args.saving_dir, f'{saving_name}-{model_key}-epoch{epoch}.pt'))
    
    print(f"Results saved: {log_file}, {var_file}")
