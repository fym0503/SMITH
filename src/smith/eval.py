import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, explained_variance_score
from scipy.stats import pearsonr
import scanpy as sc
import argparse
import os

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
