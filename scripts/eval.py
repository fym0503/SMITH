import pandas as pd
import scanpy as sc
import argparse
from smith.eval import calculate_knn_accuracy, evaluate_reconstruction, eval_regression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_panel", required=True)
    parser.add_argument("--adata_file", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--obsm_key", default="X_pca")
    parser.add_argument("--length", default=32, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output", default="evaluation_results.csv")
    args = parser.parse_args()

    with open(args.input_panel, "r") as f:
        gene_panel = [gene.upper() for gene in eval(f.read())[:args.length]]
    
    adata_full = sc.read(args.adata_file)
    adata_full.var_names = adata_full.var_names.str.upper()
    
    results = []
    
    # KNN evaluation
    adata = adata_full[:, gene_panel].copy()
    accuracy, _ = calculate_knn_accuracy(adata.X, adata.obs[[args.label]], args.label, random_state=args.seed)
    results.append({
        'Evaluation': 'KNN',
        'Label': args.label,
        'Accuracy': accuracy,
        'Explained_Variance': None,
        'Pearson_Correlation': None
    })
    print(f"KNN - Label: {args.label}, Accuracy: {accuracy:.4f}")
    
    # Reconstruction evaluation
    ev, corr = evaluate_reconstruction(adata_full, gene_panel, random_state=args.seed)
    if ev is not None:
        results.append({
            'Evaluation': 'Reconstruction',
            'Label': None,
            'Accuracy': None,
            'Explained_Variance': ev,
            'Pearson_Correlation': corr
        })
        print(f"Reconstruction - Explained Variance: {ev:.4f}, Pearson Correlation: {corr:.4f}")
    else:
        print("No unselected genes for reconstruction evaluation")
    
    # Regression evaluation
    if args.obsm_key in adata_full.obsm:
        ev_reg, corr_reg = eval_regression(adata_full, gene_panel, args.obsm_key, random_state=args.seed)
        results.append({
            'Evaluation': 'Regression',
            'Label': None,
            'Accuracy': None,
            'Explained_Variance': ev_reg,
            'Pearson_Correlation': corr_reg
        })
        print(f"Regression - Explained Variance: {ev_reg:.4f}, Pearson Correlation: {corr_reg:.4f}")
    else:
        print(f"obsm key '{args.obsm_key}' not found in adata")
    
    pd.DataFrame(results).to_csv(args.output, index=False)