import torch
import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import resample


def _infer_celltype_series(adata):
    for column in ("celltype", "cell_type", "subclass"):
        if column in adata.obs:
            return adata.obs[column].astype(str)
    return None


def _spatial_strata(adata, n_bins=8):
    if "spatial" not in adata.obsm:
        return None
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        return None
    try:
        x_bins = pd.qcut(spatial[:, 0], q=min(n_bins, np.unique(spatial[:, 0]).size), labels=False, duplicates="drop")
        y_bins = pd.qcut(spatial[:, 1], q=min(n_bins, np.unique(spatial[:, 1]).size), labels=False, duplicates="drop")
    except ValueError:
        return None
    return pd.Series(x_bins).astype(str) + "_" + pd.Series(y_bins).astype(str)


def _sampling_labels(adata, strategy):
    if strategy == "random":
        return None
    celltype = _infer_celltype_series(adata)
    spatial = _spatial_strata(adata)
    if strategy == "celltype":
        return celltype
    if strategy == "spatial":
        return spatial
    if strategy == "celltype_spatial":
        if celltype is None:
            return spatial
        if spatial is None:
            return celltype
        return celltype.reset_index(drop=True) + "|" + spatial
    raise ValueError(f"Unknown sampling strategy: {strategy}")


def _subsample_indices(labels, n_obs, max_cells, seed):
    rng = np.random.default_rng(seed)
    if labels is None:
        return np.sort(rng.choice(n_obs, size=max_cells, replace=False))
    labels = np.asarray(labels.astype(str))
    unique, counts = np.unique(labels, return_counts=True)
    allocations = np.maximum(1, np.floor(counts / counts.sum() * max_cells).astype(int))
    while allocations.sum() > max_cells:
        index = int(np.argmax(allocations))
        if allocations[index] == 1:
            break
        allocations[index] -= 1
    while allocations.sum() < max_cells:
        room = counts - allocations
        index = int(np.argmax(room))
        if room[index] <= 0:
            break
        allocations[index] += 1
    chosen = [rng.choice(np.flatnonzero(labels == label), size=int(n), replace=False) for label, n in zip(unique, allocations)]
    return np.sort(np.concatenate(chosen))

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, tensor_names, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param tensor_names: name of tensors (for feed_dict)
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.tensor_names = tensor_names

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = {}
        for k in range(len(self.tensor_names)):
            batch.update({self.tensor_names[k]: self.tensors[k][self.i:self.i+self.batch_size]})
        self.i += self.batch_size
        return batch
        

    def __len__(self):
        return self.n_batches

def balance_datas(datas, label_indices, max_samples=None, random_state=42):
    if not label_indices:
        return datas
    
    labels = pd.DataFrame({i: datas[i] for i in label_indices})
    combined = labels.apply(lambda x: "-".join(x.astype(str)), axis=1)
    counts = combined.value_counts()
    
    if max_samples is None:
        max_samples = int(counts.mean())
    
    indices = []
    for label in counts.index:
        mask = combined == label
        subset_idx = np.where(mask)[0]
        
        if len(subset_idx) != max_samples:
            subset_idx = resample(subset_idx, replace=len(subset_idx) < max_samples,
                                n_samples=max_samples, random_state=random_state)
        indices.extend(subset_idx)
    
    return [np.array(data)[indices] for data in datas]


def maybe_balance_datas(datas, label_indices, args):
    mode = getattr(args, "balance_mode", "mean")
    if mode == "off" or not label_indices:
        return datas

    if mode == "mean":
        return balance_datas(datas, label_indices, max_samples=None, random_state=getattr(args, "seed", 42) or 42)

    if mode == "capped":
        balance_cap = getattr(args, "balance_cap", None)
        if balance_cap is None:
            raise ValueError("balance_cap must be set when balance_mode='capped'")
        return balance_datas(datas, label_indices, max_samples=int(balance_cap), random_state=getattr(args, "seed", 42) or 42)

    raise ValueError(f"Unknown balance_mode: {mode}")

def get_dataset(args):
    adata = ad.read_h5ad(args.adata_file)
    adata.var_names_make_unique()
    max_cells = getattr(args, "max_cells", None)
    if max_cells and adata.n_obs > int(max_cells):
        labels = _sampling_labels(adata, getattr(args, "sampling_strategy", "random"))
        indices = _subsample_indices(labels, adata.n_obs, int(max_cells), getattr(args, "seed", 42) or 42)
        adata = adata[indices].copy()
    
    x = adata.X if args.layer == 'raw' else adata.layers[args.layer].copy()
    if isinstance(x, csr_matrix):
        x = x.toarray()
    
    if args.hvg:
        try:
            import scanpy as sc
        except ImportError as error:
            raise ImportError("--hvg requires the optional `scanpy` dependency: pip install 'smith-panel-design[scanpy]'") from error
        if args.layer == 'raw':
            sc.pp.highly_variable_genes(adata, n_top_genes=10000)
        hvg_mask = adata.var['highly_variable'].values
        x = x[:, hvg_mask]
        var = adata.var.index[hvg_mask]
    else:
        var = adata.var.index
    
    tasks = args.tasks.split(",")
    datas, new_tasks, balanced_idx = [x], ['input'], []
    counter = 1
    
    if 'recon' in tasks:
        datas.append(x)
        new_tasks.append('recon')
        counter += 1
    
    for task, col_options in [('cls', ['celltype', 'cell_type', 'subclass']),
                             ('region', ['region']), ('pathology', ['pathology'])]:
        if task in tasks:
            for col in col_options:
                if col in adata.obs.columns:
                    datas.append(adata.obs[col].astype('category').cat.codes.values)
                    new_tasks.append(task)
                    balanced_idx.append(counter)
                    counter += 1
                    break

    if 'time' in tasks:
        candidates = [getattr(args, "time_label", None), "absolute_time", "consensus_time", "time"]
        for column in [item for item in candidates if item]:
            if column in adata.obs:
                values = pd.to_numeric(adata.obs[column], errors="coerce").to_numpy(dtype=np.float32)
                if np.isnan(values).any():
                    raise ValueError(f"obs[{column!r}] contains missing or non-numeric time values")
                datas.append(values.reshape(-1, 1))
                new_tasks.append('time')
                break
        else:
            raise KeyError(f"time task requested but no time column was found: {candidates}")
    
    if any(t in tasks for t in ["coordination", "standard_coordination"]):
        spatial = np.asarray(adata.obsm['spatial'], dtype=np.float32)
        if 'standard_coordination' in tasks:
            mean = spatial.mean(axis=0, keepdims=True)
            std = np.where(spatial.std(axis=0, keepdims=True) > 0, spatial.std(axis=0, keepdims=True), 1.0)
            spatial = (spatial - mean) / std
        datas.append(spatial)
        new_tasks.append('coo')
    
    return var, new_tasks, maybe_balance_datas(datas, balanced_idx, args)

def split_dataset(tasks, datas, args, ratio=0.2):
    def to_tensor(data):
        return torch.from_numpy(data).float() if np.issubdtype(data.dtype, np.floating) \
               else torch.from_numpy(data).long()
    
    if args.val:
        n = len(datas[0])
        rng = np.random.default_rng(getattr(args, "seed", 42) or 42)
        val_idx = rng.choice(n, int(n * ratio), replace=False)
        train_idx = np.setdiff1d(np.arange(n), val_idx)
        
        train_data = [data[train_idx] for data in datas]
        val_data = [data[val_idx] for data in datas]
        
        train_loader = FastTensorDataLoader(*[to_tensor(d) for d in train_data],
                                          tensor_names=tasks, batch_size=args.batch_size, shuffle=True)
        val_loader = FastTensorDataLoader(*[to_tensor(d) for d in val_data],
                                        tensor_names=tasks, batch_size=args.batch_size, shuffle=True)
        return train_loader, val_loader
    
    train_loader = FastTensorDataLoader(*[to_tensor(d) for d in datas],
                                      tensor_names=tasks, batch_size=args.batch_size, shuffle=True)
    return train_loader, None

def get_prior_idx(args, var):
    genes = pd.read_csv(args.prior_file)['names'].tolist()
    var_list = var.tolist()
    return [var_list.index(gene) for gene in genes if gene in var_list]
