import torch
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import resample


def _infer_celltype_series(adata: ad.AnnData) -> pd.Series | None:
    for col in ["celltype", "cell_type", "subclass"]:
        if col in adata.obs.columns:
            return adata.obs[col].astype(str)
    return None


def _make_spatial_strata(adata: ad.AnnData, n_bins: int = 8) -> pd.Series | None:
    if "spatial" not in adata.obsm:
        return None
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        return None
    x = pd.Series(spatial[:, 0], index=adata.obs_names)
    y = pd.Series(spatial[:, 1], index=adata.obs_names)
    try:
        x_bins = pd.qcut(x, q=min(n_bins, x.nunique()), labels=False, duplicates="drop")
        y_bins = pd.qcut(y, q=min(n_bins, y.nunique()), labels=False, duplicates="drop")
    except ValueError:
        return None
    return x_bins.astype(str) + "_" + y_bins.astype(str)


def _build_sampling_labels(adata: ad.AnnData, strategy: str) -> pd.Series | None:
    if strategy == "random":
        return None
    if strategy == "celltype":
        return _infer_celltype_series(adata)
    if strategy == "spatial":
        return _make_spatial_strata(adata)
    if strategy == "celltype_spatial":
        celltype = _infer_celltype_series(adata)
        spatial = _make_spatial_strata(adata)
        if celltype is None and spatial is None:
            return None
        if celltype is None:
            return spatial
        if spatial is None:
            return celltype
        return celltype.astype(str) + "|" + spatial.astype(str)
    raise ValueError(f"Unknown sampling_strategy: {strategy}")


def _stratified_subsample_indices(labels: pd.Series, max_cells: int, seed: int) -> np.ndarray:
    counts = labels.value_counts()
    total = int(counts.sum())
    raw = counts / total * max_cells
    alloc = np.floor(raw).astype(int)
    alloc = np.minimum(alloc, counts)

    zero_mask = (alloc == 0) & (counts > 0)
    alloc.loc[zero_mask] = 1

    current = int(alloc.sum())
    if current > max_cells:
        overflow = current - max_cells
        reducible = alloc[alloc > 1].sort_values(ascending=False)
        for key in reducible.index:
            if overflow <= 0:
                break
            step = min(int(alloc.loc[key] - 1), overflow)
            alloc.loc[key] -= step
            overflow -= step
    elif current < max_cells:
        remainder = max_cells - current
        frac = (raw - np.floor(raw)).sort_values(ascending=False)
        for key in frac.index:
            if remainder <= 0:
                break
            room = int(counts.loc[key] - alloc.loc[key])
            if room <= 0:
                continue
            alloc.loc[key] += 1
            remainder -= 1
        if remainder > 0:
            room_left = (counts - alloc).sort_values(ascending=False)
            for key in room_left.index:
                if remainder <= 0:
                    break
                room = int(counts.loc[key] - alloc.loc[key])
                if room <= 0:
                    continue
                step = min(room, remainder)
                alloc.loc[key] += step
                remainder -= step

    rng = np.random.default_rng(seed)
    chosen = []
    for key, n_take in alloc.items():
        if n_take <= 0:
            continue
        idx = np.flatnonzero(labels.to_numpy() == key)
        picked = rng.choice(idx, size=int(n_take), replace=False)
        chosen.append(picked)
    sampled = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    sampled.sort()
    return sampled

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
    forbidden_file = getattr(args, "forbidden_file", "0")
    if forbidden_file != "0":
        forbidden_genes = set(
            pd.read_csv(forbidden_file)["names"].astype(str).str.strip().str.upper()
        )
        forbidden_genes.discard("")
        keep = ~adata.var_names.astype(str).str.upper().isin(forbidden_genes)
        adata = adata[:, np.asarray(keep)].copy()
        if adata.n_vars < int(args.panel_size):
            raise ValueError(
                f"Only {adata.n_vars} candidate genes remain after exclusions, "
                f"fewer than panel_size={args.panel_size}."
            )
    prior_file = getattr(args, "prior_file", "0")
    if prior_file != "0":
        prior_genes = pd.read_csv(prior_file)["names"].astype(str).str.strip().str.upper()
        prior_genes = list(dict.fromkeys(gene for gene in prior_genes if gene))
        available = set(adata.var_names.astype(str).str.upper())
        missing = [gene for gene in prior_genes if gene not in available]
        if missing:
            raise ValueError(f"Required prior genes are absent or excluded: {missing}")
        if len(prior_genes) > int(args.panel_size):
            raise ValueError(
                f"{len(prior_genes)} prior genes exceed panel_size={args.panel_size}."
            )
    max_cells = getattr(args, "max_cells", None)
    if max_cells is not None and adata.n_obs > int(max_cells):
        seed = getattr(args, "seed", 42) or 42
        strategy = getattr(args, "sampling_strategy", "random")
        labels = _build_sampling_labels(adata, strategy)
        if labels is None:
            rng = np.random.default_rng(seed)
            sampled_idx = np.sort(rng.choice(adata.n_obs, size=int(max_cells), replace=False))
        else:
            sampled_idx = _stratified_subsample_indices(labels.astype(str), int(max_cells), seed)
        adata = adata[sampled_idx].copy()
    
    x = adata.X if args.layer == 'raw' else adata.layers[args.layer].copy()
    if isinstance(x, csr_matrix):
        x = x.toarray()
    
    if args.hvg:
        if args.layer == 'raw':
            sc.pp.highly_variable_genes(adata, n_top_genes=10000)
        hvg_mask = adata.var['highly_variable'].values
        if prior_file != "0":
            hvg_mask = hvg_mask | adata.var_names.astype(str).str.upper().isin(prior_genes)
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
        time_candidates = []
        explicit_time_label = getattr(args, "time_label", None)
        if explicit_time_label:
            time_candidates.append(explicit_time_label)
        time_candidates.extend(["absolute_time", "consensus_time", "time"])
        for col in time_candidates:
            if col in adata.obs.columns:
                time_values = pd.to_numeric(adata.obs[col], errors="coerce").to_numpy(dtype=np.float32)
                if np.isnan(time_values).any():
                    raise ValueError(f"time task requested but obs[{col!r}] contains missing/non-numeric values")
                datas.append(time_values.reshape(-1, 1))
                new_tasks.append('time')
                counter += 1
                break
        else:
            raise KeyError(
                f"time task requested but none of the candidate columns were found: {time_candidates}"
            )
    
    if any(t in tasks for t in ["coordination", "standard_coordination"]):
        spatial_key = getattr(args, 'spatial_key', 'spatial')
        spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        if spatial.ndim != 2 or not np.isfinite(spatial).all():
            raise ValueError(f'obsm[{spatial_key!r}] must contain a finite coordinate matrix')
        if "standard_coordination" in tasks:
            # Normalize each spatial axis independently so coordinate regression
            # does not explode on raw slide-scale pixel values.
            mean = spatial.mean(axis=0, keepdims=True)
            std = spatial.std(axis=0, keepdims=True)
            std = np.where(std > 0, std, 1.0)
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
    var_list = [str(gene).upper() for gene in var.tolist()]
    indices = [var_list.index(str(gene).strip().upper()) for gene in genes if str(gene).strip()]
    if len(indices) != len(set(indices)):
        raise ValueError("The prior file contains duplicate gene names.")
    return indices
