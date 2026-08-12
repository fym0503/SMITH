import torch
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import resample

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
    
    x = adata.X if args.layer == 'raw' else adata.layers[args.layer].copy()
    if isinstance(x, csr_matrix):
        x = x.toarray()
    
    if args.hvg:
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
    
    if any(t in tasks for t in ["coordination", "standard_coordination"]):
        datas.append(adata.obsm['spatial'])
        new_tasks.append('coo')
    
    return var, new_tasks, maybe_balance_datas(datas, balanced_idx, args)

def split_dataset(tasks, datas, args, ratio=0.2):
    device = torch.device(args.device)
    
    def to_tensor(data):
        return torch.from_numpy(data).float().to(device) if np.issubdtype(data.dtype, np.floating) \
               else torch.from_numpy(data).long().to(device)
    
    if args.val:
        n = len(datas[0])
        val_idx = np.random.choice(n, int(n * ratio), replace=False)
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
