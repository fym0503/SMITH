from .stgmodel import STGCommonModel, MLPclassify, MLPModel
import numpy as np
import torch
from torch.optim import Adam, SGD, RMSprop


def _parse_hidden_dims(raw_value, fallback=None):
    if raw_value is None:
        return list(fallback or [])
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return list(fallback or [])
        return [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if isinstance(raw_value, int):
        return [raw_value]
    return list(raw_value)

def get_model(tasks, datas, args):
    device = torch.device(args.device)
    model = {}
    input_data = datas[0]
    rep_dim = getattr(args, 'rep_dim', 32)
    rep_hidden_dims = _parse_hidden_dims(getattr(args, 'rep_hidden_dims', '32'), fallback=[32])
    head_hidden_dims = _parse_hidden_dims(getattr(args, 'head_hidden_dims', ''), fallback=[args.dim])

    model['rep'] = STGCommonModel(
        input_data.shape[1],
        rep_dim,
        rep_hidden_dims,
        device=device,
        lam=args.lam,
        sigma=args.sigma,
    )
    for (task, data) in zip(tasks[1:],datas[1:]):
        if task == 'cls':
            model[task] = MLPclassify(rep_dim, len(np.unique(data)), hidden_dims=head_hidden_dims, dropout=True, dropout_rate=args.dropout_rate, activation=args.activation).to(device)
        elif task == 'recon':
            if args.hurdle:
                model[task] = MLPModel(rep_dim, data.shape[1]*2, hidden_dims=head_hidden_dims, dropout=True, dropout_rate=args.dropout_rate,activation=args.activation).to(device)
            else:
                model[task] = MLPModel(rep_dim, data.shape[1], hidden_dims=head_hidden_dims, dropout=True, dropout_rate=args.dropout_rate,activation=args.activation).to(device)
        elif task == 'coo':
            model[task] = MLPModel(rep_dim, data.shape[1], hidden_dims=head_hidden_dims, dropout=True, dropout_rate=args.dropout_rate,activation=args.activation).to(device)
        elif task == 'region':
            model[task] = MLPclassify(rep_dim, len(np.unique(data)), hidden_dims=head_hidden_dims, dropout=True, dropout_rate=args.dropout_rate,activation=args.activation).to(device)
        elif task == 'pathology':
            model[task] = MLPclassify(rep_dim, len(np.unique(data)), hidden_dims=head_hidden_dims, dropout=True, dropout_rate=args.dropout_rate,activation=args.activation).to(device)
    return model

def get_optimizer(model_params, args):
    if 'RMSprop' == args.optimizer:
        optimizer = RMSprop(model_params, lr=args.learning_rate)
    elif 'Adam' == args.optimizer:
        optimizer = Adam(model_params, lr=args.learning_rate)
    elif 'SGD' == args.optimizer:
        optimizer = SGD(model_params, lr=args.learning_rate, momentum=0.9)
    return optimizer
