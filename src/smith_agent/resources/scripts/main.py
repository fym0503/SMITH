import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import os
import smith.losses as losses
from smith.datasets import get_dataset, split_dataset, get_prior_idx
from smith.eval import save_results
import smith.model_selector as model_selector
from smith.min_norm_solvers import MinNormSolver, gradient_normalizers
import argparse
import warnings
import random

warnings.filterwarnings('ignore')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_batch(model, feed_dict, tasks, loss_fn, optimizer, mask, masks, train=True):
    input_data = feed_dict['input']
    labels = {t: feed_dict[t] for t in tasks[1:]}
    
    if train:
        optimizer.zero_grad()
        rep, mask = model['rep'](Variable(input_data.data, volatile=True), mask)
        rep = rep.float()
        
        rep_var = Variable(rep.data.clone(), requires_grad=True)
        grads, loss_data = {}, {}
        
        for t in tasks[1:]:
            optimizer.zero_grad()
            out_t, masks[t] = model[t](rep_var, None)
            reg = torch.mean(model['rep'].reg((model['rep'].mu + 0.5)/model['rep'].sigma))
            loss = loss_fn[t](out_t, labels[t]) + model['rep'].lam * reg
            loss_data[t] = loss.item()
            loss.backward()
            grads[t] = [Variable(rep_var.grad.data.clone(), requires_grad=False)]
            rep_var.grad.data.zero_()
        
        gn = gradient_normalizers(grads, loss_data, 'loss+')
        for t in tasks[1:]:
            grads[t][0] = grads[t][0] / gn[t]
        
        scale = {}
        if len(tasks) > 2:
            sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in tasks[1:]])
            for i, t in enumerate(tasks[1:]):
                scale[t] = float(sol[i])
        else:
            for t in tasks[1:]:
                scale[t] = 1
        
        optimizer.zero_grad()
        rep, _ = model['rep'](input_data, mask)
        total_loss = None
        losses_dict = {}
        
        for i, t in enumerate(tasks[1:]):
            out_t, _ = model[t](rep, masks[t])
            reg = torch.mean(model['rep'].reg((model['rep'].mu + 0.5)/model['rep'].sigma))
            loss_t = loss_fn[t](out_t, labels[t]) + model['rep'].lam * reg
            losses_dict[t] = loss_fn[t](out_t, labels[t]).detach().cpu().numpy()
            
            scaled_loss = scale[t] * loss_t
            total_loss = scaled_loss if total_loss is None else total_loss + scaled_loss
        
        total_loss.backward()
        optimizer.step()
        return losses_dict
    
    else:
        with torch.no_grad():
            rep, _ = model['rep'](input_data, None)
            losses_dict = {}
            for t in tasks[1:]:
                out_t, _ = model[t](rep, None)
                losses_dict[t] = loss_fn[t](out_t, labels[t]).detach().cpu().numpy()
            return losses_dict

def Smith(args):
    var, tasks, datas = get_dataset(args)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.saving_dir, exist_ok=True)
    
    train_dataloader, val_dataloader = split_dataset(tasks, datas, args)
    loss_fn = losses.get_loss(args)
    model = model_selector.get_model(tasks, datas, args)
    
    model_params = [p for m in model.values() for p in m.parameters()]
    optimizer = model_selector.get_optimizer(model_params, args)
    
    if args.prior_file != "0":
        prior_idx = get_prior_idx(args, var)
        for idx in prior_idx:
            model['rep'].mu.data[idx] = 10
    
    for epoch in tqdm(range(args.epoch)):
        print(f'Epoch {epoch} Started')
        
        for m in model.values():
            m.train()
        
        epoch_losses = {t: [] for t in tasks[1:]}
        mask, masks = None, {}
        
        for feed_dict in train_dataloader:
            batch_losses = process_batch(model, feed_dict, tasks, loss_fn, optimizer, mask, masks, True)
            for t in tasks[1:]:
                epoch_losses[t].append(batch_losses[t])
        
        epoch_val_losses = {t: [] for t in tasks[1:]}
        if args.val:
            for m in model.values():
                m.eval()
            for feed_dict in val_dataloader:
                batch_losses = process_batch(model, feed_dict, tasks, loss_fn, optimizer, mask, masks, False)
                for t in tasks[1:]:
                    epoch_val_losses[t].append(batch_losses[t])
        
        train_losses = [np.mean(epoch_losses[t]) for t in tasks[1:]]
        val_losses = [np.mean(epoch_val_losses[t]) for t in tasks[1:]] if args.val else []
        
        for i, t in enumerate(tasks[1:]):
            print(f'Task {t} Loss - Train: {train_losses[i]:.4f}' + 
                  (f', Val: {val_losses[i]:.4f}' if args.val else ''))
        
        if epoch % args.record == args.record - 1:
            save_results(epoch, var, model, args, tasks, train_losses, val_losses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smith')
    parser.add_argument('--adata_file', type=str, required=True)
    parser.add_argument('--saving_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--tasks', type=str, required=True)
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--hvg', action='store_true')
    parser.add_argument('--layer', type=str, default='raw')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--rep_dim', type=int, default=32)
    parser.add_argument('--rep_hidden_dims', type=str, default='32')
    parser.add_argument('--head_hidden_dims', type=str, default='')
    parser.add_argument('--panel_size', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--record', type=int, default=200)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--prior_file', type=str, default='0')
    parser.add_argument('--hurdle', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--evaluate_epoch', type=int, default=10)
    parser.add_argument('--balance_mode', type=str, default='mean', choices=['off', 'mean', 'capped'])
    parser.add_argument('--balance_cap', type=int, default=500)
    
    args = parser.parse_args()
    if args.seed:
        set_random_seed(args.seed)
    
    Smith(args)
