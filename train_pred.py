import os
import sys
from argparse import ArgumentParser
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from utils import construct_arg_parser
from models import MLP
from datasets import load_static_dataset


def train_epoch(args, epoch, model, train_dataloader, optimizer, loss_func, device):
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch}'):
        X, y = batch['X'], batch['y']
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_func(y_pred, y)

        l1_penalty = args.l1_weight * sum([p.abs().sum() for p in model.parameters()])
        l2_penalty = args.l2_weight * sum([(p**2).sum() for p in model.parameters()])
        loss_with_penalty = loss + l1_penalty + l2_penalty

        loss_with_penalty.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    metrics = {'train_loss': train_loss}
    return metrics


def eval_epoch(args, epoch, model, val_dataloader, loss_func, device, subset='val'):
    model.eval()
    val_loss, val_num = 0, 0
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, total=len(val_dataloader), desc=f'Epoch {epoch} {subset}'):
            X, y = batch['X'], batch['y']
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_func(y_pred, y)
            val_loss += loss.item() * len(y)
            val_num += len(y)
            all_y_true.append(y.detach().cpu().numpy())
            all_y_pred.append(y_pred.detach().cpu().numpy())
        val_loss /= val_num
    metrics = {f'{subset}_loss': val_loss}

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    if args.loss_type == 'BCEWithLogits':
        all_y_pred_proba = torch.sigmoid(torch.from_numpy(all_y_pred)).numpy()
        all_y_pred = (all_y_pred_proba > 0.5).astype(int)
        metrics.update({
            f'{subset}_accuracy': (all_y_true == all_y_pred).mean(),
            f'{subset}_precision': (all_y_true * all_y_pred).sum() / all_y_pred.sum(),
            f'{subset}_recall': (all_y_true * all_y_pred).sum() / all_y_true.sum(),
            f'{subset}_f1': 2 * (all_y_true * all_y_pred).sum() / (all_y_true.sum() + all_y_pred.sum())
        })
    return metrics


def main():
    parser = construct_arg_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    expname = f'{args.dataset_name}-{args.model_type}' if not args.expname else args.expname
    wandb.init(project=args.wandb_project, name=expname, tags=[args.exptag], config=args)

    device = torch.device(args.device)

    dataset_collection = load_static_dataset(args)
    train_dataloader = DataLoader(dataset_collection.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset_collection.val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset_collection.test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model_type == 'MLP':
        model = MLP(
            input_dim=dataset_collection.input_dim,
            output_dim=dataset_collection.output_dim,
            layer_num=args.layer_num,
            hidden_dims=args.hidden_dims,
            activation=args.activation,
            dropout=args.dropout
        )
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")
    model.to(device)

    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)
    loss_func = getattr(torch.nn, f'{args.loss_type}Loss')()

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        epoch_metrics = {'epoch': epoch}

        train_metrics = train_epoch(args, epoch, model, train_dataloader, optimizer, loss_func, device)
        epoch_metrics.update(train_metrics)
        
        val_metrics = eval_epoch(args, epoch, model, val_dataloader, loss_func, device, subset='val')
        epoch_metrics.update(val_metrics)
        
        wandb.log(epoch_metrics)
        
        if args.early_stopping_patience > 0:
            if epoch_metrics[args.early_stopping_metric] < best_val_loss:
                best_val_loss = epoch_metrics[args.early_stopping_metric]
                best_epoch = epoch
                model_save_dir = f'save/{wandb.run.id}'
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))
            if epoch - best_epoch > args.early_stopping_patience:
                break

    test_metrics = eval_epoch(args, epoch, model, test_dataloader, loss_func, device, subset='test')
    wandb.log(test_metrics)


if __name__ == "__main__":
    main()
