import os
import sys
from argparse import ArgumentParser
import random

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm, trange

from archipelago.explainer import Archipelago
from archipelago.application_utils.common_utils import get_efficient_mask_indices
from archipelago.application_utils.torch_utils import ModelWrapperTorch, IdXformer

from utils import construct_arg_parser
from models import MLP
from datasets import load_static_dataset


def archipelago_explain(X, y_true, model, device, output_dim=1, top_k=5):
    X = X.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()
    sample_num = X.shape[0]
    explanations = []
    for si in trange(sample_num, desc='explaining'):
    # for si in trange(3):
        Xi = X[si : si + 1]
        model_wrapper = ModelWrapperTorch(model, device, Xi.shape)
        xf = IdXformer(Xi, np.zeros_like(Xi))
        # binary classification torch models give 2-dim output for label 0/1 respectively
        if output_dim == 2:
            apgo = Archipelago(
                model_wrapper, data_xformer=xf, output_indices=1, batch_size=256
            )
        elif output_dim == 1:
            apgo = Archipelago(
                model_wrapper, data_xformer=xf, output_indices=0, batch_size=256
            )
        explanation = apgo.explain(top_k=top_k)
        explanation_sorted = sorted(explanation.items(), key=lambda item: -abs(item[1]))
        explanations.append(explanation_sorted)
    return explanations


def save_explanations(explanations, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'explanation.csv'), 'w') as f:
        for si, explanation in enumerate(explanations):
            for s, v in explanation:
                sliststr = '-'.join([str(s_) for s_ in s])
                f.write(f"{si},{sliststr},{v}\n")


def main():
    parser = construct_arg_parser()
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--save_explanation_dir", type=str, default="")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    expname = (
        f"{args.dataset_name}-{args.model_type}" if not args.expname else args.expname
    )

    device = torch.device(args.device)

    dataset_collection = load_static_dataset(args)

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
    model.load_state_dict(torch.load(args.load_model))
    model.to(device)
    
    X = dataset_collection.data['X']
    y_true = dataset_collection.data['y']
    explanations = archipelago_explain(X, y_true, model, device, dataset_collection.output_dim, args.top_k)

    save_explanations(explanations, args.save_explanation_dir)


if __name__ == "__main__":
    main()
