import os
import numpy as np


def train_grid_search():
    params = {
        'layer_num': [2, 3, 4],
        'hidden_dims': [20, 30, 40, 50, 100],
        'activation': ['ELU', 'ReLU', 'LeakyReLU'],
        'dropout': [0.0, 0.1, 0.2, 0.3],
        'batch_size': [1, 2, 4, 8, 16, 32, 64],
        'lr': [1e-2, 1e-3, 1e-4],
        'l1_weight': [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        'l2_weight': [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        'nrep': list(range(100))
    }
    command_template = 'python train_pred.py --dataset_name murine_sc_RNAseq_topf --nrep {nrep} --model_type MLP --layer_num {layer_num} --hidden_dims "{hidden_dims}" --activation {activation} --dropout {dropout} --batch_size {batch_size} --epochs 500 --lr {lr} --l1_weight {l1_weight} --l2_weight {l2_weight} --early_stopping_patience 50 --exptag murine_sc_RNAseq_param_search'

    random_exp_num = 100
    exp_i = 0

    commands = []
    while exp_i < random_exp_num:
        exp_i += 1
        nrep = np.random.choice(params['nrep'])
        layer_num = np.random.choice(params['layer_num'])
        hidden_dims = np.random.choice(params['hidden_dims'])
        activation = np.random.choice(params['activation'])
        dropout = np.random.choice(params['dropout'])
        batch_size = np.random.choice(params['batch_size'])
        lr = np.random.choice(params['lr'])
        l1_weight = np.random.choice(params['l1_weight'])
        l2_weight = np.random.choice(params['l2_weight'])

        command = command_template.format(
            nrep=nrep,
            layer_num=layer_num,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            batch_size=batch_size,
            lr=lr,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
        )
        commands.append(command)
    print('\n'.join(commands))


if __name__ == '__main__':
    train_grid_search()