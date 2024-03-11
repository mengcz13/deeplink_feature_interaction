import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim,
                 layer_num, hidden_dims, activation, dropout):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.hidden_dims = [int(x) for x in hidden_dims.split('-')]
        if len(self.hidden_dims) == 1:
            self.hidden_dims = self.hidden_dims * (layer_num - 1)
        self.activation = activation
        self.dropout = dropout

        activation_layer = getattr(nn, self.activation)

        self.layers = nn.ModuleList()
        for i, h_dim in enumerate(self.hidden_dims):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, h_dim))
            else:
                self.layers.append(nn.Linear(self.hidden_dims[i-1], h_dim))
            self.layers.append(activation_layer())
            self.layers.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Linear(self.hidden_dims[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
