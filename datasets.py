import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StaticDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class StaticDatasetCollection():
    '''
    Dataset for static (non-temporal) data.
    '''
    def __init__(self, dataset_name, raw_data_path, split_params, normalize=True):
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.normalize = normalize
        self.norm_params = None
        self.split_params = split_params
        self.load_data()
        self.split_data()

        self.train_dataset = StaticDataset(self.train_data)
        self.val_dataset = StaticDataset(self.val_data)
        self.test_dataset = StaticDataset(self.test_data)

    def load_data(self):
        X0 = np.genfromtxt(self.raw_data_path, delimiter=',', skip_header=1)
        X, y = X0[:, 0:274], X0[:, 274:275]
        if self.normalize:
            self.norm_params = {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0, ddof=1)
            }
            X -= self.norm_params['mean']
            X /= self.norm_params['std']

        self.data = {
            'X': torch.from_numpy(X).float(),
            'y': torch.from_numpy(y).float()
        }
        self.input_dim = X.shape[1]
        self.output_dim = 1

    def split_data(self):
        assert 'train' in self.split_params and 'test' in self.split_params
        if isinstance(self.split_params['train'], float):
            n_train = int(self.split_params['train'] * len(self.data['y']))
        elif isinstance(self.split_params['train'], int):
            n_train = self.split_params['train']
        else:
            raise ValueError('Invalid type for train split parameter')

        if 'val' in self.split_params:
            if isinstance(self.split_params['val'], float):
                n_val = int(self.split_params['val'] * len(self.data['y']))
            elif isinstance(self.split_params['val'], int):
                n_val = self.split_params['val']
            else:
                raise ValueError('Invalid type for val split parameter')
        else:
            n_val = 0
        
        n_test = len(self.data['y']) - n_train - n_val

        shuffled_indices = np.random.permutation(len(self.data['y']))
        train_indices = shuffled_indices[:n_train]
        val_indices = shuffled_indices[n_train:n_train+n_val] if n_val > 0 else []
        test_indices = shuffled_indices[n_train+n_val:]

        train_data = {k: v[train_indices] for k, v in self.data.items()}
        test_data = {k: v[test_indices] for k, v in self.data.items()}
        if val_indices:
            val_data = {k: v[val_indices] for k, v in self.data.items()}
        else:
            val_data = test_data

        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data


def load_static_dataset(dataset_name):
    if dataset_name == 'human_microbiome':
        raw_data_path = 'DeepLINK/Real_data_analyses/human_microbiome/data/microbiome_data_common.csv'
        split_params = {
            'train': 147,
            'test': 37
        }
        normalize = True
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')
    dataset_collection = StaticDatasetCollection(dataset_name, raw_data_path, split_params, normalize=normalize)
    return dataset_collection
