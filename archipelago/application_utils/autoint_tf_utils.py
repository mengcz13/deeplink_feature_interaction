import numpy as np
from bm_interpreters.archipelago.application_utils.common_utils import get_efficient_mask_indices


class AutoIntWrapper:
    def __init__(self, model, Xi_inst, inv_sigmoid=True):
        self.model = model
        self.Xi_inst = Xi_inst
        self.use_inv_sigmoid = inv_sigmoid

    def inv_sigmoid(self, y):
        return np.log(y / (1 - y))

    def __call__(self, Xv):
        Xi = np.repeat(self.Xi_inst, Xv.shape[0], axis=0)
        pred = self.model.predict(Xi, Xv)
        if self.use_inv_sigmoid:
            pred = self.inv_sigmoid(pred)
        return np.expand_dims(pred, 1)


class IdXformer:
    def __init__(self, input_ids, baseline_ids):
        self.input = input_ids.flatten()
        self.baseline = baseline_ids.flatten()
        self.num_features = len(self.input)

    def efficient_xform(self, inst):
        mask_indices, base, change = get_efficient_mask_indices(
            inst, self.baseline, self.input
        )
        for i in mask_indices:
            base[i] = change[i]
        return base

    def __call__(self, inst):
        id_list = self.efficient_xform(inst)
        return id_list


class get_args:
    # the original parameter configuration of AutoInt
    blocks = 3
    block_shape = [64, 64, 64]
    heads = 2
    embedding_size = 16
    dropout_keep_prob = [1, 1, 0.5]
    epoch = 5
    batch_size = 32
    learning_rate = 0.0005
    learning_rate_wide = 0.001
    optimizer_type = "adam"
    l2_reg = 0.0
    random_seed = 2018  # used in the official autoint code
    loss_type = "logloss"
    verbose = 1
    run_times = 1
    is_save = False
    greater_is_better = False
    has_residual = True
    has_wide = False
    deep_layers = None
    batch_norm = 0
    batch_norm_decay = 0.995

    def __init__(self, save_path, field_size, dataset, data_path):
        self.save_path = save_path
        self.field_size = field_size
        self.data = dataset
        self.data_path = data_path
