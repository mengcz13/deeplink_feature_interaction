import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.optim as optim

from archipelago.explainer import Archipelago
from archipelago.application_utils.common_utils import get_efficient_mask_indices


class ModelWrapperTorch:
    def __init__(self, model, device, source_input_shape):
        self.device = device
        self.model = model.to(device)
        self.source_input_shape = list(source_input_shape)
        self.source_input_shape[0] = -1  # (-1, F) or (-1, T, F)

    def __call__(self, X):  # ArchDetect use flattened feature vectors as input
        X = torch.FloatTensor(
            X.reshape(self.source_input_shape)).to(self.device)
        preds = self.model(X).data.cpu().numpy()
        if len(preds.shape) == 1:
            preds = preds[:, np.newaxis]
        return preds


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


def archdetect_interpreter(X, y_true, model, device, output_dim=2):
    X = X.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()
    sample_num = X.shape[0]
    imp_scores = []
    for si in range(sample_num):
        Xi = X[si:si + 1]
        model_wrapper = ModelWrapperTorch(model, device, Xi.shape)
        xf = IdXformer(Xi, np.zeros_like(Xi))
        # binary classification torch models give 2-dim output for label 0/1 respectively
        if output_dim == 2:
            apgo = Archipelago(model_wrapper, data_xformer=xf,
                            output_indices=1, batch_size=256)
        elif output_dim == 1:
            apgo = Archipelago(model_wrapper, data_xformer=xf, output_indices=0, batch_size=256)
        main_strengths = apgo.archdetect_idv_strength()
        main_strengths_arr = np.zeros(len(main_strengths))
        for k in range(len(main_strengths_arr)):
            main_strengths_arr[k] = main_strengths[k]
        imp_scores.append(main_strengths_arr.reshape(model_wrapper.source_input_shape))
    imp_scores = np.concatenate(imp_scores, axis=0)
    return imp_scores
