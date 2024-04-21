import numpy as np
import torch
from .strategy import Strategy

class MeanSTD(Strategy):
    def __init__(self, dataset, net, args_input, args_task, n_drop = 10):
        super(MeanSTD, self).__init__(dataset, net, args_input, args_task)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop).numpy()
        sigma_c = np.std(probs, axis=0)
        uncertainties = torch.from_numpy(np.mean(sigma_c, axis=-1))
        return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]

class MeanSTDRNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task, n_drop = 3):
        super(MeanSTDRNS, self).__init__(dataset, net, args_input, args_task)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs, seq_len = self.predict_prob_dropout_split(unlabeled_data, n_drop=self.n_drop)
        sigma_c = np.std(probs.numpy(), axis=0)
        uncertainties = torch.from_numpy(np.mean(sigma_c, axis=-1))
        uncertainties = self.smoothing_prediction(uncertainties, 8)
        to_select = self.metrics_distribution_rescaling(uncertainties, seq_len, unlabeled_idxs, n, descending=True)
        unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
        print('selected', np.sum(to_select))

        assert len(to_select) == len(unlabeled_idxs)
        return unlabeled_idxs[to_select.astype(bool)]
