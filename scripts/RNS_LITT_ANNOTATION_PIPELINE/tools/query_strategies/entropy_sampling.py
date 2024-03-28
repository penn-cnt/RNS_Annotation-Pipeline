import numpy as np
import torch
from .strategy import Strategy


class EntropySampling(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(EntropySampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs)
        uncertainties = (probs * log_probs).sum(1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]


class EntropySamplingRNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(EntropySamplingRNS, self).__init__(dataset, net, args_input, args_task)

    def query(self, n, index=None):
        unlabeled_idxs, unlabeled_data = self.dataset.get_train_data_unaugmented()
        probs, seq_len = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs)
        uncertainties = (probs * log_probs).sum(1)
        to_select = self.metrics_distribution_rescaling(uncertainties, seq_len, unlabeled_idxs, n)
        unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
        print('selected', np.sum(to_select))

        assert len(to_select) == len(unlabeled_idxs)
        return unlabeled_idxs[to_select.astype(bool)]
