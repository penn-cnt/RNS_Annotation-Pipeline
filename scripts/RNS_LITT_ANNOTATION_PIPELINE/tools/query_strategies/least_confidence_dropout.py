import numpy as np
from .strategy import Strategy
import torch

class LeastConfidenceDropout(Strategy):
    def __init__(self, dataset, net, args_input, args_task, n_drop=10):
        super(LeastConfidenceDropout, self).__init__(dataset, net, args_input, args_task)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data, n_drop=self.n_drop)
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]

class LeastConfidenceDropoutRNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task, n_drop=10):
        super(LeastConfidenceDropoutRNS, self).__init__(dataset, net, args_input, args_task)
        self.n_drop = n_drop

    def query(self, n, index = None):
        unlabeled_idxs, unlabeled_data = self.dataset.get_train_data_unaugmented()
        probs, seq_len = self.predict_prob_dropout(unlabeled_data,n_drop=self.n_drop)
        uncertainties = probs.max(1)[0]
        uncertainties = self.smoothing_prediction(uncertainties, 8)

        to_select = self.metrics_distribution_rescaling(uncertainties, seq_len, unlabeled_idxs, n)

        unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
        print('selected', np.sum(to_select))
        assert len(to_select) == len(unlabeled_idxs)
        return unlabeled_idxs[to_select.astype(bool)]