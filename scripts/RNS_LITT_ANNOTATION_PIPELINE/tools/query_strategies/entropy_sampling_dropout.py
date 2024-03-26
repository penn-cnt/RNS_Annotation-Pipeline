import numpy as np
import torch
from .strategy import Strategy

class EntropySamplingDropout(Strategy):
    def __init__(self, dataset, net, args_input, args_task, n_drop=10):
        super(EntropySamplingDropout, self).__init__(dataset, net, args_input, args_task)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data, n_drop=self.n_drop)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum(1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]

class EntropySamplingDropoutRNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task, n_drop=10):
        super(EntropySamplingDropoutRNS, self).__init__(dataset, net, args_input, args_task)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_train_data_unaugmented()
        probs, seq_len = self.predict_prob_dropout(unlabeled_data, n_drop=2)
        log_probs = torch.log(probs)
        uncertainties = (probs * log_probs).sum(1)

        indices = np.argsort(uncertainties)
        original_order = indices.argsort()
        cdf = np.arange(1, len(uncertainties) + 1) / len(uncertainties)
        linear_data = np.interp(cdf, (cdf.min(), cdf.max()), (0, 1))

        uncertainties_metric = 0.2 - linear_data[original_order]

        uncertainties_metric, seq_len = self.dataset.get_slice_from_episode(uncertainties_metric, seq_len,
                                                                            ~unlabeled_idxs)
        uncertainties_metric = np.concatenate(uncertainties_metric)

        metrics = self.dataset.combine_window_to_episode(uncertainties_metric, seq_len)
        to_select = self.get_combined_important(torch.flatten(seq_len), metrics, n)

        unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
        print('selected', np.sum(to_select), threshold)

        assert len(to_select) == len(unlabeled_idxs)

        return unlabeled_idxs[to_select.astype(bool)]
