import numpy as np
from .strategy import Strategy
import torch

class LeastConfidence(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(LeastConfidence, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]


class LeastConfidenceRNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(LeastConfidenceRNS, self).__init__(dataset, net, args_input, args_task)

    def query(self, n, index = None):
        unlabeled_idxs, unlabeled_data = self.dataset.get_train_data_unaugmented()
        probs, seq_len = self.predict_prob(unlabeled_data)
        uncertainties = probs.max(1)[0]

        uncertainties, seq_len = self.dataset.get_slice_from_episode(uncertainties, seq_len, ~unlabeled_idxs)
        uncertainties = np.concatenate(uncertainties)

        # to_select = np.ones(1)
        threshold = 0.97
        # while np.sum(to_select)<n:
        #     metrics = self.dataset.combine_window_to_episode(threshold-uncertainties, seq_len)
        #     to_select = self.get_combined_important(torch.flatten(seq_len), metrics, n)
        #     threshold+=0.01

        metrics = self.dataset.combine_window_to_episode(threshold - uncertainties, seq_len)
        to_select = self.get_combined_important(torch.flatten(seq_len), metrics, n)

        unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
        print('selected', np.sum(to_select), threshold)


        assert len(to_select) == len(unlabeled_idxs)



        return unlabeled_idxs[to_select.astype(bool)]

