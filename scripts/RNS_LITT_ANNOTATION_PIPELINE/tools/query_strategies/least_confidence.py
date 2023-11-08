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
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        uncertainties = probs.max(1)[0]
        index_train_stack = np.hstack(index[unlabeled_idxs])
        assert uncertainties.size()[0] == index_train_stack.shape[0]

        metric_array = torch.empty(len(unlabeled_idxs))

        for i, ind in enumerate(unlabeled_idxs):
            required_slice_ind = np.unique(index[ind]['episode_index'])[0]
            index_location = np.where(index_train_stack['episode_index'] == required_slice_ind)[0]
            metric_array[i] = torch.mean(uncertainties[index_location])

        return unlabeled_idxs[torch.sort(metric_array)[1][:n]]

