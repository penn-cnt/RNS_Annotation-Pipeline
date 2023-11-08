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
        uncertainties = (probs*log_probs).sum(1)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]


class EntropySamplingRNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(EntropySamplingRNS, self).__init__(dataset, net, args_input, args_task)

    def query(self, n, index = None):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs)
        uncertainties = (probs * log_probs).sum(1)

        metric_array = self.episode_arrange(uncertainties, index, unlabeled_idxs)

        return unlabeled_idxs[torch.sort(metric_array)[1][:n]]

    def episode_arrange(self, metric, index, unlabeled_idxs):
        index_train_stack = np.hstack(index[unlabeled_idxs])
        assert metric.size()[0] == index_train_stack.shape[0]

        metric_array = torch.empty(len(unlabeled_idxs))

        for i, ind in enumerate(unlabeled_idxs):
            required_slice_ind = np.unique(index[ind]['episode_index'])[0]
            index_location = np.where(index_train_stack['episode_index'] == required_slice_ind)[0]
            metric_array[i] = torch.mean(metric[index_location])

        return metric_array
