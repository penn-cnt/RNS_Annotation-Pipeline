import numpy as np
from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(MarginSampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]

class MarginSamplingRNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(MarginSamplingRNS, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_train_data_unaugmented()
        probs, seq_len = self.predict_prob(unlabeled_data)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
        to_select = self.metrics_distribution_rescaling(uncertainties, seq_len, unlabeled_idxs, n)
        unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
        print('selected', np.sum(to_select))
        return unlabeled_idxs[to_select.astype(bool)]
