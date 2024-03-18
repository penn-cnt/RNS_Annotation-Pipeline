import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(RandomSampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n, index = None):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs, seq_len = self.predict_prob(unlabeled_data)

        to_select = self.dataset.episode_to_window_translation(np.random.choice(len(seq_len), n), seq_len)

        # return np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)
        return unlabeled_idxs[to_select]
