import numpy as np
from .strategy import Strategy
import torch

class RandomSampling(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(RandomSampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n, index = None):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs, seq_len = self.predict_prob(unlabeled_data)

        to_select = self.dataset.episode_to_window_translation(np.random.choice(len(seq_len), n), seq_len)

        return unlabeled_idxs[to_select]


class RandomSamplingRNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(RandomSamplingRNS, self).__init__(dataset, net, args_input, args_task)

    def query(self, n, avg_len = 30):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        random_index = np.random.choice(len(unlabeled_idxs), int(n/avg_len))

        index_list = []
        for ri in random_index:
            index_list.append(np.arange(ri, ri + avg_len))
        index_list = np.array(index_list).flatten()

        return unlabeled_idxs[index_list]