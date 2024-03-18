import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def max_subarray_sum_with_indices_soft_margin(arr, margin):
    if not arr:  # handle empty array case
        return 0, -1, -1

    max_ending_here = max_so_far = arr[0]
    start = end = 0
    temp_start = 0
    # Introduce a margin count to keep track of how many times we've applied the soft margin
    margin_count = 0

    for i, x in enumerate(arr[1:], start=1):
        if x > max_ending_here + x:
            max_ending_here = x
            temp_start = i
            margin_count = 0  # Reset margin count for a new start
        else:
            # Apply soft margin: if the addition is negative but within the margin, continue the current subarray
            if max_ending_here + x - max_ending_here < margin:
                margin_count += 1
                max_ending_here += x
            else:
                max_ending_here += x

        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            start = temp_start
            end = i
            margin_count = 0  # Reset margin count after finding a new max

    return max_so_far, start, end


class Strategy:
    def __init__(self, dataset, net, args_input, args_task):
        self.dataset = dataset
        self.net = net
        self.args_input = args_input
        self.args_task = args_task

    def query(self, n):
        pass

    def get_labeled_count(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        return len(labeled_idxs)

    def get_model(self):
        return self.net.get_model()

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self, data=None, model_name=None):

        if model_name == None:
            if data == None:
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()

                self.net.train(labeled_data, self.dataset.get_test_data())
            else:
                self.net.train(data)
        else:
            if model_name == 'WAAL':
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
                X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()
                self.net.train(labeled_data, X_labeled, Y_labeled, X_unlabeled, Y_unlabeled)
            else:
                raise NotImplementedError

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs

    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

    def get_grad_embeddings(self, data):
        embeddings = self.net.get_grad_embeddings(data)
        return embeddings

    def get_combined_important(self, seq_len, metrics, n):
        arr = []
        for i in range(len(seq_len)):
            max_sum, start_index, end_index = max_subarray_sum_with_indices_soft_margin(metrics[i].tolist(), margin=50)
            arr.append([max_sum, start_index, end_index])
        arr = np.array(arr)

        tot = 0
        len_list = []
        for ind in np.argsort(-arr[:, 0]):
            tot += arr[ind][2] - arr[ind][1]
            len_list.append(tot)
        print('total_select_from', tot)

        starting_indices = np.roll(np.cumsum(seq_len), 1)
        starting_indices[0] = 0
        arr[:, 1] += starting_indices
        arr[:, 2] += starting_indices
        to_select_arr = np.zeros(torch.sum(seq_len))
        cleaned_arr = arr[(arr[:, 2] - arr[:, 1]) > 10]
        for i in np.argsort(-cleaned_arr[:, 0]):
            to_select_ind = np.arange(cleaned_arr[i, 1], cleaned_arr[i, 2], dtype=int)
            to_select_arr[to_select_ind] = 1
            if np.sum(to_select_arr) > n:
                break
        return to_select_arr



