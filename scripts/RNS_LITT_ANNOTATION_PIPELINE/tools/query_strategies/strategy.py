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

    def train(self, data=None, model_name=None, initialize_only = False):

        if model_name == None:
            if data == None:
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()

                self.net.train(labeled_data, self.dataset.get_test_data(), initialize_only= initialize_only)
            else:
                self.net.train(data,initialize_only= initialize_only)
        else:
            if model_name == 'WAAL':
                labeled_idxs, labeled_data = self.dataset.get_labeled_data()
                X_labeled, Y_labeled = self.dataset.get_partial_labeled_data()
                X_unlabeled, Y_unlabeled = self.dataset.get_partial_unlabeled_data()
                # print(labeled_idxs, labeled_data)
                # print(X_labeled.shape, Y_labeled.shape)
                # print(X_unlabeled.shape, Y_unlabeled.shape)
                self.net.train(labeled_data, X_labeled, Y_labeled, X_unlabeled, Y_unlabeled,
                               test_data=self.dataset.get_test_data())
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

    def metrics_distribution_rescaling(self, uncertainties, seq_len, unlabeled_idxs, n, percentile = 0.2,
                                       descending=False):
        if descending:
            uncertainties = -uncertainties

        indices = np.argsort(uncertainties)
        # original_order = indices.argsort()
        normalized_data = self.normalize(uncertainties)
        scaling = normalized_data[indices][int(percentile*len(unlabeled_idxs))]
        uncertainties_metric = scaling - normalized_data
        uncertainties_metric, seq_len = self.dataset.get_slice_from_episode(uncertainties_metric, seq_len,
                                                                            ~unlabeled_idxs)
        uncertainties_metric = np.concatenate(uncertainties_metric)
        metrics = self.dataset.combine_window_to_episode(uncertainties_metric, seq_len)
        to_select = self.get_combined_important(torch.flatten(seq_len), metrics, n)

        return to_select

    def get_importance_output_plotting(self, uncertainties, seq_len, unlabeled_idxs, percentile = 0.2,
                                       descending=False):
        if descending:
            uncertainties = -uncertainties

        indices = np.argsort(uncertainties)
        # original_order = indices.argsort()
        normalized_data = self.normalize(uncertainties)
        scaling = normalized_data[indices][int(percentile*len(unlabeled_idxs))]
        uncertainties_metric = scaling - normalized_data
        uncertainties_metric, seq_len = self.dataset.get_slice_from_episode(uncertainties_metric, seq_len,
                                                                            ~unlabeled_idxs)
        uncertainties_metric = np.concatenate(uncertainties_metric)
        metrics = self.dataset.combine_window_to_episode(uncertainties_metric, seq_len)

        return metrics

    def normalize(self, x):
        return (x - min(x)) / (max(x) - min(x))

    def smoothing_prediction(self, data, window_size):
        weights = np.ones(window_size) / window_size
        smoothed_data = np.convolve(data, weights, mode='same')
        return smoothed_data

    def keep_continuous_segments(self, arr, n):
        changed_ind = np.where(np.sign(np.diff(arr) - 1) == 1)[0]
        valid_split = np.where(np.diff(changed_ind) >= n)[0]
        start_ind = changed_ind[valid_split] + 1
        end_ind = changed_ind[valid_split + 1] + 1
        cleaned_arr = np.empty(0, dtype=int)
        for i in range(len(valid_split)):
            cleaned_arr = np.hstack((cleaned_arr, arr[start_ind[i]:end_ind[i]]))
        return cleaned_arr
