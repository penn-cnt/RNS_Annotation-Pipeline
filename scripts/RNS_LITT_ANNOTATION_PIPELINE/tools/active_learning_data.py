import numpy as np
import torch
import random
import os
import sklearn

np.random.seed(42)


class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, seq_len_train, seq_len_test, handler, args_task):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.seq_len_train = seq_len_train
        self.seq_len_test = seq_len_test
        self.handler = handler
        self.args_task = args_task

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.n_episode_train = len(self.seq_len_train)
        self.n_episode_test = len(self.seq_len_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_episode_train)
        np.random.shuffle(tmp_idxs)
        to_choose = self.episode_to_window_translation(tmp_idxs[:num], self.seq_len_train)
        self.labeled_idxs[to_choose] = True

    def get_unlabeled_data_by_idx(self, idx):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs][idx]

    def get_data_by_idx(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def get_new_data(self, X, Y):
        return self.handler(X, Y, self.args_task['transform_train'])

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs].tolist()

        return labeled_idxs, self.handler(
            self.combine_window_to_episode(self.X_train, self.seq_len_train, self.labeled_idxs),
            self.combine_window_to_episode(self.Y_train, self.seq_len_train, self.labeled_idxs),
            self.args_task['transform_train'])

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(
            self.combine_window_to_episode(self.X_train, self.seq_len_train, ~self.labeled_idxs),
            self.combine_window_to_episode(self.Y_train, self.seq_len_train, ~self.labeled_idxs),
            self.args_task['transform'])


    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, self.args_task['transform_train'])

    def get_train_data_unaugmented(self):
        return self.labeled_idxs.copy(), self.handler(
            self.combine_window_to_episode(self.X_train, self.seq_len_train),
            self.combine_window_to_episode(self.Y_train, self.seq_len_train), self.args_task['transform'])

    def get_test_data(self):
        return self.handler(self.combine_window_to_episode(self.X_test, self.seq_len_test),
                            self.combine_window_to_episode(self.Y_test, self.seq_len_test), self.args_task['transform'])

    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return self.X_train[labeled_idxs], self.Y_train[labeled_idxs]

    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]

    def cal_test_acc(self, preds):
        return sklearn.metrics.accuracy_score(preds, self.Y_test)

    def episode_to_window_translation(self, episode_index, seq_len):
        cum_sum_index = np.cumsum(seq_len)
        if isinstance(episode_index, np.ndarray):
            window_index = []
            for epi_ind in episode_index:
                if epi_ind > 0:
                    start_index = cum_sum_index[epi_ind - 1]
                    end_index = cum_sum_index[epi_ind]
                else:
                    start_index = 0
                    end_index = cum_sum_index[epi_ind]

                window_index.append(np.arange(start_index, end_index))
            return np.concatenate(window_index)
        else:
            epi_ind = episode_index
            if epi_ind > 0:
                start_index = cum_sum_index[epi_ind - 1]
                end_index = cum_sum_index[epi_ind]
            else:
                start_index = 0
                end_index = cum_sum_index[epi_ind]

            return np.arange(start_index, end_index)

    # def combine_window_to_episode(self, data, seq_len, index=None):
    #     data_out = []
    #
    #     cum_sum_index = np.cumsum(seq_len)
    #
    #     cum_sum_index = np.insert(cum_sum_index, 0, 0)
    #     assert len(data) == cum_sum_index[-1]
    #     for i in range(1, len(cum_sum_index)):
    #         start_index = cum_sum_index[i - 1]
    #         end_index = cum_sum_index[i]
    #         episode_data = data[start_index:end_index]
    #         if index is None:
    #             out = episode_data
    #         else:
    #             episode_labeled = index[start_index:end_index]
    #             out = episode_data[episode_labeled == True]
    #
    #         if len(out) > 0:
    #             data_out.append(out)
    #
    #     return np.array(data_out, dtype=object)

    def combine_window_to_episode(self, data, seq_len, index=None):
        cum_sum_index = np.cumsum(seq_len)
        cum_sum_index = np.insert(cum_sum_index, 0, 0)

        assert len(data) == cum_sum_index[-1]

        data_out = [None] * (len(cum_sum_index) - 1)

        for i in range(1, len(cum_sum_index)):
            start_index, end_index = cum_sum_index[i - 1], cum_sum_index[i]
            episode_data = data[start_index:end_index]

            if index is None:
                out = episode_data
            else:
                episode_labeled = index[start_index:end_index]
                out = episode_data[episode_labeled]

            if len(out) > 0:
                data_out[i - 1] = out

        data_out = [segment for segment in data_out if segment is not None]

        return np.array(data_out, dtype=object)

    def get_slice_from_episode(self, data, seq_len, index):
        probs_out = []
        probs_sliced = self.combine_window_to_episode(data, seq_len)
        for i, unlabeled in enumerate(self.combine_window_to_episode(index, seq_len)):
            cleaned_probs = probs_sliced[i][unlabeled]
            if len(cleaned_probs) > 0:
                probs_out.append(cleaned_probs)

        return np.array(probs_out, dtype=object), torch.tensor([len(pb) for pb in probs_out])
