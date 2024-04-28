import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm
from copy import deepcopy


class KCenterGreedyPCA(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(KCenterGreedyPCA, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        labeled_idxs, train_data = self.dataset.get_train_data()
        embeddings = self.get_embeddings(train_data)
        embeddings = embeddings.numpy()

        # downsampling embeddings if feature dim > 50
        if len(embeddings[0]) > 50:
            pca = PCA(n_components=50)
            embeddings = pca.fit_transform(embeddings)
        embeddings = embeddings.astype(np.float16)

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)

        return np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]


class KCenterGreedyPCARNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(KCenterGreedyPCARNS, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        labeled_idxs, train_data = self.dataset.get_train_data_unaugmented()
        embeddings, embeddings_t, seq_len = self.get_embeddings(train_data)
        embeddings = embeddings_t.numpy()
        labeled_idxs_copy = deepcopy(labeled_idxs)

        # downsampling embeddings if feature dim > 50
        if len(embeddings[0]) > 50:
            pca = PCA(n_components=50)
            embeddings = pca.fit_transform(embeddings)
        embeddings = embeddings.astype(np.float16)

        norm_data = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dist_mat = np.dot(norm_data, norm_data.T)
        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)

        output = np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]

        norm_data = embeddings_t / np.linalg.norm(embeddings_t, axis=1, keepdims=True)
        norm_data_core = embeddings_t[output] / np.linalg.norm(embeddings_t[output], axis=1, keepdims=True)
        similarity_matrix = np.dot(norm_data, norm_data_core.T)
        dis = np.min(similarity_matrix, 1)
        dis[output] = 1
        uncertainties = dis
        to_select = self.metrics_distribution_rescaling(uncertainties, seq_len, labeled_idxs_copy, n, descending=True)
        unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
        print('selected', np.sum(to_select))
        assert len(to_select) == len(unlabeled_idxs)

        return unlabeled_idxs[to_select.astype(bool)]
