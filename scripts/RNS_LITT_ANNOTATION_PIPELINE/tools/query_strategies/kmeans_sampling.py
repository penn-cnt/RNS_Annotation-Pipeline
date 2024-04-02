import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
from fast_pytorch_kmeans import KMeans as FKMeans
import torch

class KMeansSampling(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(KMeansSampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embeddings)
        
        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers)**2
        dis = dis.sum(axis=1)
        q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

        return unlabeled_idxs[q_idxs]


class KMeansSamplingRNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(KMeansSamplingRNS, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_train_data_unaugmented()
        embeddings, embeddings_t, seq_len = self.get_embeddings(unlabeled_data)

        cluster_learner = FKMeans(n_clusters=100, mode='cosine', verbose=1)
        cluster_learner.fit(embeddings)

        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.centroids[cluster_idxs]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        dis = cos(embeddings, centers)
        uncertainties = dis
        to_select = self.metrics_distribution_rescaling(uncertainties, seq_len, unlabeled_idxs, n)
        unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
        print('selected', np.sum(to_select))
        assert len(to_select) == len(unlabeled_idxs)

        return unlabeled_idxs[to_select.astype(bool)]

