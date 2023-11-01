import numpy as np
from .strategy import Strategy
from fast_pytorch_kmeans import KMeans


class KMeansSamplingGPU(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(KMeansSamplingGPU, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        embeddings = self.get_embeddings(unlabeled_data)

        cluster_learner = KMeans(n_clusters=n,init_method ='++')
        cluster_learner.fit(embeddings)

        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers) ** 2
        dis = dis.sum(axis=1)
        q_idxs = np.array(
            [np.arange(embeddings.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n)])

        return unlabeled_idxs[q_idxs]


class FaissKmeans:
    def __init__(self, n_clusters=8, gpu=True, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.gpu = gpu

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   gpu=self.gpu)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        D, I = self.kmeans.index.search(X.astype(np.float32), 1)
        return D, I
