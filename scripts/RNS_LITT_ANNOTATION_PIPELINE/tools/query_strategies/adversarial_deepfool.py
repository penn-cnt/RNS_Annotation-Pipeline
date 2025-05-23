import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy
from tqdm import tqdm

class AdversarialDeepFool(Strategy):
    def __init__(self, dataset, net, args_input, args_task, max_iter=50):
        super(AdversarialDeepFool, self).__init__(dataset, net, args_input, args_task)
        self.max_iter = max_iter

    def cal_dis(self, x):
        nx = torch.unsqueeze(x, 0).cuda()
        nx.requires_grad_()
        eta = torch.zeros(nx.shape).cuda()

        e1, out = self.net.net(nx+eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.cpu().numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.cpu().numpy().flatten()) * wi

            eta += ri.clone()
            nx.grad.data.zero_()
            e1, out = self.net.net(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1

        return (eta*eta).sum()

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # self.net.net.cpu()
        self.net.net.cuda()
        self.net.net.eval()
        dis = np.zeros(unlabeled_idxs.shape)
        self.net.net.freeze_backbone = False

        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = unlabeled_data[i]
            dis[i] = self.cal_dis(x)

        self.net.net.cuda()

        return unlabeled_idxs[dis.argsort()[:n]]


