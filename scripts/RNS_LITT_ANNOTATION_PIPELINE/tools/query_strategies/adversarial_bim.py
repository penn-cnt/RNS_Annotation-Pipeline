import numpy as np
import torch
import torch.nn as nn
loss = nn.CrossEntropyLoss()
import torch.nn.functional as F
from .strategy import Strategy
from tqdm import tqdm

class AdversarialBIM(Strategy):
    def __init__(self, dataset, net, args_input, args_task, eps=0.05):
        super(AdversarialBIM, self).__init__(dataset, net, args_input, args_task)
        self.eps = eps
        self.max_iter = 200

    def cal_dis(self, x):
        nx = torch.unsqueeze(x, 0).cuda()
        nx.requires_grad_()

        eta = torch.zeros(nx.shape).cuda()

        e1, out = self.net.net(nx+eta)
        py = out.max(1)[1]
        ny = out.max(1)[1]

        i_iter = 0
        while py.item() == ny.item() and i_iter < self.max_iter:
            loss = F.cross_entropy(out, ny).cuda()
            loss.backward()

            eta += self.eps * torch.sign(nx.grad.data)
            nx.grad.data.zero_()

            e1, out = self.net.net(nx+eta)
            py = out.max(1)[1]
            i_iter += 1

        return (eta*eta).sum()

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        self.net.net.cuda()
        self.net.net.eval()
        self.net.net.freeze_backbone = False
        dis = np.zeros(unlabeled_idxs.shape)

        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = unlabeled_data[i]
            dis[i] = self.cal_dis(x)

        return unlabeled_idxs[dis.argsort()[:n]]


