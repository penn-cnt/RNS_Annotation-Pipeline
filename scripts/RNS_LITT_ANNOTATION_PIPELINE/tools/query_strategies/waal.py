import numpy as np
from .strategy import Strategy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

'''
This implementation is with reference of https://github.com/cjshui/WAAL.
Please cite the original paper if you plan to use this method.
@inproceedings{shui2020deep,
  title={Deep active learning: Unified and principled method for query and training},
  author={Shui, Changjian and Zhou, Fan and Gagn{\'e}, Christian and Wang, Boyu},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1308--1318},
  year={2020},
  organization={PMLR}
}
'''


class WAAL(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(WAAL, self).__init__(dataset, net, args_input, args_task)
        self.selection = 10

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        uncertainly_score = 0.5 * self.L2_upper(probs) + 0.5 * self.L1_upper(probs)

        # prediction output discriminative score
        dis_score = self.pred_dis_score_waal(unlabeled_data)

        # computing the decision score
        total_score = uncertainly_score - self.selection * dis_score
        b = total_score.sort()[1][:n]

        return unlabeled_idxs[total_score.sort()[1][:n]]

    def L2_upper(self, probas):
        value = torch.norm(torch.log(probas), dim=1)
        return value

    def L1_upper(self, probas):
        value = torch.sum(-1 * torch.log(probas), dim=1)
        return value

    def pred_dis_score_waal(self, data):
        loader_te = DataLoader(data, shuffle=False, **self.args_task['loader_te_args'])

        self.net.fea.eval()
        self.net.dis.eval()

        scores = torch.zeros(len(data))

        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.cuda(), y.cuda()
                latent = self.net.fea(x)
                out = self.net.dis(latent).cpu()
                scores[idxs] = out.view(-1)

        return scores


class WAALRNS(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(WAALRNS, self).__init__(dataset, net, args_input, args_task)
        self.selection = 5

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_train_data_unaugmented()
        probs, seq_len = self.predict_prob(unlabeled_data)
        uncertainly_score = 0.5 * self.L2_upper(probs) + 0.5 * self.L1_upper(probs)

        # prediction output discriminative score
        dis_score, _ = self.pred_dis_score_waal(unlabeled_data)

        # computing the decision score
        uncertainties = uncertainly_score - self.selection * dis_score

        to_select = self.metrics_distribution_rescaling(uncertainties, seq_len, unlabeled_idxs, n)
        unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
        print('selected', np.sum(to_select))

        assert len(to_select) == len(unlabeled_idxs)

        return unlabeled_idxs[to_select.astype(bool)]

    def L2_upper(self, probas):
        value = torch.norm(torch.log(probas), dim=1)
        return value

    def L1_upper(self, probas):
        value = torch.sum(-1 * torch.log(probas), dim=1)
        return value

    def pred_dis_score_waal(self, data):
        loader_te = DataLoader(data, shuffle=False, **self.args_task['loader_te_args'])

        self.net.net.net_fea.eval()
        self.net.net.net_dis.eval()
        self.net.net.net_fea.cuda()
        self.net.net.net_dis.cuda()

        scores = []
        seq_len_list = []
        with torch.no_grad():
            for x, y, seq_len in tqdm(loader_te):
                x, y = x.cuda(), y.cuda()
                latent = self.net.net.net_fea(x).view(-1, 2048)
                out = self.net.net.net_dis(latent).cpu()
                scores.append(out.view(-1))
                seq_len_list.append(seq_len)

        return torch.tensor([item for sublist in scores for item in sublist]), \
            torch.tensor([item for sublist in seq_len_list for item in sublist])
