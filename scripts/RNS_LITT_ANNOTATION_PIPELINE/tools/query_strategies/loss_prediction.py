import numpy as np
import torch
from .strategy import Strategy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


'''
This implementation is with reference of https://github.com/Mephisto405/Learning-Loss-for-Active-Learning.
Please cite the original paper if you use this method.
@inproceedings{yoo2019learning,
  title={Learning loss for active learning},
  author={Yoo, Donggeun and Kweon, In So},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={93--102},
  year={2019}
}
'''
class LossPredictionLoss(Strategy):
	def __init__(self, dataset, net, args_input, args_task):
		super(LossPredictionLoss, self).__init__(dataset, net, args_input, args_task)

	def query(self, n):
		unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
		uncertainties = self.unc_lpl(unlabeled_data)
		return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]

	def unc_lpl(self, data):
		loader = DataLoader(data, shuffle=False, **self.args_task['loader_te_args'])
		self.net.clf.eval()
		self.net.clf_lpl.eval()
		uncertainty = torch.tensor([]).cuda()
		with torch.no_grad():
			for x, y, idxs in loader:
				x, y = x.cuda(), y.cuda()
				out, feature = self.net.clf(x)
				pred_loss = self.net.clf_lpl(feature)
				pred_loss = pred_loss.view(pred_loss.size(0))
				uncertainty = torch.cat((uncertainty, pred_loss), 0)

		uncertainty = uncertainty.cpu()
		return uncertainty

class LossPredictionLossRNS(Strategy):
	def __init__(self, dataset, net, args_input, args_task):
		super(LossPredictionLossRNS, self).__init__(dataset, net, args_input, args_task)

	def query(self, n):
		unlabeled_idxs, unlabeled_data = self.dataset.get_train_data_unaugmented()
		uncertainties,seq_len = self.unc_lpl(unlabeled_data)
		to_select = self.metrics_distribution_rescaling(uncertainties, seq_len, unlabeled_idxs, n, descending=True)
		unlabeled_idxs, _ = self.dataset.get_unlabeled_data()
		print('selected', np.sum(to_select))

		assert len(to_select) == len(unlabeled_idxs)

		return unlabeled_idxs[to_select.astype(bool)]


	def unc_lpl(self, data):
		loader = DataLoader(data, shuffle=False, **self.args_task['loader_te_args'])
		self.net.net.net_clf.eval()
		self.net.net.net_fea.eval()
		self.net.net.net_lpl.eval()
		self.net.net.net_clf.cuda()
		self.net.net.net_fea.cuda()
		self.net.net.net_lpl.cuda()
		uncertainty = torch.tensor([]).cuda()
		seq_len_list = []
		with torch.no_grad():
			for x, y, seq_len in loader:
				x, y = x.cuda(), y.cuda()
				lb_z, feature = self.net.net.net_fea(x)
				lb_z = lb_z.view(-1, 2048)
				lb_out, _, emb_t = self.net.net.net_clf(lb_z, seq_len)
				feature.append(emb_t)
				pred_loss = self.net.net.net_lpl(feature)
				pred_loss = pred_loss.view(pred_loss.size(0))
				uncertainty = torch.cat((uncertainty, pred_loss), 0)
				seq_len_list.append(seq_len)

		uncertainty = uncertainty.cpu()
		return uncertainty, torch.tensor([item for sublist in seq_len_list for item in sublist])