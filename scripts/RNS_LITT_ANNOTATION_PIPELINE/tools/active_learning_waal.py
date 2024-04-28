import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.init as init
from sigmoid_loss import sigmoid_focal_loss
import sklearn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, unpad_sequence

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks

# log_folder_root = 'E:/Download/RNS_Annotation-Pipeline-main/RNS_Annotation-Pipeline-main/user_data/logs/kaggle_dog_active'
# ckpt_folder_root = 'E:/Download/RNS_Annotation-Pipeline-main/RNS_Annotation-Pipeline-main/user_data/checkpoints/kaggle_dog_active'


class Classifier(pl.LightningModule):
    def __init__(self, input_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.dp = nn.Dropout1d(p=0.2)
        self.fc3 = nn.Linear(64, 2)
        # self.fc4 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = 0
        self.gamma = 5
        self.lstm = nn.LSTM(256, 128, 1, batch_first=True, bidirectional=True)
        self.enable_mc_dropout = False
        self.input_dim = input_dim

    def forward(self, x, seq_len):
        emb = x

        if self.enable_mc_dropout:
            self.dp.train()
        else:
            self.dp.eval()

        x = F.relu(self.fc1(emb))

        x = self.dp(x)
        x = torch.split(x, seq_len, dim=0)
        x = pack_sequence(x, enforce_sorted=False)
        x, (_, _) = self.lstm(x)
        x, out_len = pad_packed_sequence(x, batch_first=True)
        emb_t = torch.concat(unpad_sequence(x, out_len, batch_first=True))

        x = self.dp(emb_t)

        x = F.relu(self.fc2(x))
        pred = self.fc3(x)
        pred = self.softmax(pred)

        return pred, emb, emb_t


class Discriminator(pl.LightningModule):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = np.prod(dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z).reshape(-1)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class WAAL(pl.LightningModule):
    def __init__(self, net_fea, net_clf, net_dis, gamma_ratio=1, alpha=2e-3):
        super().__init__()
        self.net_fea = net_fea
        self.net_clf = net_clf
        self.net_dis = net_dis
        self.gamma_ratio = gamma_ratio
        self.alpha = alpha
        self.automatic_optimization = False
        self.loss_alpha = 0
        self.loss_gamma = 5

    def training_step(self, batch, batch_idx):
        label_x, label_y, seq_len_label, unlabel_x, _, seq_len_unlabel = batch
        max_len = min(len(seq_len_label), len(seq_len_unlabel))

        opt_fea, opt_clf, opt_dis = self.optimizers()

        opt_fea._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_fea._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        opt_dis._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_dis._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")

        # label_x, label_y = label_x.to(self.device), label_y.to(self.device)
        # unlabel_x = unlabel_x.to(self.device)

        # training feature extractor and predictor
        self.set_requires_grad(self.net_clf, True)
        self.set_requires_grad(self.net_fea, True, exclude=['0','1','2','3','4','5','6'])
        # self.set_requires_grad(self.net_fea, False)
        self.set_requires_grad(self.net_dis, False)

        lb_z = self.net_fea(label_x).view(-1, 2048)
        unlb_z = self.net_fea(unlabel_x).view(-1, 2048)

        opt_fea.zero_grad()
        opt_clf.zero_grad()

        lb_out, _, _ = self.net_clf(lb_z,seq_len_label)

        # prediction loss (based on provided classifier)
        # pred_loss = F.cross_entropy(lb_out, label_y)

        label = F.one_hot(label_y, num_classes=2).squeeze()
        pred_loss = sigmoid_focal_loss(lb_out.float(), label.float(), alpha=self.loss_alpha, gamma=self.loss_gamma,
                                       reduction='mean')

        # Wasserstein loss (unbalanced loss, used the redundant trick)
        wassertein_distance = self.net_dis(unlb_z).mean() - self.gamma_ratio * self.net_dis(lb_z).mean()

        with torch.no_grad():
            lb_z = self.net_fea(label_x).view(-1, 2048)
            unlb_z = self.net_fea(unlabel_x).view(-1, 2048)

        gp = self.gradient_penalty(self.net_dis, unlb_z[:max_len], lb_z[:max_len])

        loss = pred_loss + self.alpha * wassertein_distance + self.alpha * gp * 2

        self.manual_backward(loss)
        opt_clf.step()
        # opt_fea.step()

        self.set_requires_grad(self.net_fea, False)
        self.set_requires_grad(self.net_clf, False)
        self.set_requires_grad(self.net_dis, True)

        with torch.no_grad():
            lb_z = self.net_fea(label_x).view(-1, 2048)
            unlb_z = self.net_fea(unlabel_x).view(-1, 2048)

        for _ in range(10):
            # gradient ascent for multiple times like GANS training

            gp = self.gradient_penalty(self.net_dis, unlb_z[:max_len], lb_z[:max_len])

            wassertein_distance = self.net_dis(unlb_z[:max_len]).mean() - self.gamma_ratio * self.net_dis(unlb_z[:max_len]).mean()

            dis_loss = -1 * self.alpha * wassertein_distance - self.alpha * gp * 2

            opt_dis.zero_grad()
            self.manual_backward(dis_loss)
            opt_dis.step()

        self.log("train_loss", loss, prog_bar=True)
        self.log("dis_loss", dis_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y, seq_len = batch

        # Feature extraction
        z = self.net_fea(x).view(-1, 2048)

        # Prediction
        pred, _, _ = self.net_clf(z, seq_len)

        label = F.one_hot(y, num_classes=2).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.loss_alpha, gamma=self.loss_gamma, reduction='mean')
        out = torch.argmax(pred, dim=1)

        out = out.detach().cpu().numpy()
        target = y.squeeze().detach().cpu().numpy()
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(out, target, labels=[0, 1],
                                                                                             zero_division=0)
        acc = sklearn.metrics.accuracy_score(out, target)
        # print(acc)
        # print(precision)
        # print(recall)
        # print(fscore)

        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=False)
        self.log("val_precision", precision[1], prog_bar=False)
        self.log("val_recall", recall[1], prog_bar=False)
        self.log("val_f1", fscore[1], prog_bar=False)
        return pred, label

    def predict_step(self, batch, batch_idx):
        x, y, seq_len = batch

        # Feature extraction
        z = self.net_fea(x).view(-1, 2048)

        # Prediction
        pred, emb, emb_t = self.net_clf(z, seq_len)

        # Return the predicted probabilities
        return pred, y, emb, emb_t, seq_len

    def configure_optimizers(self):
        # Define optimizers for feature extractor and classifier
        optimizer_fea = optim.Adam(self.net_fea.parameters(), lr=1e-3)
        optimizer_clf = optim.Adam(self.net_clf.parameters(), lr=1e-3)
        optimizer_dis = optim.SGD(self.net_dis.parameters(), lr=1e-3, momentum=0.3)

        # Return a list of optimizers and schedulers
        return optimizer_fea, optimizer_clf, optimizer_dis

    def single_worst(self, probas):

        """
        The single worst will return the max_{k} -log(proba[k]) for each sample
        :param probas:
        :return:  # unlabeled \times 1 (tensor float)
        """

        value, _ = torch.max(-1 * torch.log(probas), 1)

        return value

    # setting gradient values
    def set_requires_grad(self, model, requires_grad=True, exclude = None):
        """
        Used in training adversarial approach
        :param model:
        :param requires_grad:
        :return:
        """
        for param in model.parameters():
            param.requires_grad = requires_grad

        if exclude is not None:
            for name, child in model.named_children():
                if name in exclude:
                    for param in child.parameters():
                        param.requires_grad =not requires_grad

    # setting gradient penalty for sure the lipschitiz property
    def gradient_penalty(self, critic, h_s, h_t):
        ''' Gradeitnt penalty approach'''
        alpha = torch.rand(h_s.size(0), 1).to(self.device)
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
        # interpolates.requires_grad_()
        preds = critic(interpolates)
        gradients = grad(preds, interpolates,
                         grad_outputs=torch.ones_like(preds),
                         retain_graph=True, create_graph=True)[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty


class Net_WAAL:
    def __init__(self, net, params, device, handler_joint, log_folder_root='kaggle_dog_active',
                 ckpt_folder_root='kaggle_dog_active'):
        self.net = net
        self.params = params
        self.device = device
        self.round = 0
        self.trainer = None
        self.handler_joint = handler_joint
        self.log_folder_root = '../../../user_data/logs/' + log_folder_root + '/'
        self.ckpt_folder_root = '../../../user_data/checkpoints/' + ckpt_folder_root + '/'

    def train(self, data, X_labeled, Y_labeled, X_unlabeled, Y_unlabeled, alpha=1e-3, test_data=None):
        # Define checkpoints and logger
        ckpt_save_n_step = 40

        checkpoint_callback = pl_callbacks.ModelCheckpoint(monitor='train_loss',
                                                           filename=
                                                           self.params['strategy_name'] + '_round_' + str(self.round)
                                                           + '-{step}-{train_loss:.5f}',
                                                           dirpath=self.ckpt_folder_root + 'active_checkpoints_'
                                                                   + self.params['strategy_name'],
                                                           save_top_k=-1,
                                                           every_n_train_steps=ckpt_save_n_step,
                                                           save_on_train_epoch_end=False)

        early_stop_callback = pl_callbacks.EarlyStopping(monitor="val_acc",
                                                         patience=10,
                                                         verbose=False,
                                                         mode="max")

        csv_logger = pl_loggers.CSVLogger(self.log_folder_root + "active_logs_" + self.params['strategy_name'],
                                          name='logger_round_' + str(self.round))

        trainer = pl.Trainer(logger=csv_logger,
                             max_epochs=self.params['n_epoch'],
                             callbacks=[checkpoint_callback, early_stop_callback],
                             # callbacks=[checkpoint_callback],
                             accelerator='gpu',
                             devices=1,
                             log_every_n_steps=50,
                             precision=16,
                             check_val_every_n_epoch=None,
                             val_check_interval=ckpt_save_n_step,
                             enable_model_summary=False,
                             )

        self.net.gamma_ratio = len(X_labeled)/ len(X_unlabeled)

        self.trainer = trainer
        self.net.to(self.device)

        if test_data is not None:
            train_loader = DataLoader(
                self.handler_joint(X_labeled, Y_labeled, X_unlabeled, Y_unlabeled,
                                   transform=self.params['transform_train']),
                shuffle=True,
                **self.params['loader_tr_args']
            )
            testloader = DataLoader(test_data, shuffle=False, **self.params['loader_te_args'])

            self.trainer.fit(self.net, train_loader, testloader)
        else:
            train_loader = DataLoader(
                self.handler_joint(X_labeled, Y_labeled, X_unlabeled, Y_unlabeled,
                                   transform=self.params['transform_train']),
                shuffle=True,
                **self.params['loader_tr_args']
            )

            self.trainer.fit(self.net, train_loader)

    def run_prediction(self, data):
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        predictions = self.trainer.predict(self.net, loader)
        return predictions

    # def predict(self, data):
    #     self.clf.eval()
    #     self.fea.eval()
    #     preds = torch.zeros(len(data), dtype=data.Y.dtype)
    #     loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
    #     with torch.no_grad():
    #         for x, y, idxs in loader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             latent = self.fea(x)
    #             out, _ = self.clf(latent)
    #             pred = out.max(1)[1]
    #             preds[idxs] = pred.cpu()
    #     return preds

    def predict_prob(self, data):

        predictions = self.run_prediction(data)
        output_list = []
        seq_len_list = []
        m = nn.Softmax(dim=1)
        for pred, y, emb, emb_t, seq_len in predictions:
            output_list.append(pred)
            seq_len_list.append(seq_len)

        pred_raw = torch.vstack(output_list).float()
        seq_len_out = torch.tensor([item for sublist in seq_len_list for item in sublist])
        probs = m(pred_raw)
        return probs, seq_len_out

    def get_embeddings(self, data, return_target=False):
        predictions = self.run_prediction(data)
        emb_list = []
        emb_t_list = []
        target_list = []
        seq_len_list = []
        for pred, y, emb, emb_t, seq_len in predictions:
            emb_list.append(emb)
            emb_t_list.append(emb_t)
            target_list.append(y)
            seq_len_list.append(seq_len)
        emb = torch.vstack(emb_list).float()
        emb_t = torch.vstack(emb_t_list).float()
        target = torch.concat(target_list)
        if return_target:
            return emb, emb_t, target, torch.tensor([item for sublist in seq_len_list for item in sublist])
        else:
            return emb, emb_t, torch.tensor([item for sublist in seq_len_list for item in sublist])
