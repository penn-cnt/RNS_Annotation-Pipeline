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
class Backbone(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.feature0 = self.backbone[:4]
        self.feature1 = self.backbone[4]
        self.feature2 = self.backbone[5]
        self.feature3 = self.backbone[6]
        self.feature4 = self.backbone[7]
        self.feature5 = self.backbone[8:]

    def forward(self, x):
        x0 = self.feature0(x)
        x1 = self.feature1(x0)
        x2 = self.feature2(x1)
        x3 = self.feature3(x2)
        x4 = self.feature4(x3)
        emb = self.feature5(x4)

        return emb, [x1, x2, x3, x4]

class Classifier(pl.LightningModule):
    def __init__(self, input_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 64)
        self.dp = nn.Dropout1d(p=0.2)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = 0
        self.gamma = 5
        self.lstm = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)
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
        x = F.relu(self.fc3(x))
        pred = self.fc4(x)
        pred = self.softmax(pred)

        return pred, emb, emb_t


class LossNet(pl.LightningModule):
    def __init__(self, feature_sizes=[64, 32, 16, 8], num_channels=[256, 512, 1024, 2048], interm_dim=128):
        super().__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)
        self.FCt = nn.Linear(512, interm_dim)

        self.linear = nn.Linear(5 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out5 = features[4]
        out5 = F.relu(self.FCt(out5))

        out = self.linear(torch.cat((out1, out2, out3, out4, out5), 1))
        return out


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class LPL(pl.LightningModule):
    def __init__(self, net_fea, net_clf, net_lpl, weight=1.0, margin = 1):
        super().__init__()
        self.net_fea = net_fea
        self.net_clf = net_clf
        self.net_lpl = net_lpl
        self.automatic_optimization = False
        self.loss_alpha = 0
        self.loss_gamma = 5
        self.margin = margin
        self.weight = weight

    def training_step(self, batch, batch_idx):
        x, y, seq_len = batch

        opt_fea, opt_clf, opt_lpl = self.optimizers()

        opt_fea._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_fea._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        opt_lpl._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_lpl._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")


        # training feature extractor and predictor
        self.set_requires_grad(self.net_clf, True)
        self.set_requires_grad(self.net_fea, False)
        self.set_requires_grad(self.net_lpl, True)

        lb_z, feature = self.net_fea(x)
        lb_z = lb_z.view(-1, 2048)

        opt_fea.zero_grad()
        opt_clf.zero_grad()
        opt_lpl.zero_grad()

        lb_out, _, emb_t = self.net_clf(lb_z, seq_len)

        # prediction loss (based on provided classifier)
        # pred_loss = F.cross_entropy(lb_out, label_y)

        label = F.one_hot(y, num_classes=2).squeeze()
        target_loss = sigmoid_focal_loss(lb_out.float(), label.float(), alpha=self.loss_alpha, gamma=self.loss_gamma,
                                       reduction='none').mean(1)
        feature.append(emb_t)

        pred_loss = self.net_lpl(feature)
        pred_loss = pred_loss.view(pred_loss.size(0))

        backbone_loss = torch.sum(target_loss) / target_loss.size(0)

        if len(pred_loss) % 2 != 0:
            pred_loss = pred_loss[:-1]
            target_loss = target_loss[:-1]

        module_loss = LossPredLoss(pred_loss, target_loss, self.margin)
        loss = backbone_loss + self.weight * module_loss

        self.manual_backward(loss)

        # opt_fea.step()
        opt_clf.step()
        opt_lpl.step()

        self.log("train_loss", backbone_loss, prog_bar=True)
        self.log("dis_loss", module_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y, seq_len = batch

        z, _ = self.net_fea(x)

        z = z.view(-1, 2048)
        # Prediction
        pred, _, _ = self.net_clf(z, seq_len)

        label = F.one_hot(y, num_classes=2).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.loss_alpha, gamma=self.loss_gamma,
                                  reduction='mean')
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
        z, _ = self.net_fea(x)
        z  = z.view(-1,2048)

        # Prediction
        pred, emb, emb_t = self.net_clf(z, seq_len)

        # Return the predicted probabilities
        return pred, y, emb, emb_t, seq_len

    def configure_optimizers(self):
        # Define optimizers for feature extractor and classifier
        optimizer_fea = optim.Adam(self.net_fea.parameters(), lr=1e-3)
        optimizer_clf = optim.Adam(self.net_clf.parameters(), lr=1e-3)
        optimizer_lpl = optim.Adam(self.net_lpl.parameters(), lr=1e-2)

        # Return a list of optimizers and schedulers
        return optimizer_fea, optimizer_clf, optimizer_lpl

    def single_worst(self, probas):

        """
        The single worst will return the max_{k} -log(proba[k]) for each sample
        :param probas:
        :return:  # unlabeled \times 1 (tensor float)
        """

        value, _ = torch.max(-1 * torch.log(probas), 1)

        return value

    # setting gradient values
    def set_requires_grad(self, model, requires_grad=True, exclude=None):
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
                        param.requires_grad = not requires_grad


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], batch size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already haved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss
