import torch.nn.functional as F
from torch import nn
import torch
import sklearn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
import sys

sys.path.append('../tools')
from sigmoid_loss import sigmoid_focal_loss


class SupervisedDownstream(pl.LightningModule):
    def __init__(self, backbone, input_dim = 512, unfreeze_backbone_at_epoch = -1):
        super().__init__()
        self.input_dim = input_dim
        self.backbone = backbone
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 3)
        self.dropout = nn.Dropout(p = 0.4)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = 0.5
        self.gamma = 8
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.enable_mc_dropout = False
        self.automatic_optimization = True
        self.freeze_backbone = True
        # self.fl = FocalLoss(self.alpha, self.gamma).cuda()

    def forward(self, x):
        if self.freeze_backbone == True:
            self.backbone.eval()
            x = self.backbone(x)
            with torch.no_grad():
                emb = x.view(-1, self.input_dim)
        else:
            x = self.backbone(x)
            emb = x.view(-1, self.input_dim)


        x = self.dropout(emb)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        pred = self.fc4(x)
        pred = self.softmax(pred)
        return emb, pred

    def training_step(self, batch, batch_idx):
        if self.unfreeze_backbone_at_epoch == -1 or self.current_epoch < self.unfreeze_backbone_at_epoch:
            self.freeze_backbone = True
        else:
            self.freeze_backbone = False
        x, y = batch
        _, pred = self(x)
        label = F.one_hot(y, num_classes=3).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')
        # loss = self.fl(torch.FloatTensor(pred).cuda(), torch.FloatTensor(label).cuda())
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, pred = self(x)
        label = F.one_hot(y, num_classes=3).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')
        # loss = self.fl(torch.FloatTensor(pred).cuda(), torch.FloatTensor(label).cuda())
        out = torch.argmax(pred, dim=1)
        out = out.detach().cpu().numpy()
        target = y.squeeze().detach().cpu().numpy()
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(out, target,
                                                                                             zero_division=0.0)
        acc = sklearn.metrics.accuracy_score(out, target)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", precision[0])
        self.log("val_recall", recall[0])
        return pred, label

    def predict_step(self, batch, batch_idx):
        x, y = batch
        if self.enable_mc_dropout:
            self.dropout.train()

        emb, pred = self(x)
        # Logging to TensorBoard (if installed) by default
        return pred, y, emb

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def cal_dis(self, x):
        self.eval()
        eps = 0.05
        nx = torch.unsqueeze(x, 0).cuda()
        nx.requires_grad_()
        nx.grad
        eta = torch.zeros(nx.shape).cuda()

        _, out = self(nx+eta)
        py = out.max(1)[1]
        ny = out.max(1)[1]

        # while py.item() == ny.item():
        celoss = nn.CrossEntropyLoss()
        loss = celoss(out, ny)
        self.manual_backward(loss)
        eta += eps * torch.sign(nx.grad.data)
        nx.grad.data.zero_()

        e1, out = self.net.net(nx+eta)
        py = out.max(1)[1]

        return (eta*eta).sum()
