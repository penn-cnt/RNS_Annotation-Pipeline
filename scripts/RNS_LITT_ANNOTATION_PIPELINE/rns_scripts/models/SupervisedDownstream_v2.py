import torch.nn.functional as F
from torch import nn
import torch
import sklearn
import pytorch_lightning as pl
import sys
sys.path.append('../tools')
from sigmoid_loss import sigmoid_focal_loss

class SupervisedDownstream(pl.LightningModule):
    def __init__(self, ft_enc, backbone, unfreeze_backbone_at_epoch=100):
        super().__init__()
        self.ft_enc = ft_enc
        self.backbone = backbone
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 64)
        self.dp = nn.Dropout1d(p=0.2)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = 0.8
        self.gamma = 5
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch

    def training_step(self, batch, batch_idx):
        x = batch[0].float()
        y = batch[1].float()

        with torch.no_grad():
            x = self.ft_enc(x)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
            x = self.backbone(x).flatten(start_dim=1)
            x = nn.functional.normalize(x, dim=1, p=2)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        pred = self.fc4(x)
        pred = self.softmax(pred)
        label = F.one_hot(y.to(torch.int64), num_classes=2).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0].float()
        y = batch[1].float()

        x = self.ft_enc(x)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        x = self.backbone(x).flatten(start_dim=1)
        x = nn.functional.normalize(x, dim=1, p=2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        pred = self.fc4(x)
        pred = self.softmax(pred)
        label = F.one_hot(y.to(torch.int64), num_classes=2).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')
        out = torch.argmax(pred, dim=1)
        # print(out.size)
        out = out.detach().cpu().numpy()
        target = y.squeeze().detach().cpu().numpy()
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(out, target,labels = [0,1],zero_division=0)
        acc = sklearn.metrics.accuracy_score(out, target)
        # print(acc)
        # print(precision)
        # print(recall)
        # print(fscore)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss,prog_bar=False)
        self.log("val_acc", acc,prog_bar=True)
        self.log("val_precision", precision[1],prog_bar=False)
        self.log("val_recall", recall[1],prog_bar=False)
        self.log("val_f1", fscore[1], prog_bar=True)

        return pred, label

    def predict_step(self, batch, batch_idx):
        # print(batch)
        x = batch[0].float()
        y = batch[1].float()

        x = self.ft_enc(x)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        x = self.backbone(x).flatten(start_dim=1)
        x = nn.functional.normalize(x, dim=1, p=2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        pred = self.fc4(x)
        # Logging to TensorBoard (if installed) by default
        return pred, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer