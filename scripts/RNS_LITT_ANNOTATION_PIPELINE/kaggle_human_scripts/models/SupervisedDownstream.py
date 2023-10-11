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
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 3)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = 0.5
        self.gamma = 8

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.backbone.eval()
        x = self.backbone(x)
        with torch.no_grad():
            x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        pred = self.fc4(x)
        pred = self.softmax(pred)
        label = F.one_hot(y, num_classes=3).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.backbone(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        pred = self.fc4(x)
        pred = self.softmax(pred)
        label = F.one_hot(y, num_classes=3).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')
        out = torch.argmax(pred, dim=1)
        out = out.detach().cpu().numpy()
        target = y.squeeze().detach().cpu().numpy()
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(out, target)
        acc = sklearn.metrics.accuracy_score(out, target)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", precision[1])
        self.log("val_recall", recall[1])
        return pred, label

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        emb = self.backbone(x)
        emb = emb.view(-1, 512)
        x = F.relu(self.fc1(emb))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        pred = self.fc4(x)
        # Logging to TensorBoard (if installed) by default
        return pred, y, emb

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
