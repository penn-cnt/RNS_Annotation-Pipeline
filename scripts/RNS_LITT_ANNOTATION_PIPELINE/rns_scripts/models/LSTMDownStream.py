import torch.nn.functional as F
from torch import nn
import torch
import sklearn
import pytorch_lightning as pl
import sys
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, unpad_sequence

sys.path.append('../tools')
from sigmoid_loss import sigmoid_focal_loss


class SupervisedDownstream(pl.LightningModule):
    def __init__(self, backbone, unfreeze_backbone_at_epoch=100):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 64)
        self.dp = nn.Dropout1d(p=0.2)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = 0
        self.gamma = 5
        self.lstm = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.enable_mc_dropout = False

    def forward(self, x, seq_len):
        if self.current_epoch < self.unfreeze_backbone_at_epoch:
            self.backbone.eval()
            x = self.backbone(x)
            with torch.no_grad():
                emb = x.view(-1, 2048)
        else:
            x = self.backbone(x)
            emb = x.view(-1, 2048)

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

        return pred, emb, emb_t

    def training_step(self, batch, batch_idx):
        x, y, seq_len = batch
        self.enable_mc_dropout = True
        pred, _, _ = self(x, seq_len)
        self.enable_mc_dropout = False
        pred = self.softmax(pred)
        label = F.one_hot(y, num_classes=2).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, seq_len = batch
        pred, _, _ = self(x, seq_len)
        pred = self.softmax(pred)
        label = F.one_hot(y, num_classes=2).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')
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
        # print(batch)
        x, y, seq_len = batch
        pred, emb, emb_t = self(x, seq_len)
        # Logging to TensorBoard (if installed) by default
        return pred, y, emb, emb_t, seq_len

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
