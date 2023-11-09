import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sklearn
import sys
import numpy as np
sys.path.append('../tools')
from sigmoid_loss import sigmoid_focal_loss
class Classifier(pl.LightningModule):
    def __init__(self, backbone, input_dim = 512, unfreeze_backbone_at_epoch = -1):
        super().__init__()
        self.backbone = backbone
        self.input_dim = input_dim
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 64)
        self.dp = nn.Dropout1d(p=0.2)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 3)

    def forward(self, x):
        if self.unfreeze_backbone_at_epoch == -1 or self.current_epoch < self.unfreeze_backbone_at_epoch:
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

class Discriminator(pl.LightningModule):
    def __init__(self, dim=32):
        super(Discriminator, self).__init__()
        self.dim = np.prod(dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        self.weight_init()

    def forward(self, z):
            return self.net(z).reshape(-1)


class WAAL(pl.LightningModule):
    def __init__(self, net_fea, net_clf, net_dis, gamma_ratio=0.1, alpha=1):
        super().__init__()
        self.net_fea = net_fea
        self.net_clf = net_clf
        self.net_dis = net_dis
        self.gamma_ratio = gamma_ratio
        self.alpha = alpha

    def training_step(self, batch, batch_idx):
        label_x, label_y, unlabel_x, _ = batch

        label_x, label_y = label_x.to(self.device), label_y.to(self.device)
        unlabel_x = unlabel_x.to(self.device)

        # training feature extractor and predictor
        self.set_requires_grad([self.net_fea, self.net_clf], True)
        self.set_requires_grad(self.net_dis, False)

        lb_z = self.net_fea(label_x)
        unlb_z = self.net_fea(unlabel_x)

        lb_out, _ = self.net_clf(lb_z)

        # prediction loss (based on provided classifier)
        pred_loss = F.cross_entropy(lb_out, label_y)

        # Wasserstein loss (unbalanced loss, used the redundant trick)
        wassertein_distance = self.net_dis(unlb_z).mean() - self.gamma_ratio * self.net_dis(lb_z).mean()

        gp = self.gradient_penalty(self.net_dis, unlb_z, lb_z)

        loss = pred_loss + self.alpha * wassertein_distance + self.alpha * gp * 5

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Feature extraction
        z = self.net_fea(x)

        # Prediction
        _, pred = self.net_clf(z)

        # Calculate loss using sigmoid focal loss
        label = F.one_hot(y, num_classes=3).squeeze()
        loss = sigmoid_focal_loss(pred.float(), label.float(), alpha=self.alpha, gamma=self.gamma, reduction='mean')

        # Calculate precision, recall, f-score, and accuracy
        out = torch.argmax(pred, dim=1)
        out = out.detach().cpu().numpy()
        target = y.squeeze().detach().cpu().numpy()
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(out, target,
                                                                                             zero_division=0.0)
        acc = sklearn.metrics.accuracy_score(out, target)

        # Log metrics to TensorBoard
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", precision[0])
        self.log("val_recall", recall[0])
        self.log("val_fscore", fscore[0])

        return pred, label

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, _ = batch

        # Feature extraction
        z = self.net_fea(x)

        # Prediction
        _, pred = self.net_clf(z)

        # Return the predicted probabilities
        return pred

    def configure_optimizers(self):
        # Define optimizers for feature extractor and classifier
        optimizer_fea = optim.Adam(self.net_fea.parameters())
        optimizer_clf = optim.Adam(self.net_clf.parameters())

        # Define scheduler for feature extractor
        scheduler_fea = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fea, patience=2, factor=0.2)

        # Define scheduler for classifier
        scheduler_clf = optim.lr_scheduler.ReduceLROnPlateau(optimizer_clf, patience=2, factor=0.2)

        # Return a list of optimizers and schedulers
        return [optimizer_fea, optimizer_clf], [scheduler_fea, scheduler_clf]

    def single_worst(self, probas):

        """
        The single worst will return the max_{k} -log(proba[k]) for each sample
        :param probas:
        :return:  # unlabeled \times 1 (tensor float)
        """

        value, _ = torch.max(-1 * torch.log(probas), 1)

        return value

    # setting gradient values
    def set_requires_grad(self, model, requires_grad=True):
        """
        Used in training adversarial approach
        :param model:
        :param requires_grad:
        :return:
        """

        for param in model.parameters():
            param.requires_grad = requires_grad

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