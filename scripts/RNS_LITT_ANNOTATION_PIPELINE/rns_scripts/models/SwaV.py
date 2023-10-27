import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from .rns_dataloader import RNS_Raw
import os
import random
from lightly.loss import SwaVLoss
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.transforms.swav_transform import SwaVTransform
from lightly.data import SwaVCollateFunction

class SwaV(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SwaVProjectionHead(2048, 2048, 128)
        self.prototypes = SwaVPrototypes(128, 2048, 1)
        self.start_queue_at_epoch = 80
        self.queues = nn.ModuleList([MemoryBankModule(size=512) for _ in range(2)])
        self.criterion = SwaVLoss(sinkhorn_epsilon = 0.05)

        data_dir = "../../../user_data/"
        self.dir_list = os.listdir(data_dir+'rns_cache')
        # self.patientIDs = [s for s in dir_list for type_string in ['HUP', 'RNS'] if type_string in s.upper()]

    def train_dataloader(self):
        for _ in range(100):
            file_list = random.sample(self.dir_list,3)
            # if self.current_epoch == 0:
            #     file_list = ['HUP101.npy']
            unlabeled_dataset = RNS_Raw(file_list, transform=True, astensor=False)

            collate_fn = SwaVCollateFunction(gaussian_blur=0, hf_prob=0, vf_prob=0, rr_prob=0, cj_prob=0,
                                             random_gray_scale=0, normalize={'mean': [0, 0, 0], 'std': [1, 1, 1]})

            dataloader = torch.utils.data.DataLoader(
                unlabeled_dataset,
                batch_size=340,
                collate_fn=collate_fn,
                shuffle=True,
                drop_last=True,
                num_workers=2
            )

            if len(dataloader) < 2700:
                return dataloader

    def training_step(self, batch, batch_idx):
        views = batch[0]
        high_resolution, low_resolution = views[:2], views[2:]
        self.prototypes.normalize()

        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]

        high_resolution_prototypes = [
            self.prototypes(x, self.current_epoch) for x in high_resolution_features
        ]
        low_resolution_prototypes = [
            self.prototypes(x, self.current_epoch) for x in low_resolution_features
        ]
        queue_prototypes = self._get_queue_prototypes(high_resolution_features)
        loss = self.criterion(
            high_resolution_prototypes, low_resolution_prototypes, queue_prototypes
        )
        self.log("swav_loss", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim

    def _subforward(self, input):
        features = self.backbone(input).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return features

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features):
        if len(high_resolution_features) != len(self.queues):
            raise ValueError(
                f"The number of queues ({len(self.queues)}) should be equal to the number of high "
                f"resolution inputs ({len(high_resolution_features)}). Set `n_queues` accordingly."
            )

        # Get the queue features
        queue_features = []
        for i in range(len(self.queues)):
            _, features = self.queues[i](high_resolution_features[i], update=True)
            # Queue features are in (num_ftrs X queue_length) shape, while the high res
            # features are in (batch_size X num_ftrs). Swap the axes for interoperability.
            features = torch.permute(features, (1, 0))
            queue_features.append(features)

        # If loss calculation with queue prototypes starts at a later epoch,
        # just queue the features and return None instead of queue prototypes.
        if (
            self.start_queue_at_epoch > 0
            and self.current_epoch < self.start_queue_at_epoch
        ):
            return None

        # Assign prototypes
        queue_prototypes = [
            self.prototypes(x, self.current_epoch) for x in queue_features
        ]
        return queue_prototypes