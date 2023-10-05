import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch

class KaggleDataset(Dataset):
    def __init__(self, ictal_data_X, interictal_data_X, test_data_X, inctal_data_y, interictal_data_y, labeled=True, transform=True, astensor = True):
        self.ictal_data_X = ictal_data_X
        self.interictal_data_X = interictal_data_X
        self.test_data_X = test_data_X
        self.ictal_data_y = np.ones(len(self.ictal_data_X))[:, np.newaxis]
        self.interictal_data_y = np.zeros(len(self.interictal_data_X))[:, np.newaxis]

        self.transform = transform
        self.labeled = labeled

        data_full = np.vstack((self.ictal_data_X, self.interictal_data_X))
        data_full = np.vstack((data_full, self.test_data_X))

        self.mean = np.mean(data_full)
        self.sd = np.std(data_full)

        if labeled:
            self.data = np.vstack((self.ictal_data_X, self.interictal_data_X))
            self.label = np.vstack((self.ictal_data_y, self.interictal_data_y))
        else:
            self.data = np.vstack((self.ictal_data_X, self.interictal_data_X))
            self.data = np.vstack((self.data, self.test_data_X))
            self.label = np.empty(len(self.data))[:, np.newaxis]

        self.length = len(self.data)

        if astensor:
            self.augmentation = T.Compose([
                T.Normalize([self.mean, self.mean, self.mean], [self.sd, self.sd, self.sd]),
                T.ToPILImage(),
                T.Resize((256, 512), interpolation=T.InterpolationMode.NEAREST),
                T.RandomApply([T.ColorJitter()], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                T.RandomInvert(p=0.2),
                T.RandomPosterize(4, p=0.2),
                T.ToTensor()
            ])

            self.totensor = T.Compose([
                T.Normalize([self.mean, self.mean, self.mean], [self.sd, self.sd, self.sd]),
                T.ToPILImage(),
                T.Resize((256, 512), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor()
            ])
        else:
            self.augmentation = T.Compose([
                T.Normalize([self.mean, self.mean, self.mean], [self.sd, self.sd, self.sd]),
                T.ToPILImage(),
                T.Resize((256, 512), interpolation=T.InterpolationMode.NEAREST),
                T.RandomApply([T.ColorJitter()], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                T.RandomInvert(p=0.2),
                T.RandomPosterize(4, p=0.2),
            ])

            self.totensor = T.Compose([
                T.Normalize([self.mean, self.mean, self.mean], [self.sd, self.sd, self.sd]),
                T.ToPILImage(),
                T.Resize((256, 512), interpolation=T.InterpolationMode.NEAREST),
            ])
