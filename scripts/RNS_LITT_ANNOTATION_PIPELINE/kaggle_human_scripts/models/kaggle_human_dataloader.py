import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch

class KaggleDataset(Dataset):
    def __init__(self, data_X, data_y, test_data_X, labeled=True, transform=True, astensor = True):
        self.data_X = data_X
        self.data_y = np.expand_dims(data_y,axis=1)
        self.test_data_X = test_data_X

        self.transform = transform
        self.labeled = labeled

        if labeled:
            self.data = self.data_X
            self.label = self.data_y
        else:
            self.data = np.vstack((self.data_X, self.test_data_X))
            self.label = np.empty(len(self.data))[:, np.newaxis]

        self.mean = np.mean(self.data)
        self.sd = np.std(self.data)

        self.length = len(self.data)

        if astensor:
            self.augmentation = T.Compose([
                T.Normalize([self.mean, self.mean, self.mean], [self.sd, self.sd, self.sd]),
                T.Resize((256, 512), interpolation=T.InterpolationMode.NEAREST),
                T.ToPILImage(),
                T.RandomApply([T.ColorJitter()], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                T.RandomInvert(p=0.2),
                T.RandomPosterize(4, p=0.2),
                T.ToTensor()
            ])

            self.totensor = T.Compose([
                T.Normalize([self.mean, self.mean, self.mean], [self.sd, self.sd, self.sd]),
                T.Resize((256, 512), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor()
            ])
        else:
            self.augmentation = T.Compose([
                T.Normalize([self.mean, self.mean, self.mean], [self.sd, self.sd, self.sd]),
                T.Resize((256, 512), interpolation=T.InterpolationMode.NEAREST),
                T.ToPILImage(),
                T.RandomApply([T.ColorJitter()], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                T.RandomInvert(p=0.2),
                T.RandomPosterize(4, p=0.2),
            ])

            self.totensor = T.Compose([
                T.Normalize([self.mean, self.mean, self.mean], [self.sd, self.sd, self.sd]),
                # T.ToPILImage(),
                T.Resize((256, 512), interpolation=T.InterpolationMode.NEAREST),
            ])
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        if self.transform:
            channel_index = np.arange(data.shape[0])
            np.random.shuffle(channel_index)
            data = data[channel_index]
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            data = self.augmentation(data)

        else:
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            data = self.totensor(data)

        return data, torch.from_numpy(label).to(dtype=torch.long), None