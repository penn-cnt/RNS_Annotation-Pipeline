import sys

sys.path.append('../tools')

from torch.utils.data import Dataset
import kaggle_data_utility
import torchvision.transforms as T
import torch
import numpy as np


class KaggleDataset(Dataset):
    def __init__(self, data_X, data_y, test_data_X, labeled=True, transform=True, astensor=True):
        self.data_X = data_X
        self.data_y = data_y
        self.test_data_X = test_data_X

        self.transform = transform
        self.labeled = labeled

        data_full = np.vstack((self.data_X, self.test_data_X))
        self.mean = np.mean(data_full)
        self.sd = np.std(data_full)

        if labeled:
            self.data = data_X
            self.label = data_y
        else:
            self.data = np.vstack((self.data_X, self.test_data_X))
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


class ActiveDataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.mean = 0.0
        self.sd = 63.05380081813939

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

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        if self.transform:
            channel_index = np.arange(x.shape[0])
            np.random.shuffle(channel_index)
            x = x[channel_index]
            x = torch.from_numpy(x).clone()
            x = x.repeat(3, 1, 1)
            x = self.augmentation(x)

        else:
            x = torch.from_numpy(x).clone()
            x = x.repeat(3, 1, 1)
            x = self.totensor(x)

        return x, torch.from_numpy(y).to(dtype=torch.long), index

    def __len__(self):
        return len(self.X)




def collate_fn(batch):
    info = list(zip(*batch))
    data = info[0]
    label = info[1]

    return torch.stack(data), torch.stack(label)


def load_annotated_data(data_dir, targets):
    ictal_data_list = [kaggle_data_utility.parse_input_data(data_dir, targets[i], 'ictal', None) for i in
                       range(len(targets))]
    interictal_data_list = [kaggle_data_utility.parse_input_data(data_dir, targets[i], 'interictal', None) for i in
                            range(len(targets))]
    ictal_data_X = np.empty((0, 16, 400))
    interictal_data_X = np.empty((0, 16, 400))
    for data in ictal_data_list:
        ictal_data_X = np.vstack((ictal_data_X, data['X']))
    for data in interictal_data_list:
        interictal_data_X = np.vstack((interictal_data_X, data['X']))

    data_X = np.empty((0, 16, 400))
    data_y = np.empty((0, 1))
    for i in range(len(ictal_data_list)):
        ictal_data = ictal_data_list[i]['X']
        interictal_data = interictal_data_list[i]['X']
        ictal_label = ictal_data_list[i]['y']
        interictal_label = interictal_data_list[i]['y']

        data = np.concatenate((ictal_data, interictal_data))
        label = np.concatenate((ictal_label, interictal_label))

        data_X = np.vstack((data_X, data))
        data_y = np.vstack((data_y, label[:, np.newaxis]))

    print(data_X.shape)

    return data_X, data_y
