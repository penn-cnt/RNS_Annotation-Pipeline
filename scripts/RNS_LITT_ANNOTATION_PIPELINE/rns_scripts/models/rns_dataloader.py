import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
import numpy.lib.recfunctions as rfn

data_dir = "../../../user_data/"

class RNSDataset(Dataset):
    def __init__(self, sliced_data, transform=False):
        # load data
        self.data = torch.tensor(np.vstack(sliced_data))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]

        return sample_data.transpose(1,0), idx

class RNS_Raw(Dataset):
    def __init__(self, file_names, transform=True, astensor=True):
        self.file_names = file_names
        self.transform = transform

        print('init_loading')
        file_name_temp = self.file_names[0]
        with open(data_dir + 'rns_cache/' + file_name_temp, 'rb') as f:
            temp_file = np.load(f)

        print('loaded first')
        self.data = np.empty((0, temp_file.shape[1], temp_file.shape[2]))
        # print(self.data.shape)

        for name in tqdm(self.file_names):
            with open(data_dir + 'rns_cache/' + name, 'rb') as f:
                cache = np.load(f)
            self.data = np.vstack((self.data, cache))

        print('data loaded')

        self.length = len(self.data)

        if astensor:
            self.augmentation = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
                T.RandomApply([T.ColorJitter()], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                T.RandomInvert(p=0.2),
                T.RandomPosterize(4, p=0.2),
                T.ToTensor()
            ])

            self.totensor = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor()
            ])
        else:
            self.augmentation = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
                T.RandomApply([T.ColorJitter()], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                T.RandomInvert(p=0.2),
                T.RandomPosterize(4, p=0.2),
            ])

            self.totensor = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
            ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data[index]

        if self.transform:
            concat_len = data.shape[1] / 4
            channel_index = np.arange(4)
            np.random.shuffle(channel_index)
            channel_index = channel_index * concat_len + (concat_len - 1) / 2
            channel_index = np.repeat(channel_index, concat_len)
            concate_len_1 = (concat_len - 1) / 2
            a_repeat = np.arange(-concate_len_1, concate_len_1 + 1)[np.newaxis].T
            base_repeat = np.repeat(a_repeat, 4, axis=1).T.flatten()
            channel_index = channel_index + base_repeat
            data = data[channel_index.astype(int)]
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            data = self.augmentation(data)

        else:
            concat_len = data.shape[1] / 4
            channel_index = np.arange(4)
            # np.random.shuffle(channel_index)
            channel_index = channel_index * concat_len + (concat_len - 1) / 2
            channel_index = np.repeat(channel_index, concat_len)
            concate_len_1 = (concat_len - 1) / 2
            a_repeat = np.arange(-concate_len_1, concate_len_1 + 1)[np.newaxis].T
            base_repeat = np.repeat(a_repeat, 4, axis=1).T.flatten()
            channel_index = channel_index + base_repeat
            data = data[channel_index.astype(int)]
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            data = self.totensor(data)

            # data = torch.from_numpy(data).clone()
            # data = data.repeat(3, 1, 1)
            # data = self.totensor(data)

        return data, [], None


class RNS_Downstream(Dataset):
    def __init__(self, data, label, transform=True, astensor=True):
        self.data = data
        self.label = label
        self.transform = transform
        print('data loaded')

        self.label = self.label[np.newaxis].T

        self.length = len(self.data)

        print(data.shape)
        print(label.shape)

        if astensor:
            self.augmentation = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
                T.RandomApply([T.ColorJitter()], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                T.RandomInvert(p=0.2),
                T.RandomPosterize(4, p=0.2),
                T.ToTensor()
            ])

            self.totensor = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor()
            ])
        else:
            self.augmentation = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
                T.RandomApply([T.ColorJitter()], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                T.RandomInvert(p=0.2),
                T.RandomPosterize(4, p=0.2),
            ])

            self.totensor = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
            ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        if self.transform:
            concat_len = data.shape[1] / 4
            channel_index = np.arange(4)
            np.random.shuffle(channel_index)
            channel_index = channel_index * concat_len + (concat_len - 1) / 2
            channel_index = np.repeat(channel_index, concat_len)
            concate_len_1 = (concat_len - 1) / 2
            a_repeat = np.arange(-concate_len_1, concate_len_1 + 1)[np.newaxis].T
            base_repeat = np.repeat(a_repeat, 4, axis=1).T.flatten()
            channel_index = channel_index + base_repeat
            data = data[channel_index.astype(int)]
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            data = self.augmentation(data)

        else:
            concat_len = data.shape[1] / 4
            channel_index = np.arange(4)
            # np.random.shuffle(channel_index)
            channel_index = channel_index * concat_len + (concat_len - 1) / 2
            channel_index = np.repeat(channel_index, concat_len)
            concate_len_1 = (concat_len - 1) / 2
            a_repeat = np.arange(-concate_len_1, concate_len_1 + 1)[np.newaxis].T
            base_repeat = np.repeat(a_repeat, 4, axis=1).T.flatten()
            channel_index = channel_index + base_repeat
            data = data[channel_index.astype(int)]
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            data = self.totensor(data)

        return data, torch.from_numpy(label).to(dtype=torch.long), None

class RNS_Active(Dataset):
    def __init__(self, data, label, transform=True, astensor=True):
        self.data = data
        self.label = label
        self.transform = transform

        self.data = np.vstack(self.data)


        self.label = np.hstack(self.label)
        self.label = self.label[np.newaxis].T
        print(self.label.shape)
        print(self.data.shape)
        self.length = len(self.data)


        if astensor:
            self.augmentation = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
                T.RandomApply([T.ColorJitter()], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                T.RandomInvert(p=0.2),
                T.RandomPosterize(4, p=0.2),
                T.ToTensor()
            ])

            self.totensor = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor()
            ])
        else:
            self.augmentation = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
                T.RandomApply([T.ColorJitter()], p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
                T.RandomInvert(p=0.2),
                T.RandomPosterize(4, p=0.2),
            ])

            self.totensor = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
            ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        if self.transform:
            concat_len = data.shape[1] / 4
            channel_index = np.arange(4)
            np.random.shuffle(channel_index)
            channel_index = channel_index * concat_len + (concat_len - 1) / 2
            channel_index = np.repeat(channel_index, concat_len)
            concate_len_1 = (concat_len - 1) / 2
            a_repeat = np.arange(-concate_len_1, concate_len_1 + 1)[np.newaxis].T
            base_repeat = np.repeat(a_repeat, 4, axis=1).T.flatten()
            channel_index = channel_index + base_repeat
            data = data[channel_index.astype(int)]
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            data = self.augmentation(data)

        else:
            concat_len = data.shape[1] / 4
            channel_index = np.arange(4)
            # np.random.shuffle(channel_index)
            channel_index = channel_index * concat_len + (concat_len - 1) / 2
            channel_index = np.repeat(channel_index, concat_len)
            concate_len_1 = (concat_len - 1) / 2
            a_repeat = np.arange(-concate_len_1, concate_len_1 + 1)[np.newaxis].T
            base_repeat = np.repeat(a_repeat, 4, axis=1).T.flatten()
            channel_index = channel_index + base_repeat
            data = data[channel_index.astype(int)]
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            data = self.totensor(data)

        return data, torch.from_numpy(label).to(dtype=torch.long), None

def collate_fn(batch):
    info = list(zip(*batch))
    data = info[0]
    label = info[1]
    return torch.stack(data), torch.stack(label)


def get_data(file_names, split=0.7):
    file_name_temp = file_names[0]
    cache = np.load(data_dir + 'rns_test_cache/' + file_name_temp, allow_pickle=True)
    temp_file = cache.item().get('data')

    train_data = np.empty((0, temp_file.shape[1], temp_file.shape[2]))
    train_label = np.array([])
    train_index = np.empty((0, 3))
    test_data = np.empty((0, temp_file.shape[1], temp_file.shape[2]))
    test_label = np.array([])
    test_index = np.empty((0, 3))

    train_ids = np.array([])
    test_ids = np.array([])


    for name in tqdm(file_names):
        cache = np.load(data_dir + 'rns_test_cache/' + name, allow_pickle=True)
        data = cache.item().get('data')
        label = cache.item().get('label')
        index = cache.item().get('indices')
        patientID = cache.item().get('patientID')
        split_n = int(data.shape[0] * (split))


        train_data = np.vstack((train_data, data[:split_n]))
        train_label = np.hstack((train_label, label[:split_n]))
        train_index = np.vstack((train_index, index[:split_n]))
        train_ids = np.hstack((train_ids, patientID[:split_n]))

        test_data = np.vstack((test_data, data[split_n:]))
        test_label = np.hstack((test_label, label[split_n:]))
        test_index = np.vstack((test_index, index[split_n:]))
        test_ids = np.hstack((test_ids, patientID[split_n:]))

    train_index = np.hstack((train_ids[:, np.newaxis], train_index.astype(int)))
    train_index = rfn.unstructured_to_structured(train_index,
                                                 np.dtype([('patient_index', '|S10'), ('episode_index', 'int32'),
                                                           ('slice_index', 'int32'), ('start_index', 'int32')]))

    test_index = np.hstack((test_ids[:, np.newaxis], test_index.astype(int)))
    test_index = rfn.unstructured_to_structured(test_index,
                                                np.dtype([('patient_index', '|S10'), ('episode_index', 'int32'),
                                                          ('slice_index', 'int32'), ('start_index', 'int32')]))

    return train_data, train_label, test_data, test_label, train_index, test_index


def get_data_by_episode(file_names, split=0.7):

    train_data = []
    train_label = []
    train_index = []
    test_data = []
    test_label = []
    test_index = []


    for name in tqdm(file_names):
        cache = np.load(data_dir + 'rns_test_cache/' + name, allow_pickle=True)
        data = cache.item().get('data')
        label = cache.item().get('label')
        index = cache.item().get('indices')
        patientID = cache.item().get('patientID')

        # print(name)
        # print(data.shape)
        # print(label.shape)
        # print(index.shape)
        # print(patientID.shape)
        # print('=====================')

        index = np.hstack((patientID[:, np.newaxis], index.astype(int)))
        index = rfn.unstructured_to_structured(index,
                                                     np.dtype([('patient_index', '|S10'), ('episode_index', 'int32'),
                                                               ('slice_index', 'int32'), ('start_index', 'int32')]))

        data_list = []
        label_list = []
        index_list = []
        for i in range(len(list(np.unique(index['episode_index'])))):

            index_location = np.where(index['episode_index'] == list(np.unique(index['episode_index']))[i])[0]

            sorted_index = [index_location[np.argsort(index[index_location], order=['start_index','slice_index'])]]
            data_list.append(data[sorted_index].squeeze(0))
            label_list.append(label[sorted_index].squeeze(0))
            index_list.append(index[sorted_index].squeeze(0))

        split_n = int(len(data_list) * (split))

        train_data.extend(data_list[:split_n])
        train_label.extend(label_list[:split_n])
        train_index.extend(index_list[:split_n])

        test_data.extend(data_list[split_n:])
        test_label.extend(label_list[split_n:])
        test_index.extend(index_list[split_n:])


    train_data = np.array(train_data, dtype=object)
    train_label = np.array(train_label, dtype=object)
    train_index = np.array(train_index, dtype=object)
    test_data = np.array(test_data, dtype=object)
    test_label = np.array(test_label, dtype=object)
    test_index = np.array(test_index, dtype=object)

    return train_data, train_label, test_data, test_label, train_index, test_index