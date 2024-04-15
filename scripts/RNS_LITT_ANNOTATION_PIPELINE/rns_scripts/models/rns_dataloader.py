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

        return sample_data.transpose(1, 0), idx


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


class RNS_Active(Dataset):
    def __init__(self, data, label, transform=True, astensor=True):
        self.data = data
        self.label = label
        self.transform = transform

        # self.data = np.vstack(self.data)

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


class RNS_Active_by_episode(Dataset):
    def __init__(self, data, label, transform=True, astensor=True):
        self.data = data
        self.label = label
        self.transform = transform

        # self.data = np.vstack(self.data)

        # self.label = np.hstack(self.label)
        # self.label = self.label[np.newaxis].T
        # print(self.label.shape)
        # print(self.data.shape)
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


class RNS_Active_by_episode_LSTM(Dataset):
    def __init__(self, data, label, transform=True, astensor=True):
        self.data = data
        self.label = label
        self.transform = transform

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
        data_arr = self.data[index]
        label_arr = self.label[index]

        data_arr_out = []

        concat_len = data_arr[0].shape[1] / 4
        channel_index = np.arange(4)
        if self.transform:
            np.random.shuffle(channel_index)
        channel_index = channel_index * concat_len + (concat_len - 1) / 2
        channel_index = np.repeat(channel_index, concat_len)
        concate_len_1 = (concat_len - 1) / 2
        a_repeat = np.arange(-concate_len_1, concate_len_1 + 1)[np.newaxis].T
        base_repeat = np.repeat(a_repeat, 4, axis=1).T.flatten()
        channel_index = channel_index + base_repeat

        for i, data in enumerate(data_arr):

            data = data[channel_index.astype(int)]
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            if self.transform:
                data = self.augmentation(data)
            else:
                data = self.totensor(data)
            data_arr_out.append(data)

        data_arr_out = torch.stack(data_arr_out)

        return data_arr_out, torch.from_numpy(label_arr).to(dtype=torch.long), None


# def collate_fn(batch):
#     info = list(zip(*batch))
#     data = info[0]
#     label = info[1]
#     return torch.stack(data), torch.stack(label)

def collate_fn(batch):
    info = list(zip(*batch))
    data = info[0]
    label = info[1]
    sequence_len = [dt.size(0) for dt in data]

    return torch.concat(data), torch.concat(label), sequence_len


def collate_fn_WAAL(batch):
    info = list(zip(*batch))
    if len(info) == 3:
        data = info[0]
        label = info[1]
        sequence_len = [dt.size(0) for dt in data]

        return torch.concat(data), torch.concat(label), sequence_len
    else:
        data_1 = info[0]
        label_1 = info[1]
        data_2 = info[2]
        label_2 = info[3]

        sequence_len_1 = [dt.size(0) for dt in data_1]
        sequence_len_2 = [dt.size(0) for dt in data_2]

        return torch.concat(data_1), torch.concat(label_1), sequence_len_1, \
            torch.concat(data_2), torch.concat(label_2), sequence_len_2


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


def get_data_by_episode(file_names, split=0.7, patient_out=True):
    train_data = []
    train_label = []
    train_index = []
    test_data = []
    test_label = []
    test_index = []

    if patient_out:
        split_n = int(len(file_names) * (split))

        for n, name in tqdm(enumerate(file_names)):
            cache = np.load(data_dir + 'rns_test_cache/' + name, allow_pickle=True)
            data = cache.item().get('data')
            label = cache.item().get('label')
            index = cache.item().get('indices')
            patientID = cache.item().get('patientID')

            index = np.hstack((patientID[:, np.newaxis], index.astype(int)))
            index = rfn.unstructured_to_structured(index,
                                                   np.dtype([('patient_index', '|S10'), ('episode_index', 'int32'),
                                                             ('slice_index', 'int32'), ('start_index', 'int32')]))

            data_list = []
            label_list = []
            index_list = []
            for i in range(len(list(np.unique(index['episode_index'])))):
                index_location = np.where(index['episode_index'] == list(np.unique(index['episode_index']))[i])[0]

                sorted_index = [index_location[np.argsort(index[index_location], order=['start_index', 'slice_index'])]]
                data_list.append(data[sorted_index].squeeze(0))
                label_list.append(label[sorted_index].squeeze(0))
                index_list.append(index[sorted_index].squeeze(0))

            if n < split_n:
                train_data.extend(data_list)
                train_label.extend(label_list)
                train_index.extend(index_list)
            else:
                test_data.extend(data_list)
                test_label.extend(label_list)
                test_index.extend(index_list)


    else:
        for name in tqdm(file_names):
            cache = np.load(data_dir + 'rns_test_cache/' + name, allow_pickle=True)
            data = cache.item().get('data')
            label = cache.item().get('label')
            index = cache.item().get('indices')
            patientID = cache.item().get('patientID')

            index = np.hstack((patientID[:, np.newaxis], index.astype(int)))
            index = rfn.unstructured_to_structured(index,
                                                   np.dtype([('patient_index', '|S10'), ('episode_index', 'int32'),
                                                             ('slice_index', 'int32'), ('start_index', 'int32')]))

            data_list = []
            label_list = []
            index_list = []
            for i in range(len(list(np.unique(index['episode_index'])))):
                index_location = np.where(index['episode_index'] == list(np.unique(index['episode_index']))[i])[0]

                sorted_index = [index_location[np.argsort(index[index_location], order=['start_index', 'slice_index'])]]
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


class Handler_joint(Dataset):

    def __init__(self, X_1, Y_1, X_2, Y_2, transform=None, astensor=True):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

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

        # returning the minimum length of two data-sets

        return min(len(self.X1), len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        x_1_out = []
        x_2_out = []

        concat_len = x_1[0].shape[1] / 4
        channel_index = np.arange(4)
        if self.transform:
            np.random.shuffle(channel_index)
        channel_index = channel_index * concat_len + (concat_len - 1) / 2
        channel_index = np.repeat(channel_index, concat_len)
        concate_len_1 = (concat_len - 1) / 2
        a_repeat = np.arange(-concate_len_1, concate_len_1 + 1)[np.newaxis].T
        base_repeat = np.repeat(a_repeat, 4, axis=1).T.flatten()
        channel_index = channel_index + base_repeat

        for i, data in enumerate(x_1):

            data = data[channel_index.astype(int)]
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            if self.transform:
                data = self.augmentation(data)
            else:
                data = self.totensor(data)
            x_1_out.append(data)

        x_1_out = torch.stack(x_1_out)

        for i, data in enumerate(x_2):

            data = data[channel_index.astype(int)]
            data = torch.from_numpy(data).clone()
            data = data.repeat(3, 1, 1)
            if self.transform:
                data = self.augmentation(data)
            else:
                data = self.totensor(data)
            x_2_out.append(data)

        x_2_out = torch.stack(x_2_out)

        return x_1_out, torch.from_numpy(y_1).to(dtype=torch.long), x_2_out, torch.from_numpy(y_2).to(dtype=torch.long)
