#!/usr/bin/env python
# coding: utf-8

# In[1]:

#
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'widget')


# In[2]:


import sys
import os

sys.path.append('../tools')
import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from torch import nn
from tqdm import tqdm
import sklearn

import torchvision.transforms as T
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks

import data_utility
import times
import segmentation
import preprocess
import autoencoder
import visualizer
import kaggle_data_utility
from kaggle_data_utility import KaggleDataset

from models.kaggle_human_dataloader import KaggleDataset

from models.SwaV import SwaV
from models.SupervisedDownstream import SupervisedDownstream
from lightly.data import SwaVCollateFunction


# In[3]:


data_dir = "../../../user_data/competition_data/clips"
log_folder_root = '../../../user_data/logs/'
ckpt_folder_root = '../../../user_data/checkpoints/'

# targets = [
#     'Dog_1',
#     'Dog_2',
#     'Dog_3',
#     'Dog_4',
# ]

targets = [
    'Patient_1',
    'Patient_2',
    'Patient_3',
    'Patient_4',
    'Patient_5',
    'Patient_6',
    'Patient_7',
    'Patient_8'
]


# In[4]:




# In[5]:


# ictal_data_list = [kaggle_data_utility.parse_input_data(data_dir, targets[i], 'ictal', None) for i in
#                    range(len(targets))]
# interictal_data_list = [kaggle_data_utility.parse_input_data(data_dir, targets[i], 'interictal', None) for i in
#                         range(len(targets))]
# test_data_list = [kaggle_data_utility.parse_input_data(data_dir, targets[i], 'test', None) for i in range(len(targets))]


# In[6]:


# ictal_data_X = np.empty((0, 16, 400))
# interictal_data_X = np.empty((0, 16, 400))
# test_data_X = np.empty((0, 16, 400))
# for data in ictal_data_list:
#     ictal_data_X = np.vstack((ictal_data_X, data['X']))
# for data in interictal_data_list:
#     interictal_data_X = np.vstack((interictal_data_X, data['X']))
# for data in test_data_list:
#     test_data_X = np.vstack((test_data_X, data['X']))


# In[7]:


# # for data in ictal_data_list:
# import torch
# import matplotlib.pyplot as plt
#
# from torchvision.transforms import Resize
#
common_size = 72
# resize = Resize((common_size,500))
data_X = np.empty((0, common_size,500))
test_data_X = np.empty((0, common_size,500))
# data_y = np.empty(0)
#
# for i in tqdm(range(len(ictal_data_list))):
#     ictal_data = ictal_data_list[i]['X']
#     interictal_data = interictal_data_list[i]['X']
#     test_data = test_data_list[i]['X']
#     ictal_label = ictal_data_list[i]['y']
#     interictal_label = interictal_data_list[i]['y']
#
#
#     data = np.concatenate((ictal_data,interictal_data))
#     label = np.concatenate((ictal_label,interictal_label))
#
#     del ictal_data, interictal_data, ictal_label, interictal_label
#
#     # print(ictal_data.shape)
#     # print(ictal_label.shape)
#     # print(interictal_data.shape)
#     # print(interictal_label.shape)
#
#     # print(np.max(ictal_data))
#     # print(np.min(ictal_data))
#     # print(np.max(interictal_data))
#     # print(np.min(interictal_data))
#     # print(np.std(ictal_data))
#     # print(np.std(interictal_data))
#     # print(np.std(test_data))
#     # print(ictal_data.shape)
#     # print(interictal_data.shape)
#     # print(test_data.shape)
#     # print('===============')
#     avg =  np.mean(np.concatenate([data,test_data]))
#     std = np.std(np.concatenate([data,test_data]))
#
#     data = (data-avg)/std
#     test_data = (test_data-avg)/std
#
#     if data.shape[-1] == 5000:
#         data = np.concatenate(np.array(np.split(data, 10, axis=2)).transpose(1,0,2,3))[:,np.newaxis,:,:]
#         data = resize(torch.tensor(data)).numpy().squeeze(1)
#         label = np.repeat(label,10)
#         test_data = np.concatenate(np.array(np.split(test_data, 10, axis=2)).transpose(1,0,2,3))[:,np.newaxis,:,:]
#         test_data = resize(torch.tensor(test_data)).numpy().squeeze(1)
#         assert data.shape[0] == label.shape[0]
#     elif data.shape[-1] == 500:
#         data = resize(torch.tensor(data[:,np.newaxis,:,:])).numpy().squeeze(1)
#         test_data = resize(torch.tensor(test_data[:,np.newaxis,:,:])).numpy().squeeze(1)
#         assert data.shape[0] == label.shape[0]
#
#     np.save('../../../user_data/kaggle_human_cache/data_X_' + str(i) + '.npy',data)
#     np.save('../../../user_data/kaggle_human_cache/data_y_' + str(i) + '.npy',label)
#     np.save('../../../user_data/kaggle_human_cache/test_data_X_' + str(i) + '.npy',test_data)



# In[8]:


# common_size = 72
# # resize = Resize((common_size,500))
# data_X = np.empty((0, common_size,500))
# test_data_X = np.empty((0, common_size,500))
# data_y = np.empty(0)
#
# for i in tqdm(range(8)):
#     data_y = np.hstack((data_y, np.load('../../../user_data/kaggle_human_cache/data_y_' + str(i) + '.npy')))


# In[9]:





# In[10]:



# In[11]:

if __name__ == "__main__":
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    determine_generator = torch.Generator()
    determine_generator.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        # True ensures the algorithm selected by CUFA is deterministic
        torch.backends.cudnn.deterministic = True
        # torch.set_deterministic(True)
        # False ensures CUDA select the same algorithm each time the application is run
        torch.backends.cudnn.benchmark = False
    print('load_data')

    data_X = np.load('../../../user_data/kaggle_human_cache/data_X.npy')
    data_y = np.load('../../../user_data/kaggle_human_cache/data_y.npy')
    # test_data_X = np.load('../../../user_data/kaggle_human_cache/test_data_X.npy')
    print('data_loaded')

    unlabeled_dataset = KaggleDataset(data_X, data_y, labeled=False, transform=True, astensor = False)
    print('dataloaoder_loaded')
    model = SwaV()

    del data_X, data_y

    # model.load_from_checkpoint()

    collate_fn = SwaVCollateFunction(gaussian_blur=0, hf_prob=0, vf_prob=0, rr_prob=0, cj_prob=0, random_gray_scale=0,
                                     normalize={'mean': [0, 0, 0], 'std': [1, 1, 1]})

    dataloader = torch.utils.data.DataLoader(
        unlabeled_dataset,
        batch_size=384,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    checkpoint_callback = pl_callbacks.ModelCheckpoint(monitor='swav_loss',
                                                       filename='kaggle_human_swav-{epoch:02d}-{swav_loss:.5f}',
                                                       save_last=True, save_top_k=-1,
                                                       dirpath=ckpt_folder_root + 'kaggle_human_swav_34')
    csv_logger = pl_loggers.CSVLogger(log_folder_root, name="kaggle_human_swav")

    trainer = pl.Trainer(logger=csv_logger, max_epochs=120, callbacks=[checkpoint_callback], accelerator='gpu',
                         devices=1, precision=16)

    # In[ ]:

    trainer.fit(model=model, train_dataloaders=dataloader)

    # random_seed = 42
    # torch.manual_seed(random_seed)
    # np.random.seed(random_seed)
    # determine_generator = torch.Generator()
    # determine_generator.manual_seed(random_seed)
    #
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(random_seed)
    #     # True ensures the algorithm selected by CUFA is deterministic
    #     torch.backends.cudnn.deterministic = True
    #     # torch.set_deterministic(True)
    #     # False ensures CUDA select the same algorithm each time the application is run
    #     torch.backends.cudnn.benchmark = False
    # print('load_data')
    #
    # data_X = np.load('../../../user_data/kaggle_human_cache/data_X.npy')
    # data_y = np.load('../../../user_data/kaggle_human_cache/data_y.npy')
    # test_data_X = np.load('../../../user_data/kaggle_human_cache/test_data_X.npy')
    # print('data_loaded')
    #
    # unlabeled_dataset = KaggleDataset(data_X, data_y, test_data_X, labeled=False, transform=True,astensor = False)
    # print('dataloaoder_loaded')
    # model = SwaV()
    #
    # # model.load_from_checkpoint()
    #
    # collate_fn = SwaVCollateFunction(gaussian_blur=0, hf_prob=0, vf_prob=0, rr_prob=0, cj_prob=0, random_gray_scale=0,
    #                                  normalize={'mean': [0, 0, 0], 'std': [1, 1, 1]})
    #
    # dataloader = torch.utils.data.DataLoader(
    #     unlabeled_dataset,
    #     batch_size=320,
    #     collate_fn=collate_fn,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=4
    # )
    # accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    #
    # checkpoint_callback = pl_callbacks.ModelCheckpoint(monitor='swav_loss',
    #                                                    filename='kaggle_human_swav-{epoch:02d}-{swav_loss:.5f}',
    #                                                    save_last=True, save_top_k=-1,
    #                                                    dirpath=ckpt_folder_root + 'kaggle_human_swav_34')
    # csv_logger = pl_loggers.CSVLogger(log_folder_root, name="kaggle_human_swav")
    #
    # trainer = pl.Trainer(logger=csv_logger, max_epochs=120, callbacks=[checkpoint_callback], accelerator='gpu',
    #                      devices=1, precision=16)
    #
    # # In[ ]:
    #
    # trainer.fit(model=model, train_dataloaders=dataloader)

# In[12]:





# In[13]:



#


#
# # In[12]:
#
#
# # unlabeled_dataset = KaggleDataset(ictal_data_X, interictal_data_X, test_data_X, labeled=False, transform=True)
# # train_set_size = int(labeled_dataset.length * 0.8)
# # valid_set_size = labeled_dataset.length - train_set_size
# # train_set, test_set = torch.utils.data.random_split(labeled_dataset, [train_set_size, valid_set_size])
#
#
# # In[13]:
#
#
# # swav = SwaV().load_from_checkpoint(ckpt_folder_root + 'kaggle_dog_swav_18/kaggle_dog_swav-epoch=116-swav_loss=2.23095.ckpt')
# # swav = SwaV().load_from_checkpoint(ckpt_folder_root + 'kaggle_dog_swav_34/kaggle_dog_swav-epoch=116-swav_loss=2.73583.ckpt')
# swav = SwaV().load_from_checkpoint(ckpt_folder_root + 'kaggle_dog_swav_50/kaggle_dog_swav-epoch=118-swav_loss=3.64062.ckpt')
# model = SupervisedDownstream(swav.backbone)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
#
# checkpoint_callback = pl_callbacks.ModelCheckpoint(monitor='val_loss',filename='kaggle_dog_linear_eval-{epoch:02d}-{val_loss:.5f}', save_last=True, save_top_k=-1, dirpath=ckpt_folder_root + 'kaggle_dog_swav_50_linear_eval')
# csv_logger = pl_loggers.CSVLogger(log_folder_root, name="kaggle_dog_swav_50_linear_eval")
#
# trainer = pl.Trainer( logger=csv_logger, max_epochs=80, callbacks=[checkpoint_callback],accelerator='gpu', devices=1)
#
#
# # In[14]:
#
#
# labeled_dataset = KaggleDataset(ictal_data_X, interictal_data_X, test_data_X, labeled=True, transform=True)
# train_set_size = int(labeled_dataset.length * 0.8)
# valid_set_size = labeled_dataset.length - train_set_size
# train_set, test_set = torch.utils.data.random_split(labeled_dataset, [train_set_size, valid_set_size],generator=torch.Generator().manual_seed(42))
#
# def collate_fn(batch):
#     info = list(zip(*batch))
#     data = info[0]
#     label = info[1]
#     return torch.stack(data), torch.stack(label)
#
# train_dataloader = torch.utils.data.DataLoader(
#     train_set,
#     batch_size=64,
#     collate_fn=collate_fn,
#     shuffle=True,
#     drop_last=True,
# )
# val_dataloader = torch.utils.data.DataLoader(
#     test_set,
#     batch_size=64,
#     collate_fn=collate_fn,
#     shuffle=False,
#     drop_last=True,
# )
# trainer.fit(model, train_dataloader, val_dataloader)
#
#
# # In[17]:
#
#
# labeled_dataset = KaggleDataset(ictal_data_X, interictal_data_X, test_data_X, labeled=True, transform=False)
# train_set_size = int(labeled_dataset.length * 0.8)
# valid_set_size = labeled_dataset.length - train_set_size
# train_set, test_set = torch.utils.data.random_split(labeled_dataset, [train_set_size, valid_set_size],generator=torch.Generator().manual_seed(42))
# train_dataloader = torch.utils.data.DataLoader(
#     train_set,
#     batch_size=128,
#     collate_fn=collate_fn,
#     shuffle=True,
#     drop_last=True,
# )
# val_dataloader = torch.utils.data.DataLoader(
#     test_set,
#     batch_size=128,
#     collate_fn=collate_fn,
#     shuffle=False,
#     drop_last=True,
# )
# predictions = trainer.predict(model,val_dataloader,ckpt_path='checkpoints/swav_pretrained_unfreeze-epoch=71-val_loss=0.00116.ckpt')
#
#
# # In[ ]:
#
#
# train_set
#
#
# # In[18]:
#
#
# output_list = []
# target_list = []
# m = nn.Softmax(dim=1)
# for pred, y,_ in predictions:
#     out = m(pred)
#     print(out)
#     output_list.append(out)
#     target_list.append(y)
#
#
# # In[19]:
#
#
# output = torch.vstack(output_list)
# target = torch.vstack(target_list)
# inds = np.where(target==1)[0]
#
# fpr, tpr, thresholds = sklearn.metrics.roc_curve(target,output[:,1],pos_label=1)
# sklearn.metrics.auc(fpr, tpr)
#
#
# # In[20]:
#
#
# output[:,1][inds]
#
#
# # In[21]:
#
#
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_predictions(
#     target,
#     output[:,1],
#     color="darkorange",
# )
# plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("One-vs-Rest ROC curves:\nVirginica vs (Setosa & Versicolor)")
# plt.legend()
# plt.show()
#
#
# # In[23]:
#
#
# output[:,1]
#
#
# # In[23]:
#
#
#
#
#
# # In[24]:
#
#
# output = torch.argmax(output, dim=1)
# output = output.detach().cpu().numpy()
# target= target.squeeze().detach().cpu().numpy()
#
#
# # In[24]:
#
#
#
#
#
# # In[25]:
#
#
# import sklearn
#
# clf_report = sklearn.metrics.classification_report(output, target, digits=6)
#
# print(f"Classification Report : \n{clf_report}")
#
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
# for batch, label in tqdm(val_dataloader):
#         batch = batch.to(device)
#         label = label.to(device)
#         label = F.one_hot(label).squeeze()
#         outputs = model(batch)
#         print(batch)
#         loss = sigmoid_focal_loss(pred.float(),label.float(), alpha = 0.5, gamma = 8,reduction = 'mean')
#         print(loss)
#         break
#
#
# # In[ ]:
#
#
# # import copy
# # import torch
# # import torchvision
# # from torch import nn
# #
# # from lightly.data import DINOCollateFunction, LightlyDataset
# # from lightly.loss import DINOLoss
# # from lightly.models.modules import DINOProjectionHead
# # from lightly.models.utils import deactivate_requires_grad, update_momentum
# # from lightly.utils.scheduler import cosine_schedule
# #
# #
# # class DINO(torch.nn.Module):
# #     def __init__(self, backbone, input_dim):
# #         super().__init__()
# #         self.student_backbone = backbone
# #         self.student_head = DINOProjectionHead(
# #             input_dim, 512, 64, 2048, freeze_last_layer=1
# #         )
# #         self.teacher_backbone = copy.deepcopy(backbone)
# #         self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
# #         deactivate_requires_grad(self.teacher_backbone)
# #         deactivate_requires_grad(self.teacher_head)
# #
# #     def forward(self, x):
# #         y = self.student_backbone(x).flatten(start_dim=1)
# #         z = self.student_head(y)
# #         return z
# #
# #     def forward_teacher(self, x):
# #         y = self.teacher_backbone(x).flatten(start_dim=1)
# #         z = self.teacher_head(y)
# #         return z
# #
# #
# # resnet = torchvision.models.resnet18()
# # backbone = nn.Sequential(*list(resnet.children())[:-1])
# # input_dim = 512
# # # instead of a resnet you can also use a vision transformer backbone as in the
# # # original paper (you might have to reduce the batch size in this case):
# # # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
# # # input_dim = backbone.embed_dim
# #
# # model = DINO(backbone, input_dim)
# #
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model.to(device)
# #
# # # # we ignore object detection annotations by setting target_transform to return 0
# # # pascal_voc = torchvision.datasets.VOCDetection(
# # #     "datasets/pascal_voc", download=True, target_transform=lambda t: 0
# # # )
# # # dataset = LightlyDataset.from_torch_dataset(pascal_voc)
# # # # or create a dataset from a folder containing images or videos:
# # # # dataset = LightlyDataset("path/to/folder")
# #
# # collate_fn = DINOCollateFunction(solarization_prob = 0, hf_prob = 0,vf_prob = 0,rr_prob=0,cj_prob=0,random_gray_scale=0)
# #
# # dataloader = torch.utils.data.DataLoader(
# #     train_set,
# #     batch_size=64,
# #     collate_fn=collate_fn,
# #     shuffle=True,
# #     drop_last=True,
# #     num_workers=1,
# # )
# #
# # criterion = DINOLoss(
# #     output_dim=2048,
# #     warmup_teacher_temp_epochs=5,
# # )
# # # move loss to correct device because it also contains parameters
# # criterion = criterion.to(device)
# #
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# #
# # epochs = 10
# #
# # print("Starting Training")
# # for epoch in range(epochs):
# #     total_loss = 0
# #     momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
# #     for views, _, _ in tqdm(dataloader):
# #         update_momentum(model.student_backbone, model.teacher_backbone, m=momentum_val)
# #         update_momentum(model.student_head, model.teacher_head, m=momentum_val)
# #         views = [view.to(device) for view in views]
# #         global_views = views[:2]
# #         teacher_out = [model.forward_teacher(view) for view in global_views]
# #         student_out = [model.forward(view) for view in views]
# #         loss = criterion(teacher_out, student_out, epoch=epoch)
# #         total_loss += loss.detach()
# #         loss.backward()
# #         # We only cancel gradients of student head.
# #         model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
# #         optimizer.step()
# #         optimizer.zero_grad()
# #
# #     avg_loss = total_loss / len(dataloader)
# #     print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
#
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
# augmentation = T.Compose([
#     T.ToPILImage(),
#     T.Resize((256, 512), interpolation=T.InterpolationMode.NEAREST),
#     T.RandomApply([T.ColorJitter()], p=0.5),
#     T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.5),
#     T.RandomInvert(p=0.2),
#     T.RandomPosterize(4, p=0.2),
# ])
#
# data = ictal_data_X[0]
#
# channel_index = np.arange(data.shape[0])
# np.random.shuffle(channel_index)
# data = data[channel_index]
# data = torch.from_numpy(data).clone()
# data = data.repeat(3, 1, 1)
# data = augmentation(data)
# data
#
#
# # In[ ]:
#
#
# channel_index
#
#
# # In[ ]:
#
#
# data[channel_index]
#
#
# # In[ ]:
#
#
# data
#
#
# # In[ ]:
#
#
# #
# # print("Starting Training")
# # for epoch in range(50):
# #     total_loss = 0
# #     i = 0
# #     for batch, label in tqdm(dataloader):
# #         batch = batch.to(device)
# #         # print(type(batch))
# #         label = label.to(device)
# #         label = F.one_hot(label).squeeze()
# #         outputs = model(batch)
# #         loss = sigmoid_focal_loss(outputs.float(),label.float(), alpha = 0.25, gamma = 7,reduction = 'mean')
# #         total_loss += loss.detach()
# #         loss.backward()
# #         optimizer.step()
# #         optimizer.zero_grad()
# #
# #     avg_loss = total_loss / len(dataloader)
# #     torch.save({
# #             'epoch': epoch,
# #             'model_state_dict': model.state_dict(),
# #             'optimizer_state_dict': optimizer.state_dict(),
# #             'loss': avg_loss,
# #             }, 'ckpt/checkpoint'+str(epoch)+'.pth')
# #
# #     print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
#
