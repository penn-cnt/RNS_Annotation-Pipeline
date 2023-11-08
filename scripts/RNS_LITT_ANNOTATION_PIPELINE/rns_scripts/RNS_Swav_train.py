#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


import sys
import os

sys.path.append('../tools')

import multicrop_dataset
import architecture

# In[3]:


import numpy as np
import matplotlib

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import sklearn

import torchvision.transforms as T
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks

import data_utility

from itertools import groupby
from tqdm import tqdm

# In[4]:
from models.rns_dataloader import RNS_Raw
# unlabeled_dataset = RNS_Raw(file_list, transform=True,astensor = False)
from lightly.data import SwaVCollateFunction

# In[5]:


data_dir = "../../../user_data/"
log_folder_root = '../../../user_data/logs/'
ckpt_folder_root = '../../../user_data/checkpoints/'

# In[6]:


# dir_list = os.listdir(data_dir+'rns_data')
# patientIDs = [s for s in dir_list for type_string in ['HUP', 'RNS'] if type_string in s.upper()]


# In[7]:


# os.listdir(data_dir+'rns_cache')


# In[8]:


# data_import = data_utility.read_files(path = data_dir+'rns_data', path_data = data_dir+'rns_raw_cache',patientIDs=patientIDs[10:], annotation_only = False, verbose=True)
# ids = list(data_import.keys())


# In[9]:


# window_len = 1
# stride = 1
# concat_n = 2
# for id in tqdm(ids):
#     data_import[id].set_window_parameter(window_length=window_len, window_displacement=stride)
#     data_import[id].set_concatenation_parameter(concatenate_window_n=concat_n)
#     data_import[id].get_windowed_data(data_import[id].catalog["Event Start idx"],data_import[id].catalog["Event End idx"])
#     data_import[id].normalize_windowed_data()
#     _, concatenated_data = data_import[id].get_concatenated_data(data_import[id].windowed_data, arrange='channel_stack')
#     np.save(data_dir+'rns_cache/'+ id +'.npy',concatenated_data)


# In[9]:


# In[10]:


file_list = ['HUP159.npy',
 'HUP096.npy',
 'RNS026.npy',
 'HUP156.npy',
 'HUP137.npy',
 'RNS029.npy',
 'HUP147.npy',
 'HUP129.npy',
 'HUP047.npy',
 'HUP101.npy',
 'HUP192.npy',
 'HUP205.npy',
 'HUP143.npy',
 'HUP128.npy',
 'HUP197.npy',
 'HUP059.npy',
 'HUP109.npy',
 'RNS022.npy',
 ]
print(file_list)
# file_list = ['HUP084.npy','HUP131.npy','HUP096.npy']
# if self.current_epoch == 0:
#     file_list = ['HUP101.npy']
# file_list = [file_list[0]]
print('loading data')

# file_list = ['RNS026.npy', 'HUP159.npy', 'HUP129.npy', 'HUP096.npy']
unlabeled_dataset = RNS_Raw(file_list, transform=True, astensor=False)

collate_fn = SwaVCollateFunction(gaussian_blur=0, hf_prob=0, vf_prob=0, rr_prob=0, cj_prob=0,
                                 random_gray_scale=0, normalize={'mean': [0, 0, 0], 'std': [1, 1, 1]})

dataloader = torch.utils.data.DataLoader(
    unlabeled_dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=10
)

print('data loaded')

# In[11]:


from models.SwaV import SwaV
from lightly.data import SwaVCollateFunction

#
# model = SwaV()

# ckpt = torch.load(ckpt_folder_root+ 'rns_swav_50_5/checkpoint31.pth')
# resnet = torchvision.models.resnet50()
# backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SwaV()
# model.load_state_dict(ckpt['model_state_dict'])

# model.load_from_checkpoint()

# collate_fn = SwaVCollateFunction(gaussian_blur = 0, hf_prob = 0,vf_prob = 0,rr_prob=0,cj_prob=0,random_gray_scale=0, normalize={'mean':[0, 0, 0], 'std':[1, 1, 1]})
#
# dataloader = torch.utils.data.DataLoader(
#     unlabeled_dataset,
#     batch_size=256,
#     collate_fn=collate_fn,
#     shuffle=True,
#     drop_last=True,
#     num_workers=4
# )
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

checkpoint_callback = pl_callbacks.ModelCheckpoint(monitor='swav_loss', filename='rns_swav-{epoch:02d}-{swav_loss:.5f}',
                                                   save_last=True, save_top_k=-1,
                                                   dirpath=ckpt_folder_root + 'rns_swav_50_all')
csv_logger = pl_loggers.CSVLogger(log_folder_root, name="rns_swav_50_all")

trainer = pl.Trainer(logger=csv_logger, max_epochs=150, callbacks=[checkpoint_callback], accelerator='gpu', devices=1,
                     precision=16)

# In[ ]:


trainer.fit(model=model, train_dataloaders=dataloader,ckpt_path=ckpt_folder_root + 'rns_swav_50_all/rns_swav-epoch=13-swav_loss=3.00627.ckpt')
