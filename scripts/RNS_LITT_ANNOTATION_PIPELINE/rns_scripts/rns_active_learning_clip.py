#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'widget')


# In[2]:


import numpy as np
import random
import sys
sys.path.append('../tools')

import os

import torch

import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
import data_utility, annotation_utility
from models.rns_dataloader import *
from active_learning_utility import get_strategy
from active_learning_data import Data
from active_learning_net import Net
from copy import deepcopy
from models.SwaV import SwaV
from models.SupervisedDownstream import SupervisedDownstream
import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Set a lower value for log_every_n_steps if you want to see logs for the training epoch*")


# In[3]:


random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    # True ensures the algorithm selected by CUFA is deterministic
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)
    # False ensures CUDA select the same algorithm each time the application is run
    torch.backends.cudnn.benchmark = False

import pytorch_lightning
pytorch_lightning.utilities.seed.seed_everything(seed=random_seed, workers=True)


# In[4]:


data_dir = "../../../user_data/"
log_folder_root = '../../../user_data/logs/'
ckpt_folder_root = '../../../user_data/checkpoints/'


# In[5]:


strategy_name = 'LeastConfidence'


# In[6]:


nStart = 1
nEnd = 20
nQuery = 2


# In[7]:


args_task = {'n_epoch': 100,
             'transform_train': True,
             'strategy_name': strategy_name,
             'transform': False,
             'loader_tr_args': {'batch_size': 256, 'num_workers': 2, 'collate_fn': collate_fn,
                                'drop_last': True,},
             'loader_te_args': {'batch_size': 256, 'num_workers': 4, 'collate_fn': collate_fn,
                                'drop_last': True,}
             }


# In[8]:


# raw_annotations = pd.read_csv(data_dir + 'full_updated_anns_annotTbl_cleaned.csv')
# ids = list(np.unique(raw_annotations[raw_annotations['descriptions'].notnull()]['HUP_ID']))
# # ids = list(np.unique(raw_annotations['HUP_ID']))
#
# data_import = data_utility.read_files(path=data_dir+'rns_data', path_data=data_dir+'rns_raw_cache', patientIDs=ids,
#                                       verbose=True)  # Import data with annotation


# In[9]:


# data_list = os.listdir(data_dir+'rns_test_cache')
# print(data_list)
data_list = ['HUP047.npy', 'HUP084.npy', 'HUP096.npy', 'HUP109.npy', 'HUP121.npy', 'HUP129.npy', 'HUP131.npy',
             'HUP137.npy', 'HUP147.npy', 'HUP156.npy', 'HUP159.npy', 'HUP182.npy', 'HUP197.npy', 'HUP199.npy',
             'RNS026.npy', 'RNS029.npy']
X_train, y_train, X_test, y_test, index_train, index_test  = get_data(data_list, split=0.8)
# data, label,_,_ = get_data(data_list, split=1)
# train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(data, label, test_size=0.8, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[10]:


n_pool = len(y_train)
n_test = len(y_test)

NUM_INIT_LB = int(nStart * n_pool / 100)
NUM_QUERY = int(nQuery * n_pool / 100) if nStart != 100 else 0
NUM_ROUND = int((int(nEnd * n_pool / 100) - NUM_INIT_LB) / NUM_QUERY) if nStart != 100 else 0
if NUM_QUERY != 0:
    if (int(nEnd * n_pool / 100) - NUM_INIT_LB) % NUM_QUERY != 0:
        NUM_ROUND += 1

print(NUM_INIT_LB)
print(NUM_QUERY)
print(NUM_ROUND)


# In[17]:


dataset = Data(X_train, y_train, X_test, y_test, RNS_Downstream, args_task)


# In[18]:
import torchvision
from torch import nn
ckpt = torch.load(ckpt_folder_root + "checkpoint31.pth")
# resnet = torchvision.models.resnet50()
# backbone = nn.Sequential(*list(resnet.children())[:-1])
# swav = SwaV(backbone)
# swav.load_state_dict(ckpt['model_state_dict'])

swav = SwaV()
swav.load_state_dict(ckpt['model_state_dict'])
model = SupervisedDownstream(swav.backbone)
# initialize model and save the model state
modelstate = deepcopy(model.state_dict())
device = "cuda" if torch.cuda.is_available() else "cpu"


net = Net(model, args_task, device, ckpt_folder_root = 'rns_active', log_folder_root = 'rns_active')


# In[19]:


modelstate


# In[20]:


strategy = get_strategy(strategy_name, dataset, net, None, args_task)


# In[ ]:


# initial round of training, round 0
dataset.initialize_labels(NUM_INIT_LB)
strategy.train()


# In[16]:


for rd in range(1, NUM_ROUND +1):
    print('round ' + str(rd))
    q_idxs = strategy.query(NUM_QUERY)
    strategy.update(q_idxs)
    strategy.net.round = rd
    strategy.net.net.load_state_dict(modelstate)
    strategy.train()


# In[16]:




