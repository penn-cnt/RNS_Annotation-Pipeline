#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import numpy as np
import random
import sys
sys.path.append('../tools')


from sklearn.model_selection import train_test_split
from models.kaggle_dog_dataloader import *
from active_learning_utility import get_strategy
from active_learning_data import Data
from active_learning_net import Net
from models.kaggle_dog_dataloader import ActiveDataHandler
from copy import deepcopy
from models.SwaV import SwaV
from models.SupervisedDownstream import SupervisedDownstream
import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Set a lower value for log_every_n_steps if you want to see logs for the training epoch*")


# In[13]:


data_dir = "../../../user_data/competition_data/clips"
log_folder_root = '../../../user_data/logs/'
ckpt_folder_root = '../../../user_data/checkpoints/'
strategy_name = 'MarginSamplingDropout'

targets = [
    'Dog_1',
    'Dog_2',
    'Dog_3',
    'Dog_4',
]

# set the pipeline to be deterministic
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


# In[14]:


nStart = 3.7
nEnd = 15
nQuery = 1


# In[15]:


args_task = {'n_epoch': 100,
             'transform_train': True,
             'strategy_name': strategy_name,
             'transform': False,
             'loader_tr_args': {'batch_size': 256, 'num_workers': 4, 'collate_fn': collate_fn,
                                'drop_last': True},
             'loader_te_args': {'batch_size': 256, 'num_workers': 4, 'collate_fn': collate_fn,
                                'drop_last': True}
             }


# In[16]:


data, label = load_annotated_data(data_dir, targets)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.30, random_state=random_seed)

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


dataset = Data(X_train, y_train, X_test, y_test, ActiveDataHandler, args_task)


# In[18]:


swav = SwaV().load_from_checkpoint(
    ckpt_folder_root + 'kaggle_dog_swav_34/kaggle_dog_swav-epoch=116-swav_loss=2.73583.ckpt')
model = SupervisedDownstream(swav.backbone)
# initialize model and save the model state
modelstate = deepcopy(model.state_dict())
device = "cuda" if torch.cuda.is_available() else "cpu"
net = Net(model, args_task, device)


# In[19]:


# In[20]:


strategy = get_strategy(strategy_name, dataset, net, None, args_task)


# In[21]:


# initial round of training, round 0
dataset.initialize_labels(NUM_INIT_LB)
strategy.train()


# In[ ]:


for rd in range(1, NUM_ROUND +1):
    print('round ' + str(rd))
    q_idxs = strategy.query(NUM_QUERY)
    strategy.update(q_idxs)
    strategy.net.round = rd
    strategy.net.net.load_state_dict(modelstate)
    strategy.train()


# In[ ]:




