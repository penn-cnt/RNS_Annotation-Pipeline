import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
import torch.nn.init as init
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks


class Net:
    def __init__(self, net, params, device, log_folder_root='kaggle_dog_active',
                 ckpt_folder_root='kaggle_dog_active'):
        self.net = net
        self.params = params
        self.device = device
        self.round = 0
        self.trainer = None
        self.log_folder_root = '../../../user_data/logs/' + log_folder_root + '/'
        self.ckpt_folder_root = '../../../user_data/checkpoints/' + ckpt_folder_root + '/'

    def train(self, data, test_data=None, initialize_only = None):
        ckpt_save_n_step = 20

        checkpoint_callback = pl_callbacks.ModelCheckpoint(monitor='train_loss',
                                                           filename=
                                                           self.params['strategy_name'] + '_round_' + str(self.round)
                                                           + '-{step}-{train_loss:.5f}',
                                                           dirpath=self.ckpt_folder_root + 'active_checkpoints_'
                                                                   + self.params['strategy_name'],
                                                           save_top_k=-1,
                                                           every_n_train_steps=ckpt_save_n_step,
                                                           save_on_train_epoch_end=False)

        early_stop_callback = pl_callbacks.EarlyStopping(monitor="val_acc",
                                                         patience=10,
                                                         verbose=False,
                                                         mode="max")

        csv_logger = pl_loggers.CSVLogger(self.log_folder_root + "active_logs_" + self.params['strategy_name'],
                                          name='logger_round_' + str(self.round))

        trainer = pl.Trainer(logger=csv_logger,
                             max_epochs=self.params['n_epoch'],
                             callbacks=[checkpoint_callback, early_stop_callback],
                             accelerator='gpu',
                             devices=1,
                             log_every_n_steps=50,
                             precision=16,
                             check_val_every_n_epoch=None,
                             val_check_interval=ckpt_save_n_step,
                             enable_model_summary=False,
                             )

        self.trainer = trainer
        self.net.to(self.device)

        if not initialize_only:
            if test_data is not None:
                testloader = DataLoader(test_data, shuffle=False, **self.params['loader_te_args'])
                loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])

                self.trainer.fit(self.net, loader, testloader)
            else:
                loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
                self.trainer.fit(self.net, loader)

    def run_prediction(self, data):
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        predictions = self.trainer.predict(self.net, loader)
        return predictions

    def predict(self, data):
        predictions = self.run_prediction(data)
        output_list = []
        for pred, y, emb, emb_t, seq_len in predictions:
            output_list.append(pred)

        pred_raw = torch.vstack(output_list).float()
        out = torch.argmax(pred_raw, dim=1)
        return out

    def predict_prob(self, data):

        predictions = self.run_prediction(data)
        output_list = []
        seq_len_list = []
        m = nn.Softmax(dim=1)
        for pred, y, emb, emb_t, seq_len in predictions:
            output_list.append(pred)
            seq_len_list.append(seq_len)

        pred_raw = torch.vstack(output_list).float()
        seq_len_out = torch.tensor([item for sublist in seq_len_list for item in sublist])
        probs = m(pred_raw)
        return probs, seq_len_out

    def predict_prob_dropout(self, data, n_drop=10):
        self.net.enable_mc_dropout = True
        prediction_list = []
        for _ in range(n_drop):
            prediction_list.append(self.run_prediction(data))

        m = nn.Softmax(dim=1)

        output_list_list = []
        for predictions in prediction_list:
            output_list = []
            seq_len_list = []
            for pred, y, emb, emb_t, seq_len in predictions:
                output_list.append(pred)
                seq_len_list.append(seq_len)
            pred_raw = torch.vstack(output_list).float()
            probs = m(pred_raw)
            output_list_list.append(probs)

        prob_dp = torch.mean(torch.stack(output_list_list), dim=0)
        seq_len_out = torch.tensor([item for sublist in seq_len_list for item in sublist])

        self.net.enable_mc_dropout = False

        return prob_dp, seq_len_out

    def predict_prob_dropout_split(self, data, n_drop=10):
        self.net.enable_mc_dropout = True
        prediction_list = []
        for _ in range(n_drop):
            prediction_list.append(self.run_prediction(data))

        m = nn.Softmax(dim=1)

        output_list_list = []
        for predictions in prediction_list:
            output_list = []
            seq_len_list = []
            for pred, y, emb, emb_t, seq_len in predictions:
                output_list.append(pred)
                seq_len_list.append(seq_len)
            pred_raw = torch.vstack(output_list).float()
            probs = m(pred_raw)
            output_list_list.append(probs)

        prob_dp = torch.stack(output_list_list)
        seq_len_out = torch.tensor([item for sublist in seq_len_list for item in sublist])
        self.net.enable_mc_dropout = False

        return prob_dp, seq_len_out

    def get_model(self):
        return self.net

    def get_embeddings(self, data, return_target=False):
        predictions = self.run_prediction(data)
        emb_list = []
        emb_t_list = []
        target_list = []
        seq_len_list = []
        for pred, y, emb, emb_t, seq_len in predictions:
            emb_list.append(emb)
            emb_t_list.append(emb_t)
            target_list.append(y)
            seq_len_list.append(seq_len)
        emb = torch.vstack(emb_list).float()
        emb_t = torch.vstack(emb_t_list).float()
        target = torch.concat(target_list)
        if return_target:
            return emb, emb_t, target, torch.tensor([item for sublist in seq_len_list for item in sublist])
        else:
            return emb, emb_t, torch.tensor([item for sublist in seq_len_list for item in sublist])

    def get_grad_embeddings(self, data):
        predictions = self.run_prediction(data)
        output_list = []
        emb_list = []
        seq_len_list = []
        m = nn.Softmax(dim=1)
        for pred, y, emb, emb_t, seq_len in predictions:
            output_list.append(pred)
            emb_list.append(emb_t)
            seq_len_list.append(seq_len)
        emb = torch.vstack(emb_list)
        out = emb.data.cpu().numpy()
        pred_raw = torch.vstack(output_list)
        batchProbs = m(pred_raw.float()).data.cpu().numpy()
        maxInds = np.argmax(batchProbs, 1)

        nLab = batchProbs.shape[1]
        embDim = emb.shape[1]
        embeddings = np.zeros([len(emb), embDim * nLab])

        for j in range(len(emb)):
            for c in range(nLab):
                if c == maxInds[j]:
                    embeddings[j][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                            1 - batchProbs[j][c]) * -1.0
                else:
                    embeddings[j][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                            -1 * batchProbs[j][c]) * -1.0

        return embeddings, torch.tensor([item for sublist in seq_len_list for item in sublist])
