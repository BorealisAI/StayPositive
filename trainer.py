# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import timeit
from time import ctime
from logging import Logger
from typing import Dict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from simple.dataset import Dataset
from model.simple import SPSimplE
from model.distmult import DistMult, SPDistMult
from regularizer import Regularizer


class Trainer:
    """
    This class implements the training pipeline of a `nn.Module` model.

    Args:
        dataset (Dataset): Specifies the dataset.
        params (Dict): Specifies the parameter configuration of the experiment.

    init_model: finds the model to use for training based on params['model'] which is specified by the user.
    to_device: moves the data to the device (cpu or gpu).
    create_dir: creates a directory for saving the models and possible calibration results when the trainer is called.
    get_loss: defines the loss funtion to use based on params['regularized']. 
    train: the main function to train the model.
    """
    def __init__(self, dataset: Dataset, params: Dict):
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.params = params
        self.writer = SummaryWriter()
        self.model = self.init_model().to(self.device)
        self.model_dir = self.create_dir()[0]
        self.regularizer = Regularizer(self.model, self.dataset, self.params)
        self.optimizer = th.optim.Adagrad(
            self.model.parameters(),
            lr=self.params['lr'],
            weight_decay=0,
            initial_accumulator_value=0.1
        )
        
    def init_model(self) -> nn.Module:
        """
        Initializes the model to use for training based on `params['model']` which is specified by user.
        """
        if self.params['model'] == 'DistMult':
            model = DistMult(self.dataset.num_ent(), self.dataset.num_rel(), self.params)
        elif self.params['model'] == 'spDistMult':
            model = SPDistMult(self.dataset.num_ent(), self.dataset.num_rel(), self.params)
        elif self.params['model'] == 'spSimplE':
            model = SPSimplE(self.dataset.num_ent(), self.dataset.num_rel(), self.params)
        else:
            raise ValueError
        return model

    def to_device(self, sample: np.ndarray) -> th.Tensor:
        """
        Moves the data to a device (cpu or gpu). If there is an available gpu, the data is moved to gpu
        otherwise, it will stay on cpu.

        Args:
            data (np.ndarray): Specifies the input data.
        """
        sample['data'] = sample['data'].to(self.device)
        sample['label'] = sample['label'].to(self.device)
        return sample

    def create_dir(self) -> str:
        """
        Creates a directory for saving the models and possible calibration results when the trainer is called.
        """
        dir = self.writer.file_writer.get_logdir()
        model_dir = dir+'/model/'
        res_dir = dir+'/result/'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
        return model_dir, res_dir

    def get_loss(self, batch_data: th.Tensor, batch_labels: th.Tensor, score: th.Tensor) -> th.Tensor:
        """
        Defines the loss funtion to use for training based on `params['regularized']`. 

        Args:
            batch_data (th.Tensor): Specifies the input batch data.
            batch_labels (th.Tensor): Specifies the corresponding labels for the input batch data.
            score (th.Tensor): Specifies the predicted scores.
        """
        bnum = self.dataset.num_batch(self.params['batch_size'])
        loss = th.sum(F.softplus(-batch_labels * score))
        if self.params['regularized']:
            loss += self.params['model_reg_lambda'] * self.regularizer.regularizer_loss(batch_data)/bnum 
        else:
            loss += self.params['l2_reg_lambda'] * self.model.l2_loss()/bnum
        return loss

    def train(self, _log: Logger):
        """
        Implements the main function to train the model.

        Args:
            _log (Logger): Specifies the logger.
        """
        _log.info('[%s]: Training ...' %ctime())
        self.model.train()
        
        avg_epoch_time = 0
        for epoch in range(1, self.params['n_e'] + 1):
            start_epoch = timeit.default_timer()
            total_loss, last_batch= 0.0, False
            self.dataset.shuffle_data()
            
            while not last_batch:
                sample_batch = self.to_device(
                    self.dataset.next_batch(self.params['batch_size'], self.params['neg_ratio'])
                )
                last_batch = self.dataset.was_last_batch()
                self.optimizer.zero_grad()
                score = self.model(sample_batch['data'])
                loss = self.get_loss(sample_batch['data'], sample_batch['label'], score)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.cpu().item()
            
            _log.info('[%s]: The training loss at [%d/%d] is %f' %(ctime(), epoch, self.params['n_e'], total_loss))

            if epoch % self.params['save_each'] == 0:
                th.save(self.model.cpu().state_dict(), self.model_dir+'%d.pt' % epoch)
                self.model = self.model.to(self.device)
            avg_epoch_time += (timeit.default_timer()-start_epoch)
        
        _log.info('[%s]: Average epoch time is %f.' %(ctime(), avg_epoch_time/self.params['n_e']))
        th.save(self.model.cpu().state_dict(), self.model_dir+'%d.pt' % self.params['n_e'])
        _log.info('[%s]: The model is trained and saved after %d epochs' %(ctime(), self.params['n_e']))
