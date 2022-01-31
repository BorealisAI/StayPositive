# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Tuple
from time import ctime
from logging import Logger
import os

import torch as th
import torch.nn as nn
import numpy as np


class Platt(nn.Module):
    """
    This class implements platt scaling proposed in:
    https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods.
    
    Note:
        To train the calibration model, each score `s` is mapped to `a*s+b` which gives the calibrated score.
        Therefore, the model only needs to learn two parameters (`a` and `b`).
        The loss function to train is usually log likelihood loss.
    """
    def __init__(self) -> nn.Module:
        super(Platt, self).__init__()
        self.net = nn.Linear(1,1)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Implements the forward pass of the network.

        Args:
            x (th.Tensor): Specifies the input uncalibrated scores.
        """
        return self.net(x)

    def l2_loss(self) -> th.Tensor:
        """
        Implement the squared l2 loss of the network parameters.

        Note:
            Since the network only contains one weight, this is equal to the squared absolute value.
        """
        return th.abs(self.net.weight)**2


class Calibrator:
    """
    This class implements the post processing step for calibrating the predictions of a previously trained model.
    It does not change the model parameters and only trains a calibrator model to post calibrate predictions.
    It is only used for datasets that contain negative labels as well as positive labels (e.g. WN11 & FB13).
    Calibration model, created datasets (predictions and targets), and 
        the calibrated results are written to the result folder of each experiment.

    Args:
        params (dict): Specifies the parameter dictionary that contains the configuration of the experiment.
        score_path (str): the path to the 2d array consisting of predicted scores and labels.
            First col includes the written predicted scores of a pre-trained model and
            the second col includes the corresponding labels/ground truths (-1 or 1).
    
    Note:
        In order to train the calibration model, the labels should be converted to their expected probabilities
        instead of their binary values (1 or -1).
    """
    def __init__(self, params: Dict, score_path: str):
        self.params =  params
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.score_path = score_path
        self.targets = self.find_target()
        self.dataset = self.update_dataset(th.load(self.score_path)).to(self.device) # a nx2 tensor
        self.model = Platt().to(self.device)
        self.model_path = '/'.join(score_path.split('/')[:-1])+'/calibration_model.pt'
        self.b_idx = 0
        self.bnum = self.batch_num(self.params['batch_size'])

    def find_target(self) -> Tuple[float, float]:
        """
        Finds the number of positive and negative labels in the predicted scores and 
        returns the target values that are used instead of 1 and -1 for calibration.
        """
        scores = th.load(self.score_path)
        n_plus = th.sum(scores[:, 1] > 0).float() # how many positives do we have in the dataset
        n_minus = th.sum(scores[:, 1] < 0).float()
        return (n_plus+1)/(n_plus+2), 1/(n_minus+2)

    def update_dataset(self, scores: th.Tensor) -> th.Tensor:
        """
        Updates the dataset in order to train the calibration model.
        
        Args:
            scores (th.Tensor): Specifies the predicted scores and their original corresponding binary labels.
                It is a 2d tensor of shape [n x 2].
                The first column corresponds to predictions and the second column corresponds to 
                the expected probability of the binary labels that is used as the target value.
        """
        t_plus, t_minus = self.targets
        scores[scores[:, 1]>0, 1] = t_plus
        scores[scores[:, 1]<0, 1] = t_minus
        return scores
    
    def batch_num(self, bs: int) -> int:
        """
        Returns the number of batches in the dataset.

        Args:
            bs (int): Specifies the batch size.
        """
        bn = self.dataset.shape[0]//bs + 1 if self.dataset.shape[0] % bs != 0 else self.dataset.shape[0]//bs
        return bn

    def next_batch(self, bs: int) -> th.Tensor:
        """
        Returns the next batch of the data that is passed to the network.

        Args:
            bs (int): Specifies the batch size.
        """
        if self.b_idx < self.bnum:
            n_batch = self.dataset[self.b_idx*bs:(self.b_idx+1)*bs, :]
            self.b_idx += 1
        else:
            n_batch = self.dataset[self.b_idx*bs:, :]
        return n_batch

    def shuffle(self):
        """
        Shuffles the dataset randomly.
        """
        indices = th.randperm(self.dataset.shape[0])
        self.dataset = self.dataset[indices, :]

    def train(self, _log: Logger):
        """
        Implements the training method of the calibrator.

        Args:
            _log (Logger): Specifies the logger that is used to log the results.
        """
        self.model.train()
        
        optimizer = th.optim.Adagrad(self.model.parameters(), lr=self.params['lr'])
        criterion = th.nn.BCEWithLogitsLoss()
        
        for epoch in range(self.params['n_epochs']):
            self.b_idx, total_loss = 0, 0
            self.shuffle()
            
            for _ in range(self.bnum):
                optimizer.zero_grad()
                sample = self.next_batch(self.params['batch_size'])
                calibrated = self.model(sample[:, 0].view(-1,1))
                loss = criterion(calibrated.view(-1), sample[:, 1])
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            _log.info('[%s]: loss at [%d/%d] is %f.' % (ctime(), epoch+1, self.params['n_epochs'], total_loss))
    
    def logLikelihood(self, prds: th.Tensor) -> float:
        """
        Calculates the log likelihood error on the predictions.

        Args:
            prds (th.Tensor): Specifies the input probabilities.
        """
        with th.no_grad():
            criterion = th.nn.BCELoss()
            label = th.load(self.score_path)[:, 1].to(self.device) # -1,1 labels
            label[label<0] = 0
            loss = criterion(prds.to(self.device), label)
            return loss.item()
    
    def brierScore(self, prds: th.Tensor) -> float:
        """
        Calculates the Brier score on the predictions.

        Args:
            prds (th.Tensor): Specifies the input probabilities.
        """
        with th.no_grad():
            label = th.load(self.score_path)[:, 1].to(self.device)
            label[label<0] = 0 # converting -1 labels to 0s
            bScore = th.sum((label - prds.to(self.device)) ** 2).item()/self.dataset.shape[0]
            return bScore

    def calibrate_scores(self) -> th.Tensor:
        """
        Returns the final calibrated scores and writes them to the result directory of the experiment.
        """
        with th.no_grad():
            sig = th.nn.Sigmoid()
            calibrated = sig(self.model(self.dataset[:, 0].view(-1,1))).view(-1).data.cpu()
            data_type = self.score_path.split('/')[-1].split('.')[0]
            path = '/'.join(self.score_path.split('/')[:-1])
            th.save(calibrated, os.path.join(path, data_type+'_calibrated.pt'))
            return calibrated
    
    def save_model(self):
        """
        Saves the trained calibration model in the specified `model_path` directory.
        """
        with th.no_grad():
            th.save(self.model.cpu().state_dict(), self.model_path)
            self.model = self.model.to(self.device)