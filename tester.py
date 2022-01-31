# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from time import ctime
from os.path import join
from typing import Dict, List, Tuple
from logging import Logger

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

import simple.tester
from simple.dataset import Dataset
from metric import Metric
from model.simple import SPSimplE
from model.distmult import DistMult, SPDistMult


class Tester(simple.tester.Tester):
    """
    This class implements testing and evaluating a model.

    Args:
        dataset (Dataset): Specifies the dataset.
        model_path (str): Specifies the path to the model.
        data_type (str): Specifies the type of the data i.e. 'valid' or 'test'.
        params (dict): Specifies the parameter configuration of the experiment.
    """
    def __init__(self, dataset, model_path, data_type, params):
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.params = params
        self.model = self.init_model().to(self.device)
        self.model.load_state_dict(th.load(model_path))
        self.metric = Metric()
        self.data_type = data_type
        self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())

    def init_model(self) -> nn.Module:
        """
        Initializes the model to use for evaluation based on `params['model']` which is specified by user.
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

    def to_device(self, data: np.ndarray) -> th.Tensor:
        """
        Moves the data to a device (cpu or gpu). If there is an available gpu, the data is moved to gpu
        otherwise, it will stay on cpu.

        Args:
            data (np.ndarray): Specifies the input data.
        """
        return th.tensor(data).long().to(self.device)

    def get_rank(self, sim_scores: np.ndarray) -> float:
        """
        Finds the rank of the prediction between all queries (assuming the first score is the prediction score).

        Args:
            sim_scores (np.ndarray): Specifies the input scores. All scores are ranked against the first score
                which is the prediction score.
        """
        return (sim_scores > sim_scores[0]).sum() + 1.0

    def create_queries(self, fact: np.ndarray, head_or_tail: str) -> List[Tuple[int]]:
        """
        Creates queries of type (?, r, t) and (h, r, ?) for the input parameter, `fact`, 
        which is a triple (h, r, t). 
        It returns all possible queries except the fact itself.

        Args:
            fact (np.ndarray): Specifies the input triple (h, r, t).
            head_or_tail (str): Specifies if the head of the triple should be corrupted or its tail.
        """
        res = []
        for id_ in range(self.dataset.num_ent()):
            if head_or_tail == "head":
                if id_ != fact[0]:
                    res.append(tuple([id_, fact[1], fact[2]]))
            else:
                if id_ != fact[2]:
                    res.append(tuple([fact[0], fact[1], id_]))
        return res

    def add_fact(self, fact: np.ndarray, queries: List[Tuple[int]], raw_or_fil: str) -> np.ndarray:
        """
        Updates the queries with the fact.
        If `raw_or_fil` is 'fil', it will filter out triples that are similar to the created queries
        from train, validation, and test datasets.

        Args:
            fact (np.ndarray): Specifies the input triple (h, r, t).
            queries (list): Specifies the list of all possible queries except the fact itself.
            raw_or_fil (str): Specifies the "raw" or "fil" setting. It is used for getting the MRR of the model.
        """
        if raw_or_fil == "raw":
            result = [tuple(fact)] + queries
        elif raw_or_fil == "fil":
            result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)
        return np.asarray(result)

    def mrr(self, _log) -> float:
        """
        Finds the Mean Recirocal Rank (MRR) based on the assigned rank to each sample.

        Args:
            _log (Logger): Specifies the logger.
        """
        with th.no_grad():
            settings = ["raw", "fil"] if self.data_type == 'test' else ["fil"]
            
            for i in range(len(self.dataset.data[self.data_type])):
                fact = self.dataset.data[self.data_type][i, :].astype('int')
                
                for head_or_tail in ["head", "tail"]:
                    queries = self.create_queries(fact, head_or_tail)
                    
                    for raw_or_fil in settings:
                        combinations = self.add_fact(fact, queries, raw_or_fil) # a nx3 matrix
                        sim_scores = self.model(self.to_device(combinations)).cpu().data.numpy()
                        rank = self.get_rank(sim_scores)
                        self.metric.update(rank, raw_or_fil)

            self.metric.normalize(len(self.dataset.data[self.data_type]))
            self.metric.view(settings, _log)
            return self.metric.mrr['fil']

    def find_predictions(self) -> th.Tensor:
        """
        Finds and returns all predictions for the validation or test set, 
        which then will be used to find log likelihood and brier score.
        """
        with th.no_grad():
            prds = th.zeros(len(self.dataset.data[self.data_type])) # tensor of all scores but on cpu
            bs = self.params['batch_size']
            batch_num = len(self.dataset.data[self.data_type]) // bs + 1
            
            for bi in range(batch_num):
                if (bi+1)*bs > len(self.dataset.data[self.data_type]):
                    prds[bi*bs:] = self.model(
                        self.to_device(self.dataset.data[self.data_type][bi*bs:, :])
                    ).cpu().data
                else:
                    prds[bi*bs:(bi+1)*bs] = self.model(
                        self.to_device(self.dataset.data[self.data_type][bi*bs:(bi+1)*bs, :])
                    ).cpu().data
            
            return prds

    def logLikelihood(self, scores: th.Tensor, _log: Logger) -> float:
        """
        Implements the log likelihood error as a function of the predictions and labels.

        Args:
            scores (th.Tensor): Specifies the predicted scores.
            _log (Logger): Specifies the logger.
        """
        with th.no_grad():
            label = th.FloatTensor(self.dataset.data[self.data_type][:, -1]).to(self.device)
            total_loss = th.sum(F.softplus(-label * scores.to(self.device))).item()/len(self.dataset.data[self.data_type])
            _log.info('Negative LogLikelihood Error: %f' % total_loss)
            return total_loss
    
    def brierScore(self, scores: th.Tensor, _log: Logger) -> float:
        """
        Implements the Brier score to measure the model's calibration.

        Args:
            scores (th.Tensor): Specifies the predicted scores.
            _log (Logger): Specifies the logger.
        """
        with th.no_grad():
            sig = nn.Sigmoid()
            label = th.FloatTensor(self.dataset.data[self.data_type][:, -1]).to(self.device)
            label[label<0] = 0 # converting -1 labels to 0s
            bScore = th.sum((label - sig(scores.to(self.device))) ** 2).item()/len(self.dataset.data[self.data_type])
            _log.info('BrierScore: %f' % bScore)
            return bScore

    def roc_stats(self, scores: th.Tensor) -> Tuple:
        """
        Returns auc, false positive, and true positive rates for predicted scores.
        
        Args:
            scores (th.Tensor): Specifies the predicted scores.
        """
        label = self.dataset.data[self.data_type][:, -1]
        scores = scores.view(-1).numpy()
        fpr, tpr, _ = roc_curve(label, scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def test(self, _log: Logger) -> float:
        """
        Evaluates the model by either getting the loglikelihood and brier score or MRR.

        Args:
            _log (Logger): Specifies the logger.
        """
        self.model.eval()
        if self.params['test_has_neg']:
            scores = self.find_predictions()
            metric = self.logLikelihood(scores, _log)
            self.brierScore(scores, _log)
        else:
            metric = self.mrr(_log)
        return metric

    def report_accuracy(self, _log: Logger) -> Tuple:
        """
        Finds the model's accuracy using roc statistics.

        Args:
            _log (Logger): Specifies the logger.
        """
        self.model.eval()
        scores = self.find_predictions()
        fpr, tpr, roc_auc = self.roc_stats(scores)
        _log.info('AUC: %f' % roc_auc)
        return fpr, tpr, roc_auc
