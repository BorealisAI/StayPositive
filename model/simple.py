# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict

import torch as th
import torch.nn as nn
import math

from .distmult import SPDistMult


class SPSimplE(SPDistMult):
    """
    This class implements the regularized version of SimplE (https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf).
    It inherits `apply_activation` and `normalize` functions from SPDistMult.

    Args:
        num_ent (int): Specifies the number of entities in the dataset.         
        num_rel (int): Specifies the number of relations in the dataset.
        params (dict): Specifies the parameter dictionary that contains the configuration of the experiment.        

    Note:
        To get the baseline model, SimplE, set `activation = none`, `k = 1`, and `shift_score = 0` 
        in parameter configuration.
    """
    def __init__(self, num_ent: int, num_rel: int, params: Dict) -> nn.Module:
        super(SPSimplE, self).__init__(num_ent, num_rel, params)
        
        self.ent_h_embs   = nn.Embedding(num_ent, params['emb_dim'])
        self.ent_t_embs   = nn.Embedding(num_ent, params['emb_dim'])
        self.rel_embs     = nn.Embedding(num_rel, params['emb_dim'])
        self.rel_inv_embs = nn.Embedding(num_rel, params['emb_dim'])
        
        sqrt_size = 6.0 / math.sqrt(params['emb_dim'])
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)
        
    def l2_loss(self) -> th.FloatTensor:
        """
        Returns the squared l2 norm of the model parameters.
        """
        return (th.norm(self.ent_h_embs.weight, p=2) ** 2 \
                + th.norm(self.ent_t_embs.weight, p=2) ** 2 \
                + th.norm(self.rel_embs.weight, p=2) ** 2 \
                + th.norm(self.rel_inv_embs.weight, p=2) ** 2
                )/2

    def forward(self, data: th.Tensor) -> th.Tensor:
        """
        Performs the forward pass of the network. 
        
        Args:
            data (th.Tensor): Specifies the input data to the network that is of shape [bs x 1].
                It contains the corresponding ids of the entities and relations in the dataset.
        """
        hh_embs = self.apply_activation(self.ent_h_embs(data[:, 0]))
        ht_embs = self.apply_activation(self.ent_h_embs(data[:, 2]))
        th_embs = self.apply_activation(self.ent_t_embs(data[:, 0]))
        tt_embs = self.apply_activation(self.ent_t_embs(data[:, 2]))
        r_embs = self.apply_activation(self.rel_embs(data[:, 1]))
        r_inv_embs = self.apply_activation(self.rel_inv_embs(data[:, 1]))
        
        scores_1 = self.dropout(hh_embs * r_embs * tt_embs) if self.params['dropout'] > 0 \
            else hh_embs * r_embs * tt_embs
        scores_2 = self.dropout(ht_embs * r_inv_embs * th_embs) if self.params['dropout'] > 0 \
            else ht_embs * r_inv_embs * th_embs
        
        sum_scores_1 = self.normalize(th.sum(scores_1, dim=1)) * self.params['k']
        sum_scores_2 = self.normalize(th.sum(scores_2, dim=1)) * self.params['k']
        
        return (sum_scores_1 + sum_scores_2)/2 + self.params['shift_score']
