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


class DistMult(nn.Module):
    """ 
    This class implements the baseline model DistMult, as proposed in https://arxiv.org/pdf/1412.6575.pdf.
    Scores with absolute values bigger than 20 will be clamped.

    Args:
        num_ent (int): Specifies the number of entities in the dataset. 
        num_rel (int): Specifies the number of relations in the dataset.
        params (dict): Specifies the parameter dictionary that contains the configuration of the experiment.
    """
    def __init__(self, num_ent: int, num_rel: int, params: Dict) -> nn.Module:
        super(DistMult, self).__init__()
        
        self.ent_embs = nn.Embedding(num_ent, params['emb_dim'])
        self.rel_embs = nn.Embedding(num_rel, params['emb_dim'])
        self.dropout = nn.Dropout(p=params['dropout']) if params['dropout']>0 else None
        
        sqrt_size = 6.0 / math.sqrt(params['emb_dim'])
        nn.init.uniform_(self.ent_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        
    def l2_loss(self) -> th.FloatTensor:
        """ 
        Returns the l2 loss of the model parameters.
        """
        return (th.norm(self.ent_embs.weight, p=2) ** 2 + th.norm(self.rel_embs.weight, p=2) ** 2)/2

    def forward(self, data: th.Tensor) -> th.Tensor:
        """ 
        Performs the forward pass of the network. 
        
        Args:
            data (th.Tensor): Specifies the input data to the network that is of shape [bs x 1].
                It contains the corresponding ids of the entities and relations in the dataset.
        """
        head = self.ent_embs(data[:, 0])
        tail = self.ent_embs(data[:, 2])
        relation = self.rel_embs(data[:, 1])
        scores = head * relation * tail if self.dropout is None else self.dropout(head * relation * tail)
        sum_score = th.sum(scores, dim=1)
        return th.clamp(sum_score, -20, 20)


class SPDistMult(DistMult):
    """
    This class implements the regularized version of the DistMult model.

    Args:
        num_ent (int): Specifies the number of entities in the dataset. 
        num_rel (int): Specifies the number of relations in the dataset.
        params (dict): Specifies the parameter dictionary that contains the configuration of the experiment.
    """
    def __init__(self, num_ent: int, num_rel: int, params: Dict) -> nn.Module:
        super(SPDistMult, self).__init__(num_ent, num_rel, params)
        
        self.params = params

    def apply_activation(self, x: th.Tensor) -> th.Tensor:
        """
        Applies an activation function to the feature map based on the activation parameter.
        The function only supports tanh as an activation function. 
        If `none` is specified as the argument, no activation function is applied to the input.

        Args:
            x (th.Tensor): Specifies the feature map.
        """
        if self.params['activation'] == 'tanh':
            return th.tanh(x)
        elif self.params['activation'] == 'none':
            return x
        else:
            raise NotImplementedError
    
    def normalize(self, score: th.Tensor) -> th.Tensor:
        """
        Normalizes the scores to be in range [-1, 1].
        Normalization is done depending on the activation parameter.

        Args:
            score (th.Tensor): Specifies the scores that need to be normalized.
        """
        if self.params['activation'] == 'tanh':
            return score/self.params['emb_dim']
        else:
            return score

    def forward(self, data: th.Tensor) -> th.Tensor:
        """
        Performs the forward pass of the network. 
        
        Args:
            data (th.Tensor): Specifies the input data to the network that is of shape [bs x 1].
                It contains the corresponding ids of the entities and relations in the dataset.
        """
        head = self.apply_activation(self.ent_embs(data[:, 0]))
        tail = self.apply_activation(self.ent_embs(data[:, 2]))
        relation = self.apply_activation(self.rel_embs(data[:, 1]))
        scores = self.dropout(head * relation * tail) if self.params['dropout'] > 0 else head * relation * tail 
        sum_score = self.normalize(th.sum(scores, dim=1)) * self.params['k']
        sum_score += self.params['shift_score']
        return sum_score
