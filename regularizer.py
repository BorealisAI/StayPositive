# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Dict, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from simple.dataset import Dataset


class Regularizer:
    """
    This class implements the regularizer term for training the knowledge graph without negative sampling.

    Args:
        model (nn.Module): Specifies the model.
        dataset (Dataset): Specifies the dataset.
        params (Dict): Specifies the parameter configuration of the experiment.
    """
    def __init__(self, model: nn.Module, dataset: Dataset, params: Dict):
        self.params = params
        self.model = model
    
    def uniq_ents_rels(self, batch: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns the unique entity and relations in a batch of data.

        Args:
            batch (th.Tensor): Specifies the input batch of data.
        """
        uniq_ents = th.unique(th.cat((batch[:, 2], batch[:, 0]), dim=0), sorted=False)
        uniq_rels = th.unique(batch[:, 1], sorted=False)
        return uniq_ents, uniq_rels

    def spDistMult(self) -> th.Tensor:
        """
        Finds the sum of all possible scores for DistMult and spDistMult models.
        """
        entity_mat = self.model.apply_activation(self.model.ent_embs.weight)
        relation_mat = self.model.apply_activation(self.model.rel_embs.weight)
        
        scores = th.sum(entity_mat, dim=0) ** 2 * th.sum(relation_mat, dim=0)
        score_sum = self.model.normalize(th.sum(scores)) * self.params['k']
        
        return score_sum

    def spDistMultBatch(self, batch: th.Tensor) -> th.Tensor:
        """
        Finds the sum of all possible scores with entities and relations that are in the input batch 
        for DistMult and spDistMult models.

        Args:
            batch (th.Tensor): Specifies the input batch of data.
        """
        uniq_ents, uniq_rels = self.uniq_ents_rels(batch)

        entity_mat   = self.model.apply_activation(self.model.ent_embs(uniq_ents))
        relation_mat = self.model.apply_activation(self.model.rel_embs(uniq_rels))

        scores = th.sum(entity_mat, dim=0) ** 2 * th.sum(relation_mat, dim=0)
        score_sum = self.model.normalize(th.sum(scores)) * self.params['k']
        
        return score_sum

    def spSimplE(self) -> th.Tensor:
        """
        Finds the sum of all possible scores for SimplE and spSimplE models.
        """
        ent_h = self.model.apply_activation(self.model.ent_h_embs.weight)
        ent_t = self.model.apply_activation(self.model.ent_t_embs.weight)
        rel = self.model.apply_activation(self.model.rel_embs.weight)
        rel_inverse = self.model.apply_activation(self.model.rel_inv_embs.weight)
        
        scores_1 = th.sum(rel, dim=0) * th.sum(ent_h, dim=0) * th.sum(ent_t, dim=0)
        scores_2 = th.sum(rel_inverse, dim=0) * th.sum(ent_h, dim=0) * th.sum(ent_t, dim=0)
        
        score_sum = 1/2 * (th.sum(scores_1) + th.sum(scores_2))
        score_sum = self.model.normalize(score_sum) * self.params['k']
        
        return score_sum

    def spSimplEBatch(self, batch: th.Tensor) -> th.Tensor:
        """
        Finds the sum of all possible scores with entities and relations that are in the input batch 
        for SimplE and spSimplE models.

        Args:
            batch (th.Tensor): Specifies the input batch of data.
        """
        uniq_ents, uniq_rels = self.uniq_ents_rels(batch)

        ent_h = self.model.apply_activation(self.model.ent_h_embs(uniq_ents))
        ent_t = self.model.apply_activation(self.model.ent_t_embs(uniq_ents))

        rel = self.model.apply_activation(self.model.rel_embs(uniq_rels))
        rel_inverse = self.model.apply_activation(self.model.rel_inv_embs(uniq_rels))
        
        scores_1 = th.sum(rel, dim=0) * th.sum(ent_h, dim=0) * th.sum(ent_t, dim=0)
        scores_2 = th.sum(rel_inverse, dim=0) * th.sum(ent_h, dim=0) * th.sum(ent_t, dim=0)
        
        score_sum = 1/2 * (th.sum(scores_1) + th.sum(scores_2))
        score_sum = self.model.normalize(score_sum) * self.params['k']
        
        return score_sum

    def model_sum(self, batch: th.Tensor) -> th.Tensor:
        """
        Finds the sum of all possible scores either in a batch or in the whole graph.

        Args:
            batch (th.Tensor): Specifies the input batch of data.
        """
        if self.params['model'] == 'spDistMult':
            if self.params['spRegType'] == 'all':
                kg_sum = self.spDistMult()
            elif self.params['spRegType'] == 'batch':
                kg_sum = self.spDistMultBatch(batch)
        
        elif self.params['model'] == 'spSimplE':
            if self.params['spRegType'] == 'all':
                kg_sum = self.spSimplE()
            elif self.params['spRegType'] == 'batch':
                kg_sum = self.spSimplEBatch(batch)
        
        else:
            raise ValueError
        
        return kg_sum

    def regularizer_loss(self, batch: th.Tensor) -> th.Tensor:
        """
        Implements the regularization loss.

        Args:
            batch (th.Tensor): Specifies the input batch of data.
        """
        batch_sum = self.model_sum(batch)
        
        if self.params['reg_loss_type'] == 'L1':
            loss = th.abs(batch_sum)
        elif self.params['reg_loss_type'] == 'L2':
            loss = batch_sum ** 2
        else:
            raise NotImplementedError
       
        return loss
