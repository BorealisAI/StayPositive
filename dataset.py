# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import random
from typing import List, Dict, Union

import numpy as np
import torch as th

from simple.dataset import Dataset


class CustomDataset(Dataset):
    """
    This class extends `Dataset` to create a custom dataset method.

    Args:
        ds_name (str): Specifies the path to the dataset.
    
    Note:
        The custom dataset is a dict consisting of three keys: train, valid, and test, where 
        each key contains a 2d array of triplets of shape [n x 3], where each row is [head, relation, tail].
        All entities and relations in the dataset are assigned a unique id and are stored
        in separate files (`entities.txt` and `relations.txt`) in the dataset folder.
    """
    def __init__(self, ds_name: str):
        self.name = ds_name.split('/')[-1]
        self.dir = ds_name+'/'
        self.ent2id = self.get_id(self.dir+'entities.txt')
        self.rel2id = self.get_id(self.dir+'relations.txt')
        self.data = {spl: self.read(self.dir + spl + ".txt") for spl in ["train", "valid", "test"]}
        self.batch_index = 0
                        
    def triple2ids(self, triple: List[str]) -> List[int]:
        """
        Converts entities and relation in each triple to their corresponding ids.

        Args:
            triple (array): Specifies the input list of shape [3].
        """
        return [self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]]]
                     
    def get_id(self, ds_path: str) -> Dict:
        """
        Loads the dictionary containing unique ids for entities and relations that are in str format in the dataset.

        Args:
            ds_path (str): Specifies the path to the json file.
        """
        with open(ds_path, 'r') as f:
            ids = json.load(f)
        return ids
            
    def next_batch(self, batch_size: int, neg_ratio: int) -> Dict[str, th.Tensor]:
        """
        Gets the next batch of samples to feed to the network which is a dictionary
        containing data and label keys.

        Args:
            batch_size (int): Specifies the batch size.
            neg_ratio (int): Specifies the negative ratio for sampling negatives.

        Note:
            `neg_ratio` should be zero or a positive number.
        """
        pos_batch = self.next_pos_batch(batch_size)
        if neg_ratio > 0:
            neg_batch = self.generate_neg(pos_batch, neg_ratio)
            batch = np.append(pos_batch, neg_batch, axis=0)
        elif neg_ratio == 0:
            batch = pos_batch
        else:
            raise ValueError('Negative ratio should be a number bigger than or equal to zero!')
        np.random.shuffle(batch)
        sample = {
            'data': th.tensor(batch[:, :-1]).long(),
            'label': th.tensor(batch[:, -1]).float()
        }
        return sample
    
    def shuffle_data(self):
        """
        Shuffles the dataset.
        """
        np.random.shuffle(self.data['train'])


class NegDataset(CustomDataset):
    """
    This class extends `CustomDataset` to support datasets that contain negative labels as well.

    Args:
        ds_name (str): Specifies the path to the dataset.
    """
    def __init__(self, ds_name: str) -> Dict[str, np.ndarray]:
        super(NegDataset, self).__init__(ds_name)
        self.data = {spl: self.read(self.dir + spl + ".txt") if spl=="train" else \
                        self.read_neg(self.dir + spl + ".txt") for spl in ["train", "valid", "test"]}
    
    def read_neg(self, file_path: str) -> np.ndarray:
        """
        Reads the dataset containing negative labels and returns it as a numpy array.

        Args:
            file_path (str): Specifies the path to the dataset containing triples and their corresponding labels.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        data = np.zeros((len(lines), 4)) # the last element is the label
        for i, line in enumerate(lines):
            data[i] = np.array(self.toIds(line.strip().split("\t")))
        return data
    
    def toIds(self, data: List[Union[str, int]]) -> List[int]:
        """
        Converts entities and relation in each triple to their corresponding ids and adds their corresponding
        labels as the last value.

        Args:
            data (str, int): Specifies the input data of shape [4]. The last one is the label.
        """
        return [self.ent2id[data[0]], self.rel2id[data[1]], self.ent2id[data[2]], data[3]]
