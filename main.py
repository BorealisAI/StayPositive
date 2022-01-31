# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from time import ctime

from trainer import Trainer
from tester import Tester
from sacred import Experiment
from utils import find_best, save, get_dataset, plot_roc

ex = Experiment('Regularized_Embedding_Model')

@ex.config
def ex_config():
    params = {
        'lr': 0.1,
        'l2_reg_lambda': 0.03, # lambda for L2-norm 
        'model_reg_lambda': 0.0001, # lambda for the proposed regularization in the paper i.e. forcing the sum of all possible scores to be close to some value that is based on user's input or some prior
        'n_e': 1000, # number of epochs
        'emb_dim': 200, # embedding dimension size
        'batch_size': 1415,
        'neg_ratio': 1, # the ratio of negative samples that need to be generated
        'dataset': 'dataset/WN18RR', # the path to the dataset folder
        'save_each': 100, # how often to save the model
        'validate_each': 500, # how often to validate the model; this should be divisible by save_each
        'test_has_neg': False, # whether the test and validation sets include negative labels/samples
        'model': 'DistMult', # type of the model to be used for training i.e. DistMult, SPDistMult, SPSimplE; SPSimplE acts as SimplE if used with k=1 and no activation
        'regularized': True, # a boolean value for using the regularized loss function or not
        'reg_loss_type': 'L1', # the type of regularization loss L1, and L2
        'k': 1, # the score range of the model; the final score of a triplet is between -k & k.
        'activation': 'tanh', # the activation function used in the model i.e. tanh and none
        'shift_score': 0, # how much the final score of the model should be shifted i.e. score += params['shift_score'],
        'spRegType': 'batch', #spRegType can be all or batch
        'dropout': 0, #dropout ratio for model scores
    }

@ex.automain
def main(params, _log):
    dataset = get_dataset(params, _log)
    trainer = Trainer(dataset, params)
    trainer.train(_log)

    best_epoch = find_best(dataset, trainer.model_dir, params, _log)
    tester = Tester(dataset, trainer.model_dir+'%d.pt' %best_epoch, 'test', params)
    exp_dir = trainer.writer.file_writer.get_logdir()

    if not params['test_has_neg']:
        tester.mrr(_log)
    else:
        tester.test(_log)
        metrics = tester.report_accuracy(_log)
        plot_roc(metrics, exp_dir)
    
    save(params, exp_dir, _log)
