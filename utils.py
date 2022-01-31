# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from time import ctime
from logging import Logger
from typing import Dict, Tuple

import seaborn as sns
import matplotlib.pyplot as plt

from tester import Tester
from dataset import CustomDataset, NegDataset
from simple.dataset import Dataset


def get_dataset(params: Dict, _log: Logger) -> Dataset:
    """
    Returns the dataset to use for the experiment based on the parameters.
    If test set has negatives, the labels of samples have to be returned as well.

    Args:
        params (Dict): Specifies the parameter configuration of the experiment.
        _log (Logger): Specifies the logger.
    """
    _log.info('[%s]: Creating the dataset for training and validation ...' %ctime())
    dataset = CustomDataset(params['dataset']) if not params['test_has_neg'] else NegDataset(params['dataset'])
    return dataset

def find_best(dataset: Dataset, model_path: str, params: Dict, _log: Logger) -> float:
    """
    Implements early stopping to find the best model with the lowest valiation error or higherst validation MRR.
    
    Args:
        dataset (Dataset): Specifies the dataset.
        model_path (str): Specifies the path to the model.
        params (Dict): Specifies the parameter configuration of the experiment.
        _log (Logger): Specifies the logger.
    """
    _log.info('[%s]: Finding the best model on validation set ...' %ctime())
    
    epochs2test = [params['validate_each']*(i + 1) for i in range(params['n_e'] // params['validate_each'])]
    if not params['n_e'] in epochs2test: # this happens only if the params['n_e'] is not divisible by params['validate_each']
        epochs2test.append(params['n_e'])
    _log.info('Epochs to validate: ' + str(epochs2test))
    
    best_mrr, best_err, best_epoch = -1.0, 1e10, 0
    for epoch in epochs2test:
        
        tester = Tester(dataset, model_path+'%d.pt' % epoch, 'valid',params)
        metric_val = tester.test(_log)
        _log.info('[%s]: %s for model at epoch %d is %f' %(ctime(), 'NLL' if params['test_has_neg'] else 'MRR', epoch, metric_val))
        
        if not params['test_has_neg']:
            if metric_val > best_mrr:
                best_mrr = metric_val
                best_epoch = epoch
        else:
            if metric_val < best_err:
                best_err = metric_val
                best_epoch = epoch
    
    _log.info('[%s]: The best epoch with the %s is %d' %(
            ctime(),
            'lowest NLL error' if params['test_has_neg'] else 'highest MRR',
            best_epoch)
    )
    return best_epoch

def plot_roc(metrics: Tuple, save_to: str):
    """
    Creates and saves the roc plot for a given metric(s) at a specified path.

    Args:
        metrics (Tuple): Specifies the metric.
        save_to (str): Specifies the path to save the roc plot.
    """
    fpr, tpr, auc = metrics
    plt.style.use('seaborn-deep')
    sns.set(rc={'figure.figsize': (20, 10)})
    fig, ax = plt.subplots()
    ax.title.set_text('ROC curves on the test set')
    ax.plot(fpr, tpr, marker='.', linestyle='-', linewidth=0.5, label='area = %0.2f' % auc)
    ax.plot([0, 1], [0, 1], marker='.', linestyle='--', linewidth=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    fig.savefig(os.path.join(save_to, 'roc.png'), bbox_inches='tight')
    plt.close(fig)

def save(params: Dict, save_to: str, _log: Logger):
    """
    Saves all parameters of the experiment in a file named `config.txt` at a specified path.
    
    Args:
        params (Dict): Specifies the parameter configuration of the experiment.
        save_to (str): Specifies the path to save the all parameters. 
        _log (Logger): Specifies the logger.
    """
    _log.info('[%s]: saving the config file ...' % ctime())
    with open(save_to+'/config.txt', 'w') as f:
        for pr in params:
            f.write(pr+': %s\n' %str(params[pr]))
