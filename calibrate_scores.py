# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from time import ctime

import torch as th
from sacred import Experiment

from calibrator import Calibrator
from dataset import NegDataset
from tester import Tester


ex = Experiment('score_calibration')

@ex.config
def ex_cfg():
    params = {
        'batch_size': 2000,
        'lr': 0.1,
        'n_epochs': 100,
        'load_model': 'sample_model.pt',
        'model': 'RegDistMult',
        'dataset': 'sample_data_path/WN18',
        'data_type': 'valid',
        'emb_dim': 200,
        'activation': 'tanh',
        'k': 5,
        'shift_score': 0,
    }

@ex.automain
def main(params, _log):
    tester = Tester(NegDataset(params['dataset']), params['load_model'], params['data_type'], params)
    scores = tester.find_predictions()
    th.save(
        th.stack((scores.view(-1).data.cpu(), th.FloatTensor(tester.dataset.data[tester.data_type][:, -1]).view(-1)), dim=1),
        tester.res_dir+'%s_scores.pt' % tester.data_type
    )
    _log.info('[%s]: the %s scores are written.' % (ctime(), params['data_type']))

    # The calibration model is trained on the validation dataset and is tested on the test dataset.
    calibrator = Calibrator(params, tester.res_dir+'%s_scores.pt' % tester.data_type)
    if params['data_type'] == 'valid':
        calibrator.train(_log)
        calibrator.save_model()
        _log.info('[%s]: calibration model is learned & saved.' % ctime())
    else:
        calibrator.model.load_state_dict(th.load(calibrator.model_path))
        _log.info('[%s]: calibration model is loaded.' % ctime())
    
    c_scores = calibrator.calibrate_scores()
    _log.info('[%s]: calibrated scores are written.' % ctime())
    llerr = calibrator.logLikelihood(c_scores)
    bscore = calibrator.brierScore(c_scores)
    _log.info('[%s]: calibrated NLLerr = %f, calibrated Bscore = %f' %(ctime(), llerr, bscore))