
# Stay Positive: Knowledge Graph Embedding Without Negative Sampling

This code is the implementation of the paper: "Stay Positive: Knowledge Graph Embedding Without Negative Sampling". It contains the implementation of two baseline models (i.e. SimplE and DistMult) with their regularized version as described in the paper. 
The datasets folder with all the used datasets is also provided.
The link to the paper can be found [here](https://arxiv.org/abs/2201.02661) (published at ICML 2020 Graph Representation Learning and Beyond workshop).

## Dependencies

* `python` version 3.7.3
* `numpy` version 1.16.2
* `pytorch` version 1.1.0
* `scikit-learn` version 0.21.3
* `seaborn` version 0.10.0
* `matplotlib` version 3.1.2
* `sacred` version 0.8.1
	* `sacred` is only used to handle logging but the code supports creating experiments and plotting training curves.
* `tensorboard` version 2.1.0
	* `tensorboard` is used for saving the models that are trained and also storing the predictions. Each time the code is run, a unique folder will be created under `runs/` for that experiment. The name of the folder is the same as the time when the code was first run. A `model/` and `result/` folder is created for storing the models and the predictions respectively.

## Run the experiments

To reproduce the results for DistMult on WN11, run:

* `python main.py with 'params.n_e=5000' 'params.batch_size=1126' 'params.dataset="dataset/WN11"' 'params.save_each=250' 'params.validate_each=250' 'params.test_has_neg=True' 'params.model="DistMult"' 'params.shift_score=0' 'params.activation="none"' 'params.emb_dim=200' 'params.k=1'  'params.regularized=False' 'params.lr=0.1' 'params.neg_ratio=1' 'params.l2_reg_lambda=0.1' 'params.dropout=0.4'`

To reproduce the results for spDistMult on WN11, run:

* `python main.py with 'params.n_e=5000' 'params.batch_size=1126' 'params.dataset="dataset/WN11"' 'params.save_each=250' 'params.validate_each=250' 'params.test_has_neg=True' 'params.model="spDistMult"' 'params.shift_score=0' 'params.activation="tanh"' 'params.emb_dim=200' 'params.reg_loss_type="L1"' 'params.k=5' 'params.model_reg_lambda=0.1' 'params.regularized=True' 'params.lr=0.1' 'params.neg_ratio=0' 'params.spRegType='batch' 'params.dropout=0.4'`

* To reproduce results for SimplE, use `spSimplE` as the model with `'params.k=1' 'params.activation=none' 'params.shift_score=0'`. These parameters make `spSimplE` act as the baseline model.

### Calibration

We can also calibrate our models after training. We used Platt scaling for this purpose. Simply load the model and get the calibrated results which are saved in the result directory of the experiment from which you load the model. Run:

`python calibrate_scores.py with 'params.dataset=<path_to_data>' 'params.model=<model>' 'params.load_model=<path_to_model>' 'params.k=<score_range>' 'params.activation=<nonlinear>' 'params.shift_score=<shift>' 'params.emb_dim=<emb_dim>' 'params.data_type=<valid or test>' 'params.batch_size=<batch_size>' 'params.n_epochs=<n_epochs>' 'params.lr=<lr>'`

We train the calibration model on the validation set and then test it on the test set. For training, use `valid` as data_type and for testing use `test` as data_type. The code writes uncalibrated predictions along with the 
calibration model and calibrated results in `runs/<name of the experiment>/result/`, where `<name of the experiment>` is equal to the date and time of the run (e.g. if the code is run on Dec 03 at 15:12:13, the name
of the experiment is `Dec03_15-12-13_<some info about machine>`).

### Arguments

* `lr`: learning rate
* `l2_reg_lambda`: L2 regularization lambda 
* `model_reg_lambda`: regularization lambda used for training spDistMult and spSimplE in the proposed framework
* `n_e`: number of epochs to train the model
* `emb_dim`: embedding size
* `batch_size`: batch size
* `neg_ratio`: how many negatives to generate per positive sample
* `dataset`: the path to the dataset that we want to train the model on
* `save_each`: how many epochs to use for saving the model being trained
* `validate_each`: how often (how many epochs) to validate the model, used for early stopping
* `test_has_neg`: a boolean flag to show if the dataset contains negative examples or not
* `model`: the type of the model (i.e. SimplE, SPSimplE, etc.)
* `regularized`: a boolean flag indicating if we want to use the regularization framework or not
* `reg_loss_type`: the type of the loss function to use for regularizing the model in the proposed framework (e.g. L1, L2)
* `k`: the score range of the model (i.e. `-k < score < k`)
* `activation`: the activation function to use after the embeddings (default is tanh)
* `shift_score`: how much we want to shift the scores produced by the model
* `spRegType`: shows the regularization type i.e. 'all' or 'batch'
* `dropout`: dropout ratio for model score (default is 0)


## Cite this work

If you use this package for published work, please cite the following paper:

```
@misc{hajimoradlou2022stay,
      title={Stay Positive: Knowledge Graph Embedding Without Negative Sampling}, 
      author={Ainaz Hajimoradlou and Mehran Kazemi},
      year={2022},
      eprint={2201.02661},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact

* Ainaz Hajimoradlou
* ainaz.hajimoradlou@borealisai.com

## License

Copyright (c) 2020-present, Royal Bank of Canada. All rights reserved. This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
