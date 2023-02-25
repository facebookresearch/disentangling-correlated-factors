"""
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Copyright 2018 The DisentanglementLib Authors.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

ORIGINAL CODE WAS CHANGED AS FOLLOWS:
- Conversion from Tensorflow to PyTorch.
- Integration as a mergable BaseMetric that can be combined with multiple other metrics for efficient computation.
- Efficiency improvements through parallelization.
- Function and variable renaming.
"""
from fastargs.decorators import param
from joblib import Parallel, delayed
import numpy as np
import scipy
import scipy.stats
import sklearn.metrics
import sklearn.preprocessing
import torch
from tqdm import trange

from .basemetric import BaseMetric

class DCId(BaseMetric):
    @param('dci.num_train')
    @param('dci.num_test')
    @param('dci.backend')
    @param('data.num_workers')
    def __init__(self, num_train, num_test, backend='sklearn', num_workers=8, **kwargs):
        super().__init__(**kwargs)
        self.num_train = num_train
        self.num_test = num_test
        if backend == 'sklearn':
            import sklearn.ensemble
            self.prediction_model = sklearn.ensemble.GradientBoostingClassifier
        elif backend == 'sklearn_forest':
            import sklearn.ensemble
            self.prediction_model = sklearn.ensemble.RandomForestClassifier(n_jobs=num_workers)
        else:
            import xgboost
            self.prediction_model = xgboost.XGBClassifier
        self.num_workers = num_workers

    @property
    def _requires(self):
        return ['stats_qzx', 'gt_factors']

    @property
    def _mode(self):
        return 'full'

    def __call__(self, stats_qzx, gt_factors, **kwargs):
        """Compute Disentanglement, Completeness and Informativness [1].

        References
        ----------
           [1] Eastwood et al. "A Framework for the Quantitative Evaluation of Disentangled
           Representations", International Conferences on Learning Representations, 2018.
        """
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = [x.detach().cpu().numpy() for x in stats_qzx.unbind(-1)]
        mean_qzx = stats_qzx[0]
        if isinstance(gt_factors, torch.Tensor):
            gt_factors = gt_factors.detach().cpu().numpy()
        if len(mean_qzx) < self.num_train + self.num_test:
            raise ValueError(
                f'Number of train- and test-samples {self.num_train}/{self.num_test} exceed total number of samples [{len(mean_qzx)}]]')
        
        total_idcs = list(range(len(mean_qzx)))
        train_idcs = np.random.choice(total_idcs, self.num_train, replace=False)
        test_idcs = list(set(total_idcs) - set(list(train_idcs)))
        test_idcs = np.random.choice(test_idcs, self.num_test, replace=False)

        gt_factors = sklearn.preprocessing.minmax_scale(gt_factors)
        mean_qzx = sklearn.preprocessing.minmax_scale(mean_qzx)

        for i in range(gt_factors.shape[-1]):
            uv = np.unique(gt_factors[:, i])
            dc = {val: i for i, val in enumerate(uv)}
            def dmap(val):
                return dc[val]
            out = list(map(dmap, gt_factors[:, i]))
            gt_factors[:, i] = np.array(out)
        gt_factors = gt_factors.astype(int)

        mean_qzx_train = mean_qzx[train_idcs]
        gt_factors_train = gt_factors[train_idcs]
        mean_qzx_test = mean_qzx[test_idcs]
        gt_factors_test = gt_factors[test_idcs]  

        num_latents = mean_qzx.shape[-1]
        num_factors = gt_factors.shape[-1]
        importance_scores = np.zeros([num_latents, num_factors])

        def get_importance(factor_id):
            pred_model = self.prediction_model()
            pred_model.fit(mean_qzx_train, gt_factors_train[:, factor_id])
            importance_scores = np.abs(pred_model.feature_importances_)
            train_preds = pred_model.predict(mean_qzx_train)
            test_preds = pred_model.predict(mean_qzx_test)
            train_score = np.mean(train_preds == gt_factors_train[:, factor_id])
            test_score = np.mean(test_preds == gt_factors_test[:, factor_id])
            train_err = 1 - train_score
            test_err = 1 - test_score
            return [factor_id, importance_scores, train_score, test_score, train_err, test_err]

        res = Parallel(n_jobs=self.num_workers)(delayed(get_importance)(factor_id) for factor_id in trange(num_factors, leave=False))

        importance_scores = np.zeros([num_latents, num_factors])
        informativeness_train_scores = []
        informativeness_test_scores = []
        informativeness_train_errors = []
        informativeness_test_errors = []
        for factor_id, importance_score_factor, train_score, test_score, train_err, test_err in res:
            importance_scores[:, factor_id] = importance_score_factor
            informativeness_train_scores.append(train_score)
            informativeness_test_scores.append(test_score)
            informativeness_train_errors.append(train_err)
            informativeness_test_errors.append(test_err)
        informativeness_train_scores = np.mean(informativeness_train_scores)
        informativeness_test_scores = np.mean(informativeness_test_scores)
        informativeness_train_errors = np.mean(informativeness_train_errors)
        informativeness_test_errors = np.mean(informativeness_test_errors)

        per_latent_disentanglement = 1. - scipy.stats.entropy(importance_scores.T + 1e-11, base=importance_scores.shape[1])
        per_factor_completeness = 1. - scipy.stats.entropy(importance_scores + 1e-11, base=importance_scores.shape[0])        
        if importance_scores.sum() == 0.:
            importance_scores = np.ones_like(importance_scores)
        total_latent_disentanglement = importance_scores.sum(axis=1) / importance_scores.sum()
        total_factor_completeness = importance_scores.sum(axis=0) / importance_scores.sum()
        disentanglement = np.sum(per_latent_disentanglement * total_latent_disentanglement)
        completeness = np.sum(per_factor_completeness * total_factor_completeness)

        return {
            'informativeness_train_errors': informativeness_train_errors,
            'informativeness_test_errors': informativeness_test_errors,
            'informativeness_train_scores': informativeness_train_scores,
            'informativeness_test_scores': informativeness_test_scores,
            'disentanglement': disentanglement,
            'completeness': completeness
        }
