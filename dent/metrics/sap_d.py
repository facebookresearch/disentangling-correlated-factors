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
import numpy as np
import sklearn.preprocessing
from sklearn import svm
import torch

from .basemetric import BaseMetric

class SAPd(BaseMetric):
    @param('sap.num_train')
    @param('sap.num_test')
    @param('sap.num_bins')            
    def __init__(self, num_train=10000, num_test=5000, num_bins=20, **kwargs):
        super().__init__(**kwargs)
        self.num_train = num_train
        self.num_test = num_test
        self.num_bins = num_bins

    @property
    def _requires(self):
        return ['stats_qzx', 'gt_factors']

    @property
    def _mode(self):
        return 'full'

    def __call__(self, stats_qzx, gt_factors, **kwargs):
        """Compute the Separated Attribute Predictability Score from [1].

        References
        ----------
           [1] Kumat et al. "Variational Inference of Disentangled Latent Concepts
           from unlabeled observations", International Conference on Learning Representations, 2018.
        """
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = [x.detach().cpu().numpy() for x in stats_qzx.unbind(-1)]
        mean_qzx = stats_qzx[0]
        if isinstance(gt_factors, torch.Tensor):
            gt_factors = gt_factors.detach().cpu().numpy()
        if len(mean_qzx) < self.num_train + self.num_test:
            raise ValueError(
                f'Number of train- and test-samples {self.num_train}/{self.num_test} exceed total number of samples [{len(stats_qzx[0])}]]')

        total_idcs = list(range(len(mean_qzx)))
        train_idcs = np.random.choice(total_idcs, self.num_train, replace=False)
        test_idcs = list(set(total_idcs) - set(list(train_idcs)))
        test_idcs = np.random.choice(test_idcs, self.num_test, replace=False)

        mean_qzx = sklearn.preprocessing.minmax_scale(mean_qzx)        
        gt_factors = sklearn.preprocessing.minmax_scale(gt_factors)

        bins = np.linspace(0, 1, self.num_bins + 1)
        mean_qzx = np.digitize(mean_qzx, bins[:-1], right=False).astype(int)
        gt_factors = np.digitize(gt_factors, bins[:-1], right=False).astype(int)

        mean_qzx_train = mean_qzx[train_idcs]
        gt_factors_train = gt_factors[train_idcs]
        mean_qzx_test = mean_qzx[test_idcs]
        gt_factors_test = gt_factors[test_idcs]

        num_latents = mean_qzx.shape[-1]
        num_factors = gt_factors.shape[-1]
        scores = np.zeros([num_latents, num_factors])

        for i in range(num_latents):
            for j in range(num_factors):
                mu_i = mean_qzx_train[:, i]
                gt_factor_j = gt_factors_train[:, j] 

                mu_test_i = mean_qzx_test[:, i]
                gt_factor_test_j = gt_factors_test[:, j]
                classifier = svm.LinearSVC(C=0.01, class_weight='balanced')
                classifier.fit(mu_i[:, np.newaxis], gt_factor_j)
                pred = classifier.predict(mu_test_i[:, np.newaxis])
                scores[i, j] = np.mean(pred == gt_factor_test_j)
    
        sorted_scores = np.sort(scores, axis=0)
        return np.mean(sorted_scores[-1, :] - sorted_scores[-2, :])
