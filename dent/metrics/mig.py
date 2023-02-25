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
from pyitlib import discrete_random_variable as drv
import sklearn.preprocessing
import sklearn.metrics
import torch
from tqdm import trange

from .basemetric import BaseMetric

class MIG(BaseMetric):
    @param('mig.num_bins')
    @param('data.num_workers')
    def __init__(self, num_bins=20, num_workers=8, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.num_workers = num_workers

    @property   
    def _requires(self):
        return ['stats_qzx', 'gt_factors']

    @property
    def _mode(self):
        return 'full'

    def __call__(self, stats_qzx, gt_factors, **kwargs):
        """Compute the mutual information gap as in [1].

        References
        ----------
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = [x.detach().cpu().numpy() for x in stats_qzx.unbind(-1)]
        mean_qzx = stats_qzx[0]
        if isinstance(gt_factors, torch.Tensor):
            gt_factors = gt_factors.detach().cpu().numpy()

        num_latents = mean_qzx.shape[-1]
        num_factors = gt_factors.shape[-1]

        mean_qzx = sklearn.preprocessing.minmax_scale(mean_qzx)
        gt_factors = sklearn.preprocessing.minmax_scale(gt_factors)

        bins = np.linspace(0, 1, self.num_bins + 1)
        mean_qzx = np.digitize(mean_qzx, bins[:-1], right=False).astype(int)        
        gt_factors = np.digitize(gt_factors, bins[:-1], right=False).astype(int)

        def compute_mutual_info(mean_qzx, gt_factors):
            factor_mutual_info_scores = []
            for latent_id in range(num_latents):
                factor_mutual_info_scores.append(
                    drv.information_mutual(
                        mean_qzx[:, latent_id], gt_factors, cartesian_product=True, base=np.e))
            sorted_factor_mutual_info_scores = sorted(factor_mutual_info_scores)
            mutual_info_gap = sorted_factor_mutual_info_scores[-1] - sorted_factor_mutual_info_scores[-2]
            factor_entropy = drv.entropy(gt_factors, base=np.e)
            normalized_mutual_info_gap = 1. / factor_entropy * mutual_info_gap
            return normalized_mutual_info_gap

        mig = Parallel(n_jobs=self.num_workers)(delayed(compute_mutual_info)(mean_qzx, gt_factors[:, factor_id]) for factor_id in trange(num_factors, desc='Computing MI for ground truth factors...', leave=False))
        return np.mean(mig)
