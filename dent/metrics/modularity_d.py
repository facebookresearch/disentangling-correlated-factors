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
import torch
import sklearn.preprocessing
from pyitlib import discrete_random_variable as drv

from .basemetric import BaseMetric

class Modularityd(BaseMetric):
    @param('modularity.num_bins')
    def __init__(self, num_bins=20, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = num_bins

    @property
    def _requires(self):
        return ['stats_qzx', 'gt_factors']

    @property
    def _mode(self):
        return 'full'

    def __call__(self, stats_qzx, gt_factors, **kwargs):
        """Compute Modularity Score as proposed in [1] (eq.2).

        References
        ----------
           [1] Ridgeway et al. "Learning deep disentangled embeddings with the f-statistic loss", 
           Advances in Neural Information Processing Systems. 2018.
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

        mutual_info_scores = np.zeros((num_factors, num_latents))
        for factor_id in range(num_factors):
            for latent_id in range(num_latents):
                mutual_info_scores[factor_id, latent_id] = drv.information_mutual(mean_qzx[:, latent_id], gt_factors[:, factor_id], cartesian_product=True, base=np.e)

        modularity = 0
        for latent_id in range(num_latents):
            factor_mutual_info_scores = mutual_info_scores[:, latent_id]
            max_mutual_info_idx = np.argmax(factor_mutual_info_scores)

            deviation_from_ideal = 0
            for factor_id, factor_mutual_info in enumerate(factor_mutual_info_scores):
                if factor_id != max_mutual_info_idx:
                    deviation_from_ideal += factor_mutual_info ** 2

            normalization = factor_mutual_info_scores[max_mutual_info_idx] ** 2 * (num_factors - 1)
            modularity += 1 - deviation_from_ideal / normalization
        
        return modularity / num_latents
