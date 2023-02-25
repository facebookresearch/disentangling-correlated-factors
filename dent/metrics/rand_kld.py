"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from fastargs.decorators import param
import itertools as it
from joblib import Parallel, delayed
import numpy as np
import torch
from tqdm import trange

from .basemetric import BaseMetric

def _kl_normal_loss(mean, logvar):
    latent_dim = mean.size(1)
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    return latent_kl.sum()

class randKLD(BaseMetric):
    @param('kld.batch_size')
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    @property
    def _requires(self):
        return ['stats_qzx']

    @property
    def _mode(self):
        return 'full'

    def __call__(self, stats_qzx, **kwargs):
        """Compute KL-Divergence to unit normal.
        """
        stats_qzx = [x for x in stats_qzx.unbind(-1)]
        kld_coll = []

        n_batches = int(np.ceil(len(stats_qzx[0]) / self.batch_size))
        with torch.no_grad():
            for i in range(n_batches):
                sub_stats_qzx = [x[np.random.choice(len(x), self.batch_size, replace=True)] for x in stats_qzx]
                kld = _kl_normal_loss(*sub_stats_qzx).item()
                kld_coll.append(kld)
        return {
            'kld': np.mean(kld_coll)
        }