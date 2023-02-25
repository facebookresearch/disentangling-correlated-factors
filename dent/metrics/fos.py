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

class FoS(BaseMetric):
    @param('fos.num_pairs')
    @param('fos.batch_size')
    def __init__(self, num_pairs, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.num_pairs = num_pairs
        self.batch_size = batch_size

    @property
    def _requires(self):
        return ['stats_qzx']

    @property
    def _mode(self):
        return 'full'

    def __call__(self, stats_qzx, **kwargs):
        """Compute Factorization of Support using pairwise approximation.
        """
        stats_qzx = [x for x in stats_qzx.unbind(-1)]
        means_qzx = stats_qzx[0]

        latent_dim = stats_qzx[0].shape[-1]
        pairs_of_latents = []

        avail_pairs_of_latents = np.array(list(it.combinations(range(latent_dim), 2)))

        num_pairs = np.clip(self.num_pairs, None, len(avail_pairs_of_latents))
        if num_pairs == -1:
            num_pairs = len(avail_pairs_of_latents)

        fos_coll = []

        n_batches = int(np.ceil(len(means_qzx) / self.batch_size))
        with torch.no_grad():
            for i in range(n_batches):
                if not len(pairs_of_latents) or self.num_pairs != len(avail_pairs_of_latents):
                    pairs_of_latents = avail_pairs_of_latents[np.random.choice(len(avail_pairs_of_latents), num_pairs, replace=False)]
                
                #Convert z [BS x D] to sub_z [BS x NUM_PAIRS x 2].
                sub_z = means_qzx[i*self.batch_size:(i+1)*self.batch_size, pairs_of_latents]
                eff_bs = sub_z.size(0)
                
                ref_range = torch.arange(eff_bs, device=means_qzx.device)
                idcs_a = torch.tile(ref_range, dims=(eff_bs,))
                idcs_b = torch.repeat_interleave(ref_range, eff_bs)         
                factorized_z = torch.cat([sub_z[idcs_a, :, 0:1], sub_z[idcs_b, :, 1:2]], dim=-1)
                dists = ((factorized_z.unsqueeze(1) - sub_z.unsqueeze(0)) ** 2).sum(-1)
                
                fos_coll.append(dists.min(1)[0].max(0)[0].sum().detach().cpu().numpy())

        return {
            'fos': np.mean(fos_coll)
        }