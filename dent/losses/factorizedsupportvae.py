"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import math
import time

from fastargs.decorators import param
import itertools as it
import numpy as np
import torch

import dent.losses.baseloss
from .utils import _reconstruction_loss, _kl_normal_loss, linear_annealing

class Loss(dent.losses.baseloss.BaseLoss):
    """
    Compute a factorized support constraints, which does not enforce full factorization of latents, 
    but instead only factorization of the support.

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    beta : float
        Weight on factorized support constraint.

    matching : str
        Type of matching between latent support and full factorized support. Currently included: hausdorff_hard, hausdorff_soft
    
    factorized_support_estimation : str
        How to estimate the distance to the fully factorized support. Either by selecting random pairs, OR by directly computing
        the full factorized support.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """


    @param('factorizedsupportvae.use_rec')
    @param('factorizedsupportvae.beta')
    @param('factorizedsupportvae.gamma')
    @param('factorizedsupportvae.delta')
    @param('factorizedsupportvae.reg_mode')
    @param('factorizedsupportvae.matching')
    @param('factorizedsupportvae.factorized_support_estimation')
    @param('factorizedsupportvae.num_support_estimators')
    @param('factorizedsupportvae.latent_select')
    @param('factorizedsupportvae.num_latent_pairs')
    @param('factorizedsupportvae.temperature_1')
    @param('factorizedsupportvae.temperature_2')
    @param('factorizedsupportvae.inner_prob_samples')
    @param('factorizedsupportvae.outer_prob_samples')
    @param('factorizedsupportvae.log_components')    
    def __init__(
        self, 
        n_data, 
        use_rec=1,
        beta=1., 
        gamma=1., 
        delta=1., 
        reg_mode='minimal_support', 
        reg_range=[0., 1.],
        matching='hausdorff_soft', 
        factorized_support_estimation='random', 
        num_support_estimators=100,
        latent_select='pair',
        num_latent_pairs=25,
        temperature_1=1.,
        temperature_2=1.,
        inner_prob_samples=5,
        outer_prob_samples=20,
        log_components=False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.use_rec = use_rec
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.reg_mode = reg_mode
        self.reg_range = reg_range
        self.matching = matching
        self.factorized_support_estimation = factorized_support_estimation
        self.num_support_estimators = num_support_estimators
        self.latent_select = latent_select
        self.num_latent_pairs = num_latent_pairs        
        if self.latent_select == 'all':
            self.num_latent_pairs = 1
            self.factorized_support_estimation = 'random'
        self.temperature_1 = temperature_1
        self.temperature_2 = temperature_2
        self.inner_prob_samples = inner_prob_samples
        self.outer_prob_samples = outer_prob_samples
        self.log_components = log_components
        self.eps = 1e-6
        self.avail_pairs_of_latents = None

    def __call__(self, data, reconstructions, stats_qzx, is_train, samples_qzx, **kwargs):
        self._pre_call(is_train)
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = stats_qzx.unbind(-1)
        batch_size = data.size(0)
        log_data = {}

        ### Reconstruction Loss (i.e. E_q[log p(x_n|z)]) (Part 1 in Eq. 1).
        if self.use_rec:
            rec_loss = _reconstruction_loss(data, reconstructions, distribution=self.rec_dist)
        else:
            rec_loss = 0

        ### Compute factorized support match.
        z = stats_qzx[0]

        ##--- Option 1 & 2: Hard and Soft Hausdorff over N random latent tuples (pairs/triplets)
        latent_dim = stats_qzx[0].shape[-1]
        self.pairs_of_latents = []

        if self.avail_pairs_of_latents is None:
            self.avail_pairs_of_latents = np.array(list(it.combinations(range(latent_dim), 2)))

        if self.latent_select == 'pair':
            self.num_latent_pairs = np.clip(self.num_latent_pairs, None, len(self.avail_pairs_of_latents))
            self.pairs_of_latents = self.avail_pairs_of_latents[np.random.choice(len(self.avail_pairs_of_latents), self.num_latent_pairs, replace=False)]
            n_latent_samples = 2
        elif self.latent_select == 'all':
            self.pairs_of_latents = np.arange(latent_dim).reshape(1, -1)
            n_latent_samples = latent_dim

        factorizedsupport_loss = 0.

        #Convert z [BS x D] to sub_z [BS x NUM_PAIRS x 2].
        sub_z = z[..., self.pairs_of_latents]

        if self.factorized_support_estimation == 'random':
            #Extract <num_support_estimators> arbitrary combinations of pairwise 1D latent supports for each latent pairing.
            rand_idcs_ab = torch.randint(0, len(z), (self.num_support_estimators, self.num_latent_pairs, n_latent_samples), device=z.device)
            #Select the respective latent entries and support values, then detach factorized support.
            factorized_z = sub_z.gather(0, rand_idcs_ab)
        elif self.factorized_support_estimation == 'full':
            ref_range = torch.arange(len(z), device=z.device)
            idcs_a = torch.tile(ref_range, dims=(len(z),))
            idcs_b = torch.repeat_interleave(ref_range, len(z))
            # support_idcs_ab = torch.stack([idcs_a, idcs_b], dim=-1).unsqueeze(1)
            # factorized_z = sub_z.gather(0, support_idcs_ab)
            #Because gather is not broadcastable, we resort to hacky concatenating.            
            factorized_z = torch.cat([sub_z[idcs_a, :, 0:1], sub_z[idcs_b, :, 1:2]], dim=-1)

        dists = ((factorized_z.unsqueeze(1) - sub_z.unsqueeze(0)) ** 2).sum(-1)
        # ### Detaching factorized values consistently causes divergence.
        # dists = ((factorized_z.unsqueeze(1).detach() - sub_z.unsqueeze(0)) ** 2).sum(-1)
        log_data['max_distance'] = dists.max().item()
        
        if 'hausdorff_hard' in self.matching:
            #Supportloss is just the Hausdorff distance.
            if 'mean' in self.matching:
                dists = dists.min(1)[0]
                norm = torch.sum(dists > 0, dim=0)
                factorizedsupport_loss = torch.sum(dists.sum(0) / norm)
            else:
                factorizedsupport_loss = dists.min(1)[0].max(0)[0].sum()
                # factorizedsupport_loss = dists.min(1)[0].sum(1).max()
        elif self.matching == 'hausdorff_prob':
            inner_term_probs = torch.softmax(-1. * dists / self.temperature_1, dim=1).permute(0, 2, 1)
            sample_indices = torch.multinomial(inner_term_probs.reshape(-1, batch_size), self.inner_prob_samples, replacement=True)
            sample_indices = sample_indices.reshape(*inner_term_probs.shape[:-1], self.inner_prob_samples).permute(0, 2, 1)
            # Convert distance tensor from 
            # -> len_fact_supp x batch_size x num_latent_pairs 
            # -> len_fact_supp x inner_prob_samples x num_latent_pairs 
            # -> len_fact_supp x num_latent_pairs
            dists = dists.gather(1, sample_indices).mean(1)
            outer_term_probs = torch.softmax(dists / self.temperature_2, dim=0).permute(1, 0)
            sample_indices = torch.multinomial(outer_term_probs, self.outer_prob_samples, replacement=True).permute(1, 0)
            # Convert distance tensor from 
            # -> len_fact_supp x num_latent_pairs
            # -> outer_prob_samples x num_latent_pairs
            # -> 1
            dists = dists.gather(0, sample_indices).mean(0)
            factorizedsupport_loss = dists.sum()
        elif 'hausdorff_soft_single' in self.matching:
            min_dists_idcs = torch.argmin(dists, dim=1).unsqueeze(1)
            if 'min' in self.matching:
                dists_weights = torch.softmax(-1. * dists / self.temperature_1, dim=1).gather(1, min_dists_idcs)
            elif 'max' in self.matching:
                dists_weights = torch.softmax(dists / self.temperature_1, dim=1).gather(1, min_dists_idcs)
            weighted_dists = dists_weights * dists.gather(1, min_dists_idcs)
            max_w_dists_idcs = torch.argmax(weighted_dists, dim=0).unsqueeze(0)
            weighted_dists_weights = torch.softmax(weighted_dists / self.temperature_2, dim=0).gather(0, max_w_dists_idcs)
            weighted_dists = weighted_dists_weights * weighted_dists.gather(0, max_w_dists_idcs)
            factorizedsupport_loss = weighted_dists.sum() 
        elif 'hausdorff_soft_full' in self.matching:
            if 'min' in self.matching:
                weighted_dists = torch.sum(torch.softmax(-1. * dists / self.temperature_1, dim=1) * dists, dim=1)
            elif 'max' in self.matching:
                weighted_dists = torch.sum(torch.softmax(dists / self.temperature_1, dim=1) * dists, dim=1)
            factorizedsupport_loss = torch.sum(torch.softmax(weighted_dists / self.temperature_2, dim=0) * weighted_dists)

        log_data['factorizedsupport_loss'] = factorizedsupport_loss.item()

        # Ensure correct scale matching.
        if self.reg_mode == 'variance':
            factorizedsupport_scale_reg = torch.relu(1 - torch.sqrt(torch.var(z, dim=1) + self.eps)).sum()
        elif self.reg_mode == 'minimal_support':
            factorizedsupport_scale_reg = torch.sum(torch.relu(self.reg_range[1] - torch.max(z, dim=1).values) + torch.relu(torch.min(z, dim=1).values - self.reg_range[0]))
        else:
            factorizedsupport_scale_reg = torch.Tensor([0.]).to(z.device)
        log_data['factorizedsupport_scale_reg'] = factorizedsupport_scale_reg.item()

        # Compute Standard Beta-VAE KL-Loss.
        kl_loss = _kl_normal_loss(*stats_qzx, return_components=True)
        if self.log_components:
            log_data.update(
                {f'kl_loss_{i}': value.item() for i, value in enumerate(kl_loss)})
        kl_loss = kl_loss.sum()
        log_data['kl_loss'] = kl_loss.item()
        
        loss = rec_loss + self.beta * kl_loss + self.gamma * factorizedsupport_loss + self.delta * factorizedsupport_scale_reg
        # from IPython import embed; embed()
        log_data['loss'] = loss.item()

        return {'loss': loss, 'to_log': log_data}

    def attrs_to_chkpt(self):
        return {'n_train_steps': self.n_train_steps}
