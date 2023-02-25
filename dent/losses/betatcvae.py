# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fastargs.decorators import param
import torch

import dent.losses.baseloss
from .utils import _reconstruction_loss, _kl_normal_loss, linear_annealing

class Loss(dent.losses.baseloss.BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """


    @param('betatcvae.alpha')
    @param('betatcvae.beta')
    @param('betatcvae.gamma')
    @param('betatcvae.is_mss')
    @param('betatcvae.log_components')        
    def __init__(self, n_data, alpha=1., gamma=1., beta=6., log_components=False, is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.is_mss = is_mss
        self.log_components = log_components

    def __call__(self, data, reconstructions, stats_qzx, is_train, samples_qzx, **kwargs):
        self._pre_call(is_train)
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = stats_qzx.unbind(-1)
        batch_size = data.size(0)
        log_data = {}

        ### Reconstruction Loss (i.e. E_q[log p(x_n|z)]) (Part 1 in Eq. 1).
        rec_loss = _reconstruction_loss(data, reconstructions, distribution=self.rec_dist)
        
        ##### beta-TCVAE breaks down the standard KL-term in beta-VAE, KL[q(z|x_n)||p(z)], into multiple components including 
        ##### the total correlation, which the authors believe to be the most important aspect, and such which to scale independently:

        ### Compute Total Correlation (Part 2 in Eq. 2).
        # Compute log(q((z_j)|x_i)) for every sample in batch: [batch_size x batch_size x latent_dim]
        # i.e. compute the probability of latents z_j under q(*|x_i) induced by sample x_i.
        log_qzx_cross = dent.utils.math.log_density_gaussian(
            samples_qzx.unsqueeze(dim=1), *[x.unsqueeze(dim=0) for x in stats_qzx])

        # (Optional) Apply minibatch stratified sampling (c.f. Eq. S6) to log_qzx_cross[i, j, :].
        # In essence, we estimate q(z) using a minibatch {x_1, ..., x_m} for a z that was originally sampled from q(z|x_*).
        # See alse Eq. S5 for a derivation.
        if self.is_mss:
            N, M = self.n_data, batch_size - 1
            M = batch_size - 1
            strat_weight = (N - M) / (N * M)
            importance_weight = torch.Tensor(batch_size, batch_size).fill_(1 / M)
            importance_weight.view(-1)[::M + 1] = 1 / N
            importance_weight.view(-1)[1::M + 1] = strat_weight
            importance_weight[M - 1, 0] = strat_weight #What is the purpose of htis line?
            importance_weight = importance_weight.log().view(batch_size, batch_size, 1).to(samples_qzx.device)
            # Compute log prod_l q(z(x_j)_l) = sum_l(log(sum_i(q(z(x_j)_l|x_i))).
            # Note the use of logsumexp, where the exp converts log(q((z_j)|x_i)) to q((z_j)|x_i) for marginalization.
            # I.e. we first marginalize out x_i from q(z_j|x_i) -> q(z_j) before computing prod_l q(z_l).
            log_qz_product = torch.logsumexp(importance_weight + log_qzx_cross, dim=1, keepdim=False).sum(1, keepdim=False)
            # Compute the final log(q(z)) for the Total Correlation KL[q(z)||prod_l q(z_l)], which is given as
            # log(sum_i(prod_l q(z(x_j)_l|x_i))) = log(sum_i(log(sum_l q(z(x_j)_l|x_i)))):
            log_qz = torch.logsumexp(importance_weight.squeeze(-1) + log_qzx_cross.sum(2), dim=1, keepdim=False)
            # log_qz = torch.logsumexp(log_qzx_cross.sum(dim=2, keepdim=False), dim=1, keepdim=False)
        else:
            log_qz_product = (torch.logsumexp(log_qzx_cross, dim=1, keepdim=False) - math.log(batch_size * self.n_data)).sum(1)
            log_qz = (torch.logsumexp(log_qzx_cross.sum(2), dim=1, keepdim=False) - math.log(batch_size * self.n_data))

        # Finally, the total correlation (assuming z ~ q(*) to be uniform) KL[q(z)||prod_l q(z_l)] = E_q(z)[log(q(z)) - prod_l q(z_l)] = Mean[...]
        total_correlation = torch.mean(log_qz - log_qz_product)

        ### Compute the mutual information (Part 1, Eq. 2): I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        log_qzx = dent.utils.math.log_density_gaussian(samples_qzx, *stats_qzx).view(batch_size, -1).sum(1)
        mutual_information = (log_qzx - log_qz.unsqueeze(-1)).mean()

        ### Compute Dimension-wise KL (Part 3, Eq. 2): KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        zeros = torch.zeros_like(samples_qzx, device=samples_qzx.device)
        log_pz = dent.utils.math.log_density_gaussian(samples_qzx, zeros, zeros).sum(1)
        dim_kld = (log_qz_product - log_pz).mean()

        # total loss
        loss = rec_loss + (self.alpha * mutual_information + self.beta * total_correlation + self.gamma * dim_kld)

        log_data['loss'] = loss.item()
        log_data['rec_loss'] = rec_loss.item()
        log_data['mutual_information'] = mutual_information.item()
        log_data['total_correlation'] = total_correlation.item()
        log_data['dim_kld'] = dim_kld.item()

        # computing this for storing and comparison purposes
        kl_loss = _kl_normal_loss(*stats_qzx, return_components=True)
        if self.log_components:
            log_data.update(
                {f'kl_loss_{i}': value.item() for i, value in enumerate(kl_loss)})

        return {'loss': loss, 'to_log': log_data}

    def attrs_to_chkpt(self):
        return {'n_train_steps': self.n_train_steps}
