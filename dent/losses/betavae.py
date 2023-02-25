# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fastargs.decorators import param
import torch

import dent.losses.baseloss
from .utils import _reconstruction_loss, _kl_normal_loss

class Loss(dent.losses.baseloss.BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    @param('betavae.beta')
    @param('betavae.log_components')
    def __init__(self, beta, log_components, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.log_components = log_components

    def __call__(self, data, reconstructions, stats_qzx, is_train, **kwargs):   
        self._pre_call(is_train)
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = stats_qzx.unbind(-1)     

        log_data = {}
        rec_loss = _reconstruction_loss(data, reconstructions, distribution=self.rec_dist)
        log_data['rec_loss'] = rec_loss.item()

        kl_loss = _kl_normal_loss(*stats_qzx, return_components=True)
        if self.log_components:
            log_data.update(
                {f'kl_loss_{i}': value.item() for i, value in enumerate(kl_loss)})
        kl_loss = kl_loss.sum()
        log_data['kl_loss'] = kl_loss.item()

        loss = rec_loss + self.beta * kl_loss
        log_data['loss'] = loss.item()

        return {'loss': loss, 'to_log': log_data}
