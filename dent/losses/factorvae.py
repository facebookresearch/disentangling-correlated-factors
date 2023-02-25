# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fastargs.decorators import param
import torch
import torch.nn as nn
import torch.nn.functional as F

import dent.losses.baseloss
import dent.utils.initialization
from .utils import _reconstruction_loss, _kl_normal_loss
from .utils import _permute_dims


class Loss(dent.losses.baseloss.BaseLoss):
    """
    Compute the Factor-VAE loss as per Algorithm 2 of [1]

    Parameters
    ----------
    device : torch.device

    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.

    optimizer_d : torch.optim

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """

    @param('factorvae.gamma')
    @param('factorvae.discr_lr')
    @param('factorvae.discr_betas')
    @param('factorvae.log_components')
    def __init__(self, device, gamma, discr_lr, discr_betas, log_components, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.device = device
        self.discriminator = FactorDiscriminator().to(self.device)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=discr_lr,
                                            betas=discr_betas)
        self.log_components = log_components
        self.mode = 'optimizes_internally'

    def __call__(self, data, model, optimizer, **kwargs):
        is_train = model.training
        self._pre_call(is_train)

        log_data = {}

        # factor-vae split data into two batches. In the paper they sample 2 batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # Factor VAE Loss
        model_out1 = model(data1)
        if isinstance(model_out1['stats_qzx'], torch.Tensor):
            model_out1['stats_qzx'] = model_out1['stats_qzx'].unbind(-1)

        rec_loss = _reconstruction_loss(data1,
                                        model_out1['reconstructions'],
                                        distribution=self.rec_dist)
        log_data['rec_loss'] = rec_loss.item()

        kl_loss = _kl_normal_loss(*model_out1['stats_qzx'], return_components=True)
        if self.log_components:
            log_data.update(
                {f'kl_loss_{i}': value.item() for i, value in enumerate(kl_loss)})
        kl_loss = kl_loss.sum()
        log_data['kl_loss'] = kl_loss.item()

        d_z = self.discriminator(model_out1['samples_qzx'])
        # We want log(p_true/p_false). If not using logisitc regression but softmax
        # then p_true = exp(logit_true) / Z; p_false = exp(logit_false) / Z
        # so log(p_true/p_false) = logit_true - logit_false
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # with sigmoid (not good results) should be `tc_loss = (2 * d_z.flatten()).mean()`

        vae_loss = rec_loss + kl_loss + self.gamma * tc_loss

        log_data['loss'] = vae_loss.item()
        log_data['tc_loss'] = tc_loss.item()

        if not is_train:
            # don't backprop if evaluating
            return {'loss': vae_loss, 'to_log': log_data}

        # Compute VAE gradients
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)

        # Discriminator Loss
        # Get second sample of latent distribution
        samples_qzx2 = model.sample_qzx(data2)
        z_perm = _permute_dims(samples_qzx2).detach()
        d_z_perm = self.discriminator(z_perm)

        # Calculate total correlation loss
        # for cross entropy the target is the index => need to be long and says
        # that it's first output for d_z and second for perm
        ones = torch.ones(half_batch_size,
                          dtype=torch.long,
                          device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) +
                           F.cross_entropy(d_z_perm, ones))
        # with sigmoid would be :
        # d_tc_loss = 0.5 * (self.bce(d_z.flatten(), ones) + self.bce(d_z_perm.flatten(), 1 - ones))

        # TO-DO: check ifshould also anneals discriminator if not becomes too good ???
        #d_tc_loss = anneal_reg * d_tc_loss

        # Compute discriminator gradients
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()

        # Update at the end (since pytorch 1.5. complains if update before)
        optimizer.step()
        self.optimizer_d.step()

        log_data['discrim_loss'] = d_tc_loss.item()

        return {'loss': vae_loss, 'to_log': log_data}

    def attrs_to_chkpt(self):
        return {
            'discriminator.state_dict': self.discriminator.state_dict(),
            'optimizer_d.state_dict': self.optimizer_d.state_dict(),
            'n_train_steps': self.n_train_steps
        }

class FactorDiscriminator(nn.Module):

    @param('factorvae.discr_neg_slope', 'neg_slope')
    @param('factorvae.discr_hidden_units', 'hidden_units')
    @param('factorvae.discr_latent_dim', 'latent_dim')
    def __init__(self, neg_slope, latent_dim, hidden_units):
        """Discriminator proposed in [1].

        Parameters
        ----------
        neg_slope: float
            Hyperparameter for the Leaky ReLu

        latent_dim : int
            Dimensionality of latent variables.

        hidden_units: int
            Number of hidden units in the MLP

        Model Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 logits

        References:
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
            arXiv preprint arXiv:1802.05983 (2018).

        """
        super(FactorDiscriminator, self).__init__()

        # Activation parameters
        self.neg_slope = neg_slope
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        self.hidden_units = hidden_units
        # theoretically 1 with sigmoid but gives bad results => use 2 and softmax
        out_units = 2

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, hidden_units)
        self.lin3 = nn.Linear(hidden_units, hidden_units)
        self.lin4 = nn.Linear(hidden_units, hidden_units)
        self.lin5 = nn.Linear(hidden_units, hidden_units)
        self.lin6 = nn.Linear(hidden_units, hidden_units)
        self.lin7 = nn.Linear(hidden_units, out_units)

        self.reset_parameters()

    def forward(self, z):

        # Fully connected layers with leaky ReLu activations
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.leaky_relu(self.lin6(z))
        z = self.lin7(z)

        return z

    def reset_parameters(self):
        self.apply(dent.utils.initialization.weights_init)
