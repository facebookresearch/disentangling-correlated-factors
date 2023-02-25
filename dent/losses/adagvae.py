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
- Integration as a BaseMetric.
- Fixed pairwise sampling errors between code and paper proposal.
- Function and variable renaming.
"""
from fastargs.decorators import param
import torch

import dent.losses.baseloss
from .utils import _reconstruction_loss, _kl_normal_loss, _kl_divergence, linear_annealing

class Loss(dent.losses.baseloss.BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    gamma : float, optional
        Weight of the KL divergence term.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    @param('adagvae.thresh_mode')
    @param('adagvae.average_mode')
    @param('adagvae.thresh_ratio')
    @param('adagvae.beta')
    @param('adagvae.annealing')
    @param('adagvae.C_fin')
    @param('adagvae.C_init')
    @param('adagvae.gamma')
    @param('adagvae.anneal_steps')
    @param('adagvae.log_components')
    def __init__(self, thresh_mode, average_mode, thresh_ratio, beta, annealing, C_fin, C_init, gamma, anneal_steps, log_components, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.annealing = annealing
        self.gamma = gamma
        self.C_fin = C_fin
        self.C_init = C_init
        self.average_mode = average_mode
        self.thresh_mode = thresh_mode
        self.thresh_ratio = thresh_ratio
        self.anneal_steps = anneal_steps        
        self.log_components = log_components
        self.mode = 'pre_forward'

    def __call__(
        self, model, data, paired_data, shared_idcs=None, is_train=True, **kwargs):
        """Main AdaGVAE criterion.

        Parameters
        ----------
        model : torch.nn.Module or any inherited instance thereof.
            This model should have a model.encoder corresponding to the first and second moment
            ouf the approximate posterior (i.e. mean and logvar) denoted with 'stats_qzx'. In 
            addition, it should also contain a model.decoder to reconstruct from latents, as well
            as some functionality to sample those given 'stats_qzx' or adapted version thereof.
        data, paird_data : torch.Tensor.
            Of size [batch_size, channels, height, width] containing the (paired) input training images.
            Note that the i-th entry of data has its corresponding paired image in the i-th entry of paired_data.
        shared_idcs : None or torch.Tensor.
            If given, provides ground-truth information of shared latent entries between embeddings of 
            data and paired_data. If not given, will be derived.
        is_train : Bool.
            Simple flag to put the objective into training or evaluation mode.
        """
        log_data = {}

        model_out = model.encoder(data)
        paired_model_out = model.encoder(paired_data)

        stats_qzx = model_out['stats_qzx']
        paired_stats_qzx = paired_model_out['stats_qzx']

        self._pre_call(is_train)
        if isinstance(stats_qzx, torch.Tensor):
            mean_qzx, logvar_qzx = stats_qzx.unbind(-1)
        if isinstance(paired_stats_qzx, torch.Tensor):
            paired_mean_qzx, paired_logvar_qzx = paired_stats_qzx.unbind(-1)

        #Compute deltas between base and paired latents, either using 
        #(Symmetric) KL-Divergence or simple absolute distance of means.
        if self.thresh_mode == 'kl':
            delta_latents = _kl_divergence(
                mean_1=mean_qzx, logvar_1=logvar_qzx, 
                mean_2=paired_mean_qzx, logvar_2=paired_logvar_qzx)
        elif self.thresh_mode == 'symmetric_kl':
            kl_deltas_base_pair = _kl_divergence(
                mean_1=mean_qzx, logvar_1=logvar_qzx, 
                mean_2=paired_mean_qzx, logvar_2=paired_logvar_qzx)
            kl_deltas_pair_base = _kl_divergence(
                mean_1=paired_mean_qzx, logvar_1=paired_logvar_qzx, 
                mean_2=mean_qzx, logvar_2=logvar_qzx)
            delta_latents = 0.5 * kl_deltas_pair_base + 0.5 * kl_deltas_base_pair
        elif self.thresh_mode == 'dist':
            delta_latents = torch.abs(mean_qzx - paired_mean_qzx)

        if shared_idcs is None:
            #Compute threshold following Locatello et al. (post Eq.6),
            #after which latent entries are deemed "shared" if KL_DIV < threshold.
            #These shared indices are only computed when ground truth indices are NOT given.
            max_deltas = delta_latents.max(axis=1, keepdim=True).values
            min_deltas = delta_latents.min(axis=1, keepdim=True).values
            z_threshs = 0.5 * (min_deltas + max_deltas)
            shared_idcs = delta_latents < z_threshs       

        # compute averaged posterior based on either GVAE or ML-VAE.
        if self.average_mode == 'gvae':
            avg_logvar_qzx = torch.log(0.5 * torch.exp(logvar_qzx) + 0.5 * torch.exp(paired_logvar_qzx))
            avg_mean_qzx = 0.5 * mean_qzx + 0.5 * paired_mean_qzx
        elif self.average_mode == 'mlvae':
            var_qzx = torch.exp(logvar_qzx)
            paired_var_qzx = torch.exp(paired_logvar_qzx)
            avg_var_qzx = 2 * var_qzx * paired_var_qzx / (var_qzx + paired_var_qzx)
            avg_mean_qzx = (mean_qzx/var_qzx + paired_mean_qzx/paired_var_qzx) * avg_var_qzx * 0.5
            avg_logvar_qzx = torch.log(avg_var_qzx)

        #Replace each original entry in e.g. mean_qzx with the averaged variant when shared_idcs == 1.
        mean_qzx = torch.where(shared_idcs, avg_mean_qzx, mean_qzx)
        paired_mean_qzx = torch.where(shared_idcs, avg_mean_qzx, paired_mean_qzx)
        logvar_qzx = torch.where(shared_idcs, avg_logvar_qzx, logvar_qzx)
        paired_logvar_qzx = torch.where(shared_idcs, avg_logvar_qzx, paired_logvar_qzx)
        log_data['mean_num_shared'] = shared_idcs.sum(-1).float().mean(-1).item()

        kl_loss_base = _kl_normal_loss(mean_qzx, logvar_qzx, return_components=True)
        kl_loss_pair = _kl_normal_loss(paired_mean_qzx, paired_logvar_qzx, return_components=True)
        kl_loss = 0.5 * (kl_loss_base + kl_loss_pair)
        if self.log_components:
            log_data.update(
                {f'kl_loss_{i}': value.item() for i, value in enumerate(kl_loss)})
        kl_loss = kl_loss.sum()
        log_data['kl_loss'] = kl_loss.item()

        #Generate reconstructions from averaged latents.
        samples_qzx = model.reparameterize(mean_qzx, logvar_qzx)['samples_qzx']
        reconstructions = model.decoder(samples_qzx)['reconstructions']

        paired_samples_qzx = model.reparameterize(paired_mean_qzx, paired_logvar_qzx)['samples_qzx']        
        paired_reconstructions = model.decoder(paired_samples_qzx)['reconstructions']

        rec_loss_base = _reconstruction_loss(data, reconstructions, distribution=self.rec_dist)
        rec_loss_pair = _reconstruction_loss(paired_data, paired_reconstructions, distribution=self.rec_dist)
        rec_loss = 0.5 * (rec_loss_base + rec_loss_pair)
        log_data['rec_loss'] = rec_loss.item()

        if self.annealing == 'higgins':
            loss = rec_loss + self.beta * kl_loss            
        else:
            C = (linear_annealing(self.C_init, self.C_fin, self.n_train_steps, self.anneal_steps) if is_train else self.C_fin)
            loss = rec_loss + self.gamma * (kl_loss - C).abs()
        log_data['loss'] = loss.item()

        if 'to_log' in model_out:
            log_data.update(model_out['to_log'])

        return {'loss': loss, 'to_log': log_data}

    def attrs_to_chkpt(self):
        return {'n_train_steps': self.n_train_steps}
