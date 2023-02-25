"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import collections
import logging
import math

from fastargs.decorators import param
import numpy as np
import torch
from tqdm import tqdm, trange

import dent.utils.math
import dent.metrics

METRICS = [
    'mig',
    'aam',
    'sap_d',
    'dci_d',
    'fos',
    'kld',
    'rand_fos',
    'rand_kld',
    'modularity_d',
    'reconstruction_error'
]

def select_metric(name, **kwargs):
    if name not in METRICS:
        err = f"{name} not a valid metric. Select from {METRICS}."
        raise ValueError(err)

    if name == 'mig':
        return dent.metrics.MIG(**kwargs)
    if name == 'sap_d':
        return dent.metrics.SAPd(**kwargs)
    if name == 'dci_d':
        return dent.metrics.DCId(**kwargs)
    if name == 'fos':
        return dent.metrics.FoS(**kwargs)
    if name == 'kld':
        return dent.metrics.KLD(**kwargs)
    if name == 'rand_fos':
        return dent.metrics.randFoS(**kwargs)
    if name == 'rand_kld':
        return dent.metrics.randKLD(**kwargs)
    if name == 'modularity_d':
        return dent.metrics.Modularityd(**kwargs)
    if name == 'reconstruction_error':
        return dent.metrics.ReconstructionError(**kwargs)

class MetricGroup:
    @param('eval.metrics', 'metric_names')
    def __init__(self, device, metric_names, logger=logging.Logger(__name__), **kwargs):
        self.metric_names = metric_names
        self.device = device
        self.logger = logger
        self._initialize()

    def _initialize(self):
        self.metrics = collections.defaultdict(dict)
        self.requirements = collections.defaultdict(list)
        for metric_name in self.metric_names:
            metric = select_metric(name=metric_name, device=self.device)
            self.metrics[metric._mode][metric_name] = metric
            metric_requirements = metric._requires
            self.requirements[metric._mode].extend(metric_requirements)
        self.requirements = {key: list(set(item)) for key, item in self.requirements.items()}

    def compute(self, dataloader, model, n_samples=10000, disable_tqdm=False):
        self.logger.info(f'Computing Metrics: {self.metric_names}.')
        self.logger.info('Metric prerequisites -- {}.'.format(' | '.join(f'{key}: {item}' for key, item in self.requirements.items())))
        #Extract and aggregate relevant data and model properties.
        task = {
            'n_data_samples': len(dataloader.dataset),
            'img_size': dataloader.dataset.img_size,
            'dist_nparams': model.dist_nparams,
            'latent_dim': model.latent_dim,
            'n_samples': n_samples,
            'device': self.device,
            'logger': self.logger
        }
        if task['n_samples'] > task['n_data_samples']:
            raise ValueError(f"Can't draw more samples {task['n_samples']} then "
                "datapoints ({task['n_data_samples']}) available!")
        if hasattr(dataloader.dataset, 'lat_sizes') and hasattr(dataloader.dataset, 'lat_names'):
            task['n_latent_variations'] = dataloader.dataset.lat_sizes
            task['n_latent_factors'] = len(task['n_latent_variations'])
            task['gt_factor_names'] = dataloader.dataset.lat_names
        else:
            if 'marginal_entropy' in self.requirements['full'] or 'conditional_entropy' in self.requirements['full']:
                raise AttributeError('The chosen dataset provides no explicit factors of variation to compute the conditional entropy!')

        #Compute q(z|x) through full dataloader forward. In this particular case, 
        #the encoder returns the sufficient statistics for q(z|x) (stats), which are
        #then used to draw samples from q(z|x). See EQ.5 in https://arxiv.org/pdf/1802.04942.pdf.
        model_outputs, computed_instance_metrics = self._compute_model_outputs_and_instance_metrics(
            dataloader, model, task, disable_tqdm=disable_tqdm)

        if 'full' in self.requirements:
            requirements = {}
            for requirement in self.requirements['full']:
                if requirement == 'samples_qzx':
                    requirements[requirement] = model_outputs['samples_qzx']
                if requirement == 'stats_qzx':
                    requirements[requirement] = model_outputs['stats_qzx']
                if requirement == 'gt_factors':
                    requirements[requirement] = model_outputs['gt_factors']
                if requirement == 'marginal_entropy':
                    #Estimate the marginal entropy H(z_j). See EQ.5 in https://arxiv.org/pdf/1802.04942.pdf.
                    #In particular, we compute H(z_j) = E_{q(z_j)} [-log q(z_j)] = E_{p(x)}E_{q(z_j|x)}[-log q(z_j)] 
                    #with q(z_j|x) the respective encoder output and p(x) the empirical distribution p(x), assumed
                    #to be uniform, i.e. q(z) = E_{p(x)}[q(z|x)] = sum_{n=1}^N 1/N q(z|x_n).
                    #This mean we
                    #   * first estimate q(z) in a Monte-Carlo fashion using q(z_j|x). This is done by fixing 
                    #     a single z, varying x and looking at the output probabilities.
                    #   * compute entropy.
                    requirements[requirement] = self._estimate_marginal_entropy(
                        **model_outputs, task=task)                
                if requirement == 'conditional_entropy':
                    #Compute the conditional entropy for z given ground truth factors v: H(z|v)
                    requirements[requirement] = self._estimate_conditional_entropy(
                        **model_outputs, task=task)                
                if requirement == 'reconstructions':
                    requirements[requirement] = model_outputs['reconstructions']
                if requirement in task:
                    requirements[requirement] = task[requirement]
            
        computed_metrics = {} if 'full' in self.metrics else None
        if computed_metrics is not None:
            for metric_name, metric in self.metrics['full'].items():
                self.logger.info(f'Computing metric: {metric_name}')
                metric_out = metric(**requirements)
                if isinstance(metric_out, dict):
                    for key, item in metric_out.items():
                        computed_metrics[metric_name+'_'+key] = item
                else:
                    computed_metrics[metric_name] = metric_out

        return {
            'computed_metrics': computed_metrics, 
            'computed_instance_metrics': computed_instance_metrics}

    def _compute_model_outputs_and_instance_metrics(self, dataloader, model, task, disable_tqdm=False):
        model_outputs = {
            'stats_qzx': torch.zeros(task['n_data_samples'], task['latent_dim'], task['dist_nparams'], device=task['device']),
            'samples_qzx': torch.zeros(task['n_data_samples'], task['latent_dim'], device=task['device'])
        }
        count = 0
        computed_instance_metrics = collections.defaultdict(list) if 'instance' in self.metrics else None
        with torch.no_grad():
            for x, gt_factors in tqdm(dataloader, leave=False, desc='Computing embeddings...', disable=disable_tqdm):
                x = x.to(task['device'])
                batch_size = x.size(0)
                idcs = slice(count, count + batch_size)
                model_output = model(x)
                for key in model_outputs.keys():
                    if key != 'gt_factors':
                        model_outputs[key][idcs] = model_output[key]
                if gt_factors is not None:
                    if 'gt_factors' not in model_outputs:
                        model_outputs['gt_factors'] = torch.zeros(
                            task['n_data_samples'], gt_factors.size(-1), device=task['device'])
                    model_outputs['gt_factors'][idcs] = gt_factors
                count += batch_size

                if computed_instance_metrics is not None:
                    instance_metrics_requirements = {'data_samples': x, **model_output}
                    for metric_name, metric in self.metrics['instance'].items():
                        computed_instance_metrics[metric_name].extend(metric(**instance_metrics_requirements))

        if computed_instance_metrics is not None:
            computed_instance_metrics = {key: torch.mean(torch.stack(item)) for key, item in computed_instance_metrics.items()}

        return model_outputs, computed_instance_metrics

    def _estimate_marginal_entropy(self, samples_qzx, stats_qzx, task, **kwargs):
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = stats_qzx.unbind(-1)
        n_samples_qzx, latent_dim_qzx = samples_qzx.shape
        H_z = torch.zeros(latent_dim_qzx, device=samples_qzx.device)
        n_samples_to_draw = np.clip(task['n_samples'], None, n_samples_qzx)
        samples_x_idcs = torch.randperm(n_samples_qzx, device=samples_qzx.device)[:n_samples_to_draw]
        samples_qzx = samples_qzx.index_select(0, samples_x_idcs).permute(1,0)
        samples_qzx = samples_qzx.unsqueeze(0).expand(n_samples_qzx, latent_dim_qzx, n_samples_to_draw)

        mean_qzx = stats_qzx[0].unsqueeze(-1).expand(n_samples_qzx, latent_dim_qzx, n_samples_to_draw)
        log_var_qzx = stats_qzx[1].unsqueeze(-1).expand(n_samples_qzx, latent_dim_qzx, n_samples_to_draw)
        log_n_data_samples = math.log(n_samples_qzx)

        mini_batch_size = 10
        for k in range(0, n_samples_to_draw, mini_batch_size):
            # Compute log q(z_j|x) for n_samples_to_draw.
            idcs = slice(k, k + mini_batch_size)
            log_qzx = dent.utils.math.log_density_gaussian(
                samples_qzx[..., idcs], mean_qzx[..., idcs], log_var_qzx[..., idcs])
            log_qz = -log_n_data_samples + torch.logsumexp(log_qzx, dim=0, keepdim=False)
            H_z += (-log_qz).sum(1)
        return H_z / n_samples_to_draw

    def _estimate_conditional_entropy(self, samples_qzx, stats_qzx, task, **kwargs):
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = stats_qzx.unbind(-1)
        n_samples_qzx, latent_dim_qzx = samples_qzx.shape
        samples_qzx = samples_qzx.view(*task['n_latent_variations'], latent_dim_qzx)
        stats_qzx = tuple(p.view(*task['n_latent_variations'], latent_dim_qzx) for p in stats_qzx)
        H_zv = torch.zeros(task['n_latent_factors'], latent_dim_qzx, device=samples_qzx.device)
        for v, (n_latent_variations_v, gt_factor_name_v) in enumerate(zip(task['n_latent_variations'], task['gt_factor_names'])):
            idcs = [slice(None)] * task['n_latent_factors']

            self.logger.info(f'Estimating conditional entropies for {gt_factor_name_v}.')
            for i in trange(n_latent_variations_v, leave=False):
                idcs[v] = i
                samples_qzx_v = samples_qzx[idcs].contiguous().view(n_samples_qzx // n_latent_variations_v, latent_dim_qzx)
                stats_qzx_v = tuple(
                    p[idcs].contiguous().view(n_samples_qzx//n_latent_variations_v, latent_dim_qzx) for p in stats_qzx)
                H_zv[v] += self._estimate_marginal_entropy(samples_qzx_v, stats_qzx_v, task) / n_latent_variations_v        
        return H_zv