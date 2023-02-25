"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import logging
import math
from functools import reduce
from collections import defaultdict
import json
from timeit import default_timer

from fastargs.decorators import param
from tqdm import trange, tqdm
import numpy as np
import torch

import dent.metrics.utils
import dent.utils.math
import dent.utils.io

TEST_LOSSES_FILE = "test_losses.log"
METRICS_FILENAME = "metrics.log"
METRIC_HELPERS_FILE = "metric_helpers.pth"

class BaseEvaluator:
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """
    @param('eval.to_compute')
    def __init__(self,
                 to_compute=['metrics', 'losses'],
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 is_progress_bar=True):

        self.to_compute=to_compute
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger.info("Testing Device: {}".format(self.device))

    def __call__(self, model, data_loader, loss_f=None):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_metrics: bool, optional
            Whether to compute and store the disentangling metrics.

        is_losses: bool, optional
            Whether to compute and store the test losses.
        """
        start = default_timer()
        is_still_training = model.training
        model.eval()

        output_data = {}
        model_outputs = None
        #TODO: DON'T RETAIN ALL DECODINGS IF NOT NEEDED
        if 'metrics' in self.to_compute:
            self.logger.info('Computing metrics...')
            computed_metrics = self.compute_metrics(model, data_loader)
            output_data['metrics'] = computed_metrics

        if 'losses' in self.to_compute:
            if loss_f is None:
                raise ValueError('Please provide a loss_f to compute losses!')
            self.logger.info('Computing losses...')
            total_loss, loss_collect = self.compute_losses(model, data_loader, loss_f)
            output_data['total_loss'] = total_loss
            output_data['losses'] = loss_collect

        if is_still_training:
            model.train()

        self.logger.info('Finished evaluating after {:.1f} min.'.format(
            (default_timer() - start) / 60))

        return output_data

    def compute_losses(self, model, dataloader, loss_f):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        log_storer = defaultdict(list)
        loss_storer = []

        iterator = tqdm(dataloader, leave=False, disable=not self.is_progress_bar)
        for data, _ in iterator:
            data = data.to(self.device)

            if not loss_f.optimizes_internally:
                model_output = model(data)
                loss_out = loss_f(data, is_train=model.training, **model_output)
            else:
                loss_out = loss_f(data, model, None)
            for key, item in loss_out['to_log'].items():
                log_storer[key].append(item)
            loss_storer.append(loss_out['loss'].item())
        return np.mean(loss_storer), {key: np.mean(item) for key, item in log_storer.items()}

    def compute_metrics(self, model, dataloader):
        """Compute all the metrics.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        # import importlib 
        # import dent.metrics.utils
        # importlib.reload(dent.metrics.utils)
        # importlib.reload(dent.metrics.aam)
        # importlib.reload(dent.metrics.mig)
        metric_group = dent.metrics.utils.MetricGroup(device=self.device, logger=self.logger)
        metric_out = metric_group.compute(dataloader, model)
        aggregated_metrics = {}
        if metric_out['computed_metrics']:
            metric_out['computed_metrics'] = {key: item.item() for key, item in metric_out['computed_metrics'].items()}
            aggregated_metrics.update({f'full_{key}': item for key, item in metric_out['computed_metrics'].items()})
        if metric_out['computed_instance_metrics']:
            metric_out['computed_instance_metrics'] = {key: item.item() for key, item in metric_out['computed_instance_metrics'].items()}            
            aggregated_metrics.update({f'instance_{key}': item for key, item in metric_out['computed_instance_metrics'].items()})
        return aggregated_metrics