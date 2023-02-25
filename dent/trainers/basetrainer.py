"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import collections
import logging
import os
import time
from timeit import default_timer
from uuid import uuid4

from fastargs.decorators import param
import imageio
import numpy as np
from tqdm import trange
import torch
from torch.nn import functional as F

import dent.utils.io
from .utils import LossesLogger
import datalogger

TRAIN_LOSSES_LOGFILE = "train_losses.log"

class BaseTrainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: dent.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: dent.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 loss_f,
                 device,
                 logger,
                 read_dir,
                 write_dir,
                 is_progress_bar=True):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.logger.info("Training Device: {}".format(self.device))
        self.start_epoch = 0
        self.read_dir = read_dir
        self.write_dir = write_dir

    def initialize(self, uid=None):
        self.uid = uid
        if uid is None:
            self.uid = str(uuid4())
        self.data_logger = datalogger.DataLogger(self.write_dir, uid=self.uid)

    @param('train.epochs')
    @param('train.iterations')    
    @param('train.checkpoint_every')
    @param('train.checkpoint_first')
    @param('log.printlevel')
    def __call__(self, data_loader, epochs=10, iterations=-1, checkpoint_every=10, checkpoint_first=0, printlevel=1):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()

        #If number of iterations is given, adapt #training epochs to a suitable value.
        #Generally opt for #epochs * #iter_per_epoch >= #iterations.
        if iterations > 0:
            epochs = int(np.ceil(iterations * 1. / len(data_loader)))

        for epoch in range(self.start_epoch, epochs):
            self.epoch = epoch
            lrs = [x['lr'] for x in self.optimizer.param_groups]
            self.logger.info(f'Ep.{epoch+1}/{epochs} - LRS: {lrs}')
            epoch_log_data = self._train_epoch(data_loader, epoch)
            epoch_log_data['num_samples'] = len(data_loader.dataset)

            for i, lr in enumerate(lrs):
                epoch_log_data[f'lr_group-{i}'] = lr
            if printlevel == 1:
                log_str = f'Loss: {epoch_log_data["loss"]:.4f}'
            elif printlevel == 2:
                log_str = ' | '.join(f'{key}: {item:.4f}' for key, item in epoch_log_data.items())
            self.logger.info(
                'Ep.{}/{} - {}'.format(
                    epoch + 1, epochs, log_str))

            epoch_log_data['epoch'] = epoch

            self.data_logger.log(epoch_log_data, log_key='train')

            self.scheduler.step()

            if (epoch + 1) % checkpoint_every == 0 or epoch < checkpoint_first:
                self.save_checkpoint('chkpt-{}.pth.tar'.format(epoch+1))
            self.save_checkpoint(dent.utils.io.CHECKPOINT)

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info(
            'Finished training after {:.1f} min.'.format(delta_time))

    def save_checkpoint(self, checkpoint_name):
        checkpoint_data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss_f': self.loss_f.attrs_to_chkpt(),
            'epoch': self.epoch + 1,
            'uid': self.uid
        }
        dent.utils.io.save_checkpoint(
            checkpoint_data,
            self.data_logger.write_dir,
            checkpoint_name
        )

    def initialize_from_checkpoint(self, chkpt_data, uid=None):
        self.model.load_state_dict(chkpt_data['model'])
        self.optimizer.load_state_dict(chkpt_data['optimizer'])
        self.scheduler.load_state_dict(chkpt_data['scheduler'])
        self.start_epoch = chkpt_data['epoch']
        uid = uid if uid else chkpt_data['uid']
        self.initialize(uid=uid)
        for attr, value in chkpt_data['loss_f'].items():
            if 'state_dict' in attr:
                temp = getattr(self.loss_f, attr.split('.')[0])
                temp.load_state_dict(value, strict=False)
            else:
                setattr(self.loss_f, attr, value)

    def _train_epoch(self, data_loader, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        to_log = collections.defaultdict(list)
        kwargs = dict(desc="Epoch {}".format(epoch + 1),
                      leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            # latent_vals = []
            for _, data_out in enumerate(data_loader):
                data = data_out[0]                
                iter_out = self._train_iteration(data)
                # latent_vals.append(data_out[1].detach().cpu().numpy())
                for key, item in iter_out['to_log'].items():
                    to_log[key].append(item)

                t.set_postfix(loss=iter_out['loss'],
                              norm_loss=iter_out['loss'] / len(data))
                t.update()
        # latent_vals = np.vstack(latent_vals)
        # from IPython import embed; embed()
        # corrs = {}
        # for i in range(5):
        #     for j in range(5):
        #         if i != j:
        #             corrs[(i,j)] = np.corrcoef(latent_vals[..., i], latent_vals[..., j])[1, 0]
        return {key: np.mean(item) for key, item in to_log.items()}


    def _train_iteration(self, samples):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        samples: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        samples = samples.to(self.device)

        if self.loss_f.mode == 'post_forward':
            model_out = self.model(samples)
            inputs = {
                'data': samples, 
                'is_train': self.model.training, 
                **model_out,
            }

            loss_out = self.loss_f(**inputs)
            loss = loss_out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if 'to_log' in model_out:
                loss_out['to_log'].update(model_out['to_log'])            
        elif self.loss_f.mode == 'pre_forward':
            inputs = {
                'model': self.model,
                'data': samples,
                'is_train': self.model.training
            }
            loss_out = self.loss_f(**inputs)
            loss = loss_out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if 'to_log' in model_out:
                loss_out['to_log'].update(model_out['to_log'])            
        elif self.loss_f.mode == 'optimizes_internally':
            # for losses that use multiple optimizers (e.g. Factor)
            loss_out = self.loss_f(samples, self.model, self.optimizer)
            loss = loss_out['loss']

        return {'loss': loss.item(), 'to_log': loss_out['to_log']}
