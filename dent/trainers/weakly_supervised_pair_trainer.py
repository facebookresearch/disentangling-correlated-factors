"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import collections

from fastargs.decorators import param
import numpy as np
from tqdm import trange
import torch
from torch.nn import functional as F

from .basetrainer import BaseTrainer

class WeaklySupervisedPairTrainer(BaseTrainer):
    def __init__(self, infer_k=False, **kwargs):
        super().__init__(**kwargs)
        self.infer_k = infer_k

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
        # latent_vals = []
        # pair_latent_vals = []
        with trange(len(data_loader), **kwargs) as t:
            for _, data_out in enumerate(data_loader):
                data = data_out[0]
                shared_idcs = data_out[2] if not self.infer_k else None
                iter_out = self._train_iteration(
                    samples=data[0], paired_samples=data[1], shared_idcs=shared_idcs)
                for key, item in iter_out['to_log'].items():
                    to_log[key].append(item)
                # latent_vals.append(data_out[1][0].detach().cpu().numpy())
                # pair_latent_vals.append(data_out[1][1].detach().cpu().numpy())
                t.set_postfix(loss=iter_out['loss'],
                              norm_loss=iter_out['loss'] / len(data))
                t.update()
        # latent_vals = np.vstack(latent_vals)
        # pair_latent_vals = np.vstack(pair_latent_vals)
        # from IPython import embed; embed()
        # corrs = {}
        # pair_corrs = {}
        # for i in range(6):
        #     for j in range(6):
        #         if i != j:
        #             corrs[(i,j)] = np.corrcoef(latent_vals[..., i], latent_vals[..., j])[1, 0]
        #             pair_corrs[(i,j)] = np.corrcoef(pair_latent_vals[..., i], pair_latent_vals[..., j])[1, 0]
        # from IPython import embed; embed()
        return {key: np.mean(item) for key, item in to_log.items()}

    def _train_iteration(self, samples, paired_samples, shared_idcs=None):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        samples: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        paired_samples: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        shared_idcs: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        """
        samples = samples.to(self.device)
        paired_samples = paired_samples.to(self.device)

        if self.loss_f.mode == 'post_forward':
            model_out = self.model(samples)
            paired_model_out = self.model(paired_samples)
            paired_model_out = {
                f'paired_{key}': item for key, item in paired_model_out.items()}
            
            inputs = {
                'data': samples, 
                'paired_data': paired_samples,
                'shared_idcs': shared_idcs,
                'is_train': self.model.training, 
                **model_out,
                **paired_model_out
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
                'paired_data': paired_samples,
                'shared_idcs': shared_idcs,
                'is_train': self.model.training
            }
            loss_out = self.loss_f(**inputs)
            loss = loss_out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()       
        elif self.loss_f.mode == 'optimizes_internally':
            # for losses that use multiple optimizers (e.g. Factor)
            loss_out = self.loss_f(samples, self.model, self.optimizer)
            loss = loss_out['loss']

        return {'loss': loss.item(), 'to_log': loss_out['to_log']}
