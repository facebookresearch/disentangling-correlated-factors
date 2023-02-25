# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
from torch import nn

class Encoder(nn.Module):

    def __init__(self, img_size, latent_dim=10, dist_nparams=2):
        r"""MLP Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        dist_nparams : int
            number of distribution statistics to return

        References:
            [1] Chen et al. "Isolating Sources of Disentanglement in Variational Autoencoders"
        """
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.hidden_dim = 1200
        self.input_dim = np.prod(self.img_size[-3:])
        # Layer parameters
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act = nn.ReLU(inplace=True)
        # Fully connected layers for mean and variance
        self.dist_nparams = dist_nparams
        self.dist_statistics = nn.Linear(self.hidden_dim, self.latent_dim * self.dist_nparams)
        

    def forward(self, x):
        h = x.view(-1, self.input_dim)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        dist_statistics = self.dist_statistics(h)
        return {'stats_qzx': dist_statistics.view(-1, self.latent_dim, self.dist_nparams)}
