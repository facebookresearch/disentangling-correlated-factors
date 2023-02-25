# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
from torch import nn


class Decoder(nn.Module):

    def __init__(self, img_size, latent_dim=10):
        r"""Decoder of the model proposed utilised in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Locatello et al. "Weakly-Supervised Disentanglement without Compromises" 
            arXiv preprint https://arxiv.org/abs/2002.02886.
        """
        super(Decoder, self).__init__()

        # Layer parameters
        kernel_size = 4
        self.img_size = img_size
        # Shape required to start transpose convs
        n_chan = self.img_size[0]
        self.img_size = img_size
        self.reshape = (64, kernel_size, kernel_size)

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, 256)
        self.lin2 = nn.Linear(256, np.prod(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        self.convT1 = nn.ConvTranspose2d(64, 64, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(32, 32, kernel_size, **cnn_kwargs)        
        self.convT4 = nn.ConvTranspose2d(32, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        x = torch.relu(self.convT3(x))
        x = torch.sigmoid(self.convT4(x))

        return {'reconstructions': x}
