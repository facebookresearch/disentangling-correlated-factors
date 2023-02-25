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
        kernel_size = 5
        self.img_size = img_size

        # Convolutional layers
        cnn_kwargs = dict(stride=1, padding=2)
        self.conv1 = nn.Conv2d(latent_dim + 2, 64, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(64, 64, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(64, 64, kernel_size, **cnn_kwargs)        
        self.conv4 = nn.Conv2d(64, 64, kernel_size, **cnn_kwargs)        
        self.conv5 = nn.Conv2d(64, self.img_size[0], kernel_size, **cnn_kwargs)        
        
        # XY Mesh.
        x, y = np.meshgrid(
            np.linspace(-1, 1, self.img_size[-2]),
            np.linspace(-1, 1, self.img_size[-1]))
        x = x.reshape(self.img_size[-2], self.img_size[-2], 1)
        y = y.reshape(self.img_size[-1], self.img_size[-1], 1)
        self.xy_mesh = torch.from_numpy(np.concatenate((x,y), axis=-1)).to(torch.float).unsqueeze(0)

    def spatial_broadcast(self, z):
        if self.xy_mesh.device != z.device:
            self.xy_mesh = self.xy_mesh.to(z.device)
        z_sb = torch.tile(z, (1, np.prod(self.img_size[-2:])))
        z_sb = z_sb.reshape(z.size(0), *self.img_size[-2:], z.size(-1))
        return torch.cat((z_sb, torch.tile(self.xy_mesh, (z.size(0), 1, 1, 1))), dim=3).permute(0, 3, 2, 1)

    def forward(self, z):
        # Apply Spatial Broadcasting.
        x = self.spatial_broadcast(z)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.sigmoid(self.conv5(x))

        return {'reconstructions': x}
