# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess

import h5py
import numpy as np
import sklearn.preprocessing
import torchvision

import datasets
import datasets.base

class Shapes3D(datasets.base.DisentangledDataset):
    """Shapes3D Dataset from [1].

    3dshapes is a dataset of 3D shapes procedurally generated from 6 ground truth independent 
    latent factors. These factors are floor colour (10), wall colour (10), object colour (10), size (8), type (4) and azimuth (15). 
    All possible combinations of these latents are present exactly once, generating N = 480000 total images.

    Notes
    -----
    - Link : https://storage.googleapis.com/3d-shapes
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Hyunjik Kim, Andriy Mnih (2018). Disentangling by Factorising.

    """
    urls = {
        "train":
        "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
    }
    files = {"train": "3dshapes.h5"}
    lat_names = ('floorCol', 'wallCol', 'objCol', 'objSize', 'objType', 'objAzimuth')
    lat_sizes = np.array([10, 10, 10, 8, 4, 15])
    img_size = (3, 64, 64)
    background_color = datasets.COLOUR_WHITE
    lat_values = {
        'floorCol': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        'wallCol': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        'objCol': np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        'objSize': np.linspace(0.75, 1.25, 8),
        'objType': np.array([0., 1., 2., 3.]),
        'objAzimuth': np.linspace(-30., 30., 15)
    }

    def __init__(self, root='data/shapes3d/', **kwargs):
        super().__init__(root, [torchvision.transforms.ToTensor()], **kwargs)

        # with h5py.File(self.train_data, 'r') as dataset:
        #     self.imgs = dataset['images'][()]
        #     self.lat_values = dataset['labels'][()]
        self.imgs = np.load(self.train_data.replace('.h5', '_imgs.npy'))
        self.lat_values = np.load(self.train_data.replace('.h5', '_labs.npy'))
        self.lat_values = sklearn.preprocessing.minmax_scale(self.lat_values)

        if self.subset < 1:
            n_samples = int(len(self.imgs) * self.subset)
            subset = np.random.choice(len(self.imgs), n_samples, replace=False)
            self.imgs = self.imgs[subset]
            self.lat_values = self.lat_values[subset]

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call([
            "curl", "-L",
            type(self).urls["train"], "--output", self.train_data
        ])
        #For faster loading, a numpy copy will be created (reduces loading times by 300% at the cost of more storage).
        with h5py.File(self.train_data, 'r') as dataset:
            imgs = dataset['images'][()]
            lat_values = dataset['labels'][()]
        np.save(self.train_data.replace('.h5', '_imgs.npy'), imgs)
        np.save(self.train_data.replace('.h5', '_labs.npy'), lat_values)
                
    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        return self.transforms(self.imgs[idx]), self.lat_values[idx]
