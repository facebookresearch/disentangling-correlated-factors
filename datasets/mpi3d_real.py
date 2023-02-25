# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess

import numpy as np
import sklearn.preprocessing
import torchvision

import datasets
import datasets.base

class MPI3D_real(datasets.base.DisentangledDataset):
    """MPI3D Dataset as part of the NeurIPS 2019 Disentanglement Challenge.

    A data-set which consists of over one million images of physical 3D objects with seven factors of variation, 
    such as object color, shape, size and position.

    Notes
    -----
    - Link : https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    """
    urls = {
        "train":
        "https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz"
    }
    files = {"train": "mpi3d_real.npz"}
    lat_names = ('objCol', 'objShape', 'objSize', 'cameraHeight', 'backCol', 'posX', 'posY')
    lat_sizes = np.array([6, 6, 2, 3, 3, 40, 40])
    img_size = (3, 64, 64)
    background_color = datasets.COLOUR_WHITE
    lat_values = {
        'objCol': np.linspace(0, 1, 6),
        'objShape': np.linspace(0, 1, 6),
        'objSize': np.linspace(0, 1, 2),
        'cameraHeight': np.linspace(0, 1, 3),
        'backCol': np.linspace(0, 1, 3),
        'posX': np.linspace(0, 1, 40),
        'posY': np.linspace(0, 1, 40)
    }

    def __init__(self, root='data/mpi3d_real/', **kwargs):
        super().__init__(root, [torchvision.transforms.ToTensor()], **kwargs)
        self.logger.info('Loading MPI3D [REAL] (~12GB) - this can take some time...')
        data = np.load(self.train_data)

        self.imgs = data['images']
        lat_values = []
        for col in self.lat_values['objCol']:
            for shp in self.lat_values['objShape']:
                for siz in self.lat_values['objSize']:
                    for hgt in self.lat_values['cameraHeight']:
                        for bck in self.lat_values['backCol']:
                            for x in self.lat_values['posX']:
                                for y in self.lat_values['posY']:
                                    lat_values.append([col, shp, siz, hgt, bck, x, y])
        self.lat_values = lat_values
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
