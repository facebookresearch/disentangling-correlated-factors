# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import tarfile

from fastargs.decorators import param
import numpy as np
from scipy.io import loadmat
import torchvision
import tqdm

import datasets
import datasets.base

class Cars3D(datasets.base.DisentangledDataset):
    """Cars3D Dataset from [1].

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Deep Visual Analogy-Making (https://papers.nips.cc/paper/5845-deep-visual-analogy-making)

    """
    urls = {
        "train":
        "http://www.scottreed.info/files/nips2015-analogy-data.tar.gz"
    }
    files = {"train": "cars3d.npz"}
    lat_names = ('elevation', 'azimuth', 'object_type')
    lat_sizes = np.array([4, 24, 183])
    img_size = (3, 64, 64)
    background_color = datasets.COLOUR_WHITE
    lat_values = {
        'elevation': np.arange(4)/4.,
        'azimuth': np.arange(24)/24.,
        'object_type': np.arange(183)/183.
    }

    def __init__(self, root='data/cars3d/', **kwargs):
        super().__init__(root, [torchvision.transforms.ToTensor()], **kwargs)

        dataset_zip = np.load(self.train_data)
        self.imgs = dataset_zip['images']
        gt_factors = []
        for idx in range(len(self.imgs)):
            gt_factors.append(np.array(np.unravel_index(idx, self.lat_sizes)))
        gt_factors = np.stack(gt_factors, axis=0)
        self.lat_values = gt_factors / (self.lat_sizes.reshape(1, -1).astype(float) - 1)

        if self.subset < 1:
            n_samples = int(len(self.imgs) * self.subset)
            subset = np.random.choice(len(self.imgs), n_samples, replace=False)
            self.imgs = self.imgs[subset]
            self.lat_values = self.lat_values[subset]

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'cars3d.tar.gz')
        os.makedirs(self.root)
        subprocess.check_call([
            "curl", "-L",
            type(self).urls["train"], "--output", save_path
        ])

        with tarfile.open(save_path) as file:
            self.logger.info("Extracting Cars3D ...")
            file.extractall(self.root)

        os.remove(save_path)

        self.logger.info("Loading Cars3D data...")
        aug_root_dir = os.path.join(self.root, 'data')
        #Load images
        images = []
        with open(os.path.join(aug_root_dir, 'cars/list.txt'), 'r') as img_names:
            for name in img_names.readlines():
                img_path = os.path.join(aug_root_dir, f'cars/{name.strip()}.mat')
                images.append(loadmat(img_path)['im'])
        images = np.stack(images, axis=0)
        assert images.shape == (183, 128, 128, 3, 24, 4)
        #Reshape & transpose: (183, 128, 128, 3, 24, 4) -> (4, 24, 183, 128, 128, 3) -> (17568, 128, 128, 3)
        images = images.transpose([5, 4, 0, 1, 2, 3]).reshape([-1, 128, 128, 3])

        #Resize images to 64x64.
        res_images = []
        self.logger.info("Resizing Cars3D from 128x128 -> 64x64...")
        for image in tqdm.tqdm(images, total=len(images), leave=False, desc='Resizing images...'):
            image = torchvision.transforms.functional.to_pil_image(image)
            image = torchvision.transforms.functional.resize(image, [64, 64])
            image = np.array(image)
            if image.ndim == 2:
                image = image[..., None]
            res_images.append(image)
        res_images = np.stack(res_images)

        file_path = os.path.join(self.root, self.files['train'])
        np.savez_compressed(file_path, images=res_images)

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length len(self.factors_to_use), that gives the value of each factor of variation that is included in self.factors_to_use.
        """
        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        return self.transforms(self.imgs[idx]), self.lat_values[idx]
