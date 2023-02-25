# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import os

from fastargs.decorators import param
import torch
import torchvision

class DisentangledDataset(torch.utils.data.Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    @param('data.subset', 'subset')
    def __init__(self,
                 root,
                 transforms_list=[],
                 logger=logging.getLogger(__name__),
                 subset=1):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = torchvision.transforms.Compose(transforms_list)
        self.logger = logger
        self.subset = subset
        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass
