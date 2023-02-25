# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import tarfile

import torchvision

import datasets

class Chairs(torchvision.datasets.ImageFolder):
    """Chairs Dataset from [1].
    Notes
    -----
    - Link : https://www.di.ens.fr/willow/research/seeing3Dchairs
    Parameters
    ----------
    root : string
        Root directory of dataset.
    References
    ----------
    [1] Aubry, M., Maturana, D., Efros, A. A., Russell, B. C., & Sivic, J. (2014).
        Seeing 3d chairs: exemplar part-based 2d-3d alignment using a large dataset
        of cad models. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 3762-3769).
    """
    urls = {"train": "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"}
    files = {"train": "chairs_64"}
    img_size = (1, 64, 64)
    background_color = datasets.COLOUR_WHITE

    def __init__(self, root='data/chairs', logger=logging.getLogger(__name__)):
        self.root = root
        self.logger = logger
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

        super().__init__(self.train_data, transform=self.transforms)

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'chairs.tar')
        os.makedirs(self.root)
        subprocess.check_call(["curl", type(self).urls["train"],
                               "--output", save_path])

        self.logger.info("Extracting Chairs ...")
        tar = tarfile.open(save_path)
        tar.extractall(self.root)
        tar.close()
        os.rename(os.path.join(self.root, 'rendered_chairs'), self.train_data)

        os.remove(save_path)

        self.logger.info("Preprocessing Chairs ...")
        datasets.preprocess(
            os.path.join(self.train_data, '*/*'),  # root/*/*/*.png structure
            size=type(self).img_size[1:],
            center_crop=(400, 400))
