# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torchvision

import datasets

class MNIST(torchvision.datasets.MNIST):
    """Mnist wrapper. Docs: `datasets.MNIST.`"""
    img_size = (1, 32, 32)
    background_color = datasets.COLOUR_BLACK

    def __init__(self, root='data/mnist', **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=torchvision.transforms.Compose([
                             torchvision.transforms.Resize(32),
                             torchvision.transforms.ToTensor()
                         ]))
