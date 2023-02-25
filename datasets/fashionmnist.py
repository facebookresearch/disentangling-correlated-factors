"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torchvision

import datasets

class FashionMNIST(torchvision.datasets.FashionMNIST):
    """Fashion Mnist wrapper. Docs: `datasets.FashionMNIST.`"""
    img_size = (1, 32, 32)
    background_color = datasets.COLOUR_BLACK

    def __init__(self, root='data/fashionMnist', **kwargs):
        super().__init__(root,
                         train=True,
                         download=True,
                         transform=torchvision.transforms.Compose([
                             torchvision.transforms.Resize(32),
                             torchvision.transforms.ToTensor()
                         ]))
