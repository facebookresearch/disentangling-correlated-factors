"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from .basemetric import BaseMetric

class ReconstructionError(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def _requires(self):
        return ['reconstructions', 'data_samples']

    @property
    def _mode(self):
        return 'instance'

    def __call__(self, reconstructions, data_samples, **kwargs):
        """Mean-squared reconstruction error.
        """
        return torch.mean(
            (reconstructions - data_samples).view(len(data_samples), -1)**2, 
            dim=-1).unbind(-1)