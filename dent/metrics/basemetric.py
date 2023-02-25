"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import abc

class BaseMetric(abc.ABC):
    def __init__(self, device):
        self.device = device
        self.available_modes = ['full', 'batch', 'instance']
        
    @abc.abstractproperty
    def _requires(self):
        """Return elements required to compute specific metric.
        """

    @abc.abstractproperty
    def _mode(self):
        """Return if metric is computed as batch average or over full dataset.

        Should return a mode in self.available_modes.
        """

    @abc.abstractmethod
    def __call__(self):
        """Compute metric.
        """