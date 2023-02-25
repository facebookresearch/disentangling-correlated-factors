"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from fastargs.decorators import param

LOSS_LIST = ['betavae', 'annealedvae', 'vae', 'factorvae', 'betatcvae', 'adagvae', 'factorizedsupportvae', 'factorizedsupporttcvae']
RECON_DISTS = ["bernoulli", "laplace", "gaussian"]

@param('train.loss', 'name')
def select(device, name, **kwargs):
    """Return the correct loss function given the argparse arguments."""
    if name == "betavae":
        from dent.losses.betavae import Loss
        return Loss(**kwargs)
    if name == "vae":
        from dent.losses.betavae import Loss
        return Loss(beta=1, **kwargs)
    if name == "annealedvae":
        from dent.losses.annealedvae import Loss
        return Loss(**kwargs)
    if name == "factorvae":
        from dent.losses.factorvae import Loss
        return Loss(device, **kwargs)
    if name == "betatcvae":
        from dent.losses.betatcvae import Loss
        return Loss(**kwargs)
    if name == "adagvae":
        from dent.losses.adagvae import Loss
        return Loss(**kwargs)
    if name == 'factorizedsupportvae':
        from dent.losses.factorizedsupportvae import Loss
        return Loss(**kwargs)
    if name == 'factorizedsupporttcvae':
        from dent.losses.factorizedsupporttcvae import Loss
        return Loss(**kwargs)
    err = "Unkown loss.name = {}. Possible values: {}"
    raise ValueError(err.format(name, LOSS_LIST))
