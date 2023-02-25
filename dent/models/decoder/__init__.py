# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

AVAILABLE_DECODERS = ['burgess', 'chen_mlp', 'locatello', 'montero_small', 'montero_large', 'sbd']

def select(name):
    if name not in AVAILABLE_DECODERS:
        raise ValueError(
            'No decoder [{name}] available. Please choose from {AVAILABLE_DECODERS}.')
    
    if name == 'chen_mlp':
        from .chen_mlp import Decoder
    if name == 'burgess':
        from .burgess import Decoder
    if name == 'locatello':
        from .locatello import Decoder
    if name == 'montero_small':
        from .montero_small import Decoder
    if name == 'montero_large':
        from .montero_large import Decoder
    if name == 'sbd':
        from .sbd import Decoder

    return Decoder
