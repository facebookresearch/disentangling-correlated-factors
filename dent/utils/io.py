# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re

import fastargs
import numpy as np
import pathlib
import torch

import dent.models
import utils

CHECKPOINT = "chkpt.pth.tar"
META_FILENAME = "specs.json"

def save_checkpoint(store_dict, directory, filename=CHECKPOINT):
    """
    Save a model and corresponding metadata.

    Parameters
    ----------
    model : nn.Module
        Model.

    directory : str
        Path to the directory where to save the data.

    metadata : dict
        Metadata to save.
    """
    metadata = utils.get_config_dict(fastargs.get_current_config())
    save_metadata(metadata, directory)
    store_dict.update({'metadata': metadata})
    path_to_model = os.path.join(directory, filename)
    torch.save(store_dict, path_to_model)

def load_from_checkpoint(chkpt_data, device, overwrite=True):
    """Load a trained model.

    Parameters
    ----------
    chkpt_data : dict
        comprises model weights & metadata.

    device : torch.device
        Target device on which model is placed.
    """
    metadata = chkpt_data['metadata']
    if overwrite:
        utils.overwrite_config(metadata)
    model = dent.model_select(
        device=device,
        name=chkpt_data['metadata']['model.name'],
        img_size=chkpt_data['metadata']['data.img_size']).to(device)
    model.load_state_dict(chkpt_data['model'], strict=False)
    model.eval()
    return model

def load_checkpoint(directory, device, chkpt_name=CHECKPOINT, overwrite=True):
    """Load a trained model.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU is available.
    """
    chkpt_data = torch.load(directory/chkpt_name, map_location=device)
    metadata = chkpt_data['metadata']
    if overwrite:
        utils.overwrite_config(metadata)
    model = dent.model_select(device=device,
                              name=metadata['model.name'],
                              img_size=metadata['data.img_size'],
                              **chkpt_data['metadata']).to(device)
    model.load_state_dict(chkpt_data['model'], strict=False)
    model.eval()
    return model, chkpt_data, fastargs.get_current_config()

def load_metadata(directory, chkpt_name=None, meta_name=META_FILENAME):
    """Load the metadata of a training directory.
    """
    if chkpt_name is None:
        path_to_metadata = os.path.join(directory, meta_name)
        with open(path_to_metadata) as metadata_file:
            metadata = json.load(metadata_file)
    else:
        metadata = torch.load(os.path.join(directory, chkpt_name))['metadata']
    return metadata

def save_metadata(metadata, directory, filename=META_FILENAME, **kwargs):
    """Load the metadata of a training directory.

    Parameters
    ----------
    metadata:
        Object to save

    directory: string
        Path to folder where to save model. For example './experiments/mnist'.

    kwargs:
        Additional arguments to `json.dump`
    """
    path_to_metadata = os.path.join(directory, filename)
    #Convert pathlib.PosixPaths to str, as they are not serializable.
    for key in metadata.keys():
        if isinstance(metadata[key], pathlib.PosixPath):
            metadata[key] = str(metadata[key])
    #Dumb data into human-readable json.
    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


def load_checkpoints(directory, device):
    """Load all chechpointed models.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU .
    """
    checkpoints = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            results = re.search(r'.*?-([0-9].*?).pt', filename)
            if results is not None:
                epoch_idx = int(results.group(1))
                model = load_checkpoint(root, device, filename=filename)
                checkpoints.append((epoch_idx, model))

    return checkpoints
