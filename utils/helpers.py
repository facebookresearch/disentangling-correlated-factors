# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import configparser
import logging
import os
import pathlib
import random
import shutil

import fastargs
from fastargs import Section, Param
from fastargs.decorators import param
import numpy as np
import torch

import dent.utils.io
import utils.fastargs_types

#-------------- DATA LOGGING HELPERS -----------------
@param('run.restore_from')
@param('run.restore_with_config')
@param('run.restore_to_new')
def set_save_paths_or_restore(
    config, logger, device, restore_from, restore_with_config=False, restore_to_new=False):
    info_dict = {
        'start_from_chkpt': False,
        'chkpt_data': None,
        'read_dir': None,
        'write_dir': None
    }
    if restore_from == 'n/a':
        base_dir = set_save_paths(incr=True)
        logger.info(f'Storing data to [{base_dir}].')
        info_dict['read_dir'] = base_dir
        info_dict['write_dir'] = base_dir
    elif restore_from == 'overwrite':
        base_dir = set_save_paths(overwrite=True)
        logger.info(f'Storing data to [{base_dir}].')
        info_dict['read_dir'] = base_dir
        info_dict['write_dir'] = base_dir
    elif restore_from == 'continue':
        base_dir = set_save_paths(check_for_chkpt=True, incr=True)
        chkpt_path = base_dir/dent.utils.io.CHECKPOINT
        if chkpt_path.exists():
            info_dict['start_from_chkpt'] = True
            info_dict['chkpt_data'] = torch.load(chkpt_path, map_location=device)
            config = overwrite_config(info_dict['chkpt_data']['metadata'])
            logger.info(f'Continuing training from [{chkpt_path}].')
        else:
            logger.info(f'Storing data to [{base_dir}].')
        info_dict['read_dir'] = base_dir
        info_dict['write_dir'] = base_dir
    else:
        info_dict['start_from_chkpt'] = True
        chkpt_path = pathlib.Path(restore_from)
        info_dict['chkpt_data'] = torch.load(chkpt_path)
        base_dir = chkpt_path.parents[0]        
        if restore_with_config:
            config = overwrite_config(info_dict['chkpt_data']['metadata'])            
        logger.info(f'Continuing from chkpt [{chkpt_path}].')
        info_dict['read_dir'] = base_dir
        if restore_to_new:
            info_dict['write_dir'] = utils.set_save_paths(create=False)
        else:
            info_dict['write_dir'] = base_dir

    # Globally include read/write paths into fastargs config.
    config_dict = get_config_dict(config)
    config_dict['log.read_dir'] = info_dict['read_dir']
    config_dict['log.write_dir'] = info_dict['write_dir']
    config = overwrite_config(config_dict)

    return config, info_dict

@param('log.base_dir')
@param('log.group')
@param('log.project')
@param('data.name', 'data_name')
@param('train.seed', 'train_seed')
@param('train.loss', 'train_loss')
@param('train.lr', 'train_lr')
@param('train.batch_size', 'train_batch_size')
@param('model.name', 'model_name')
def set_save_paths(
    base_dir, group, project, data_name, train_seed, train_loss, train_lr,
    train_batch_size, model_name, check_for_chkpt=False, incr=True, overwrite=False,
    create=True):
    base_dir = pathlib.Path(base_dir)
    base_dir /= project

    if group == 'default':
        group = f'data-{data_name}_loss-{train_loss}_model-{model_name}_lr-{train_lr}_bs-{train_batch_size}'

    subgroup = group + f'_s-{train_seed}'
    base_dir /= group

    run_incr = incr
    if check_for_chkpt:
        cond = (base_dir/subgroup).exists()
        cond = cond and 'chkpt.pth.tar' in os.listdir(base_dir)
        run_incr = run_incr and cond

    if run_incr and not overwrite:
        temp_subgroup = subgroup
        count = 0
        while (base_dir/temp_subgroup).exists():
            count += 1
            temp_subgroup = subgroup + f'_{count}'
        subgroup = temp_subgroup

    base_dir /= subgroup
    if create:
        if overwrite and base_dir.exists():
            import shutil
            shutil.rmtree(base_dir)    
        base_dir.mkdir(parents=True, exist_ok=not overwrite)

    return base_dir

def create_safe_directory(directory, logger=None):
    """Create a directory and archive the previous one if already existed."""
    if os.path.exists(directory):
        if logger is not None:
            warn = "Directory {} already exists. Archiving it to {}.zip"
            logger.warning(warn.format(directory, directory))
        shutil.make_archive(directory, 'zip', directory)
        shutil.rmtree(directory)
    os.makedirs(directory)

@param('log.level', 'log_level')
def set_logger(name, log_level):
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s - %(module)s|%(funcName)s (L%(lineno)s): %(message)s',
        "%H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger

#-------------- FASTARGS HELPERS -----------------
def get_config_dict(fastargs_config):
    config_dict = {}
    for path in fastargs_config.entries.keys():
        try:
            value = fastargs_config[path]
            if value is not None:
                config_dict['.'.join(path)] = fastargs_config[path]
        except:
            pass
    return config_dict

def make_config():
    config = fastargs.get_current_config()            
    parser = argparse.ArgumentParser(
        description='Disentangled Representation Learning.')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')

def insert_config(section_handle,
                  value,
                  section_msg='placeholder',
                  value_msg='placeholder'):
    section, handle = section_handle.split('.')
    Section(section, section_msg).params(
        **{
            handle:
            Param(utils.fastargs_types.type_select(value),
                  value_msg,
                  default=value)
        })
    return fastargs.get_current_config()

def overwrite_config(config_dict, section_msgs=[], value_msgs=[]):
    for i, (key, value) in enumerate(config_dict.items()):
        section_msg = 'placeholder'
        value_msg = 'placeholder'
        if len(section_msgs):
            section_msg = section_msgs[i]
        if len(value_msgs):
            value_msg = value_msgs[i]
        insert_config(key, value, section_msg, value_msg)
    return fastargs.get_current_config()


#-------------- TRAINING HELPERS -----------------
@param('scheduler.name')
@param('scheduler.tau')
@param('scheduler.gamma')
def get_scheduler(optimizer, name='none', tau=[], gamma=[]):
    if name == 'none':
        #If no learning rate scheduler is desired, we simply use a constant scheduler.
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0)
    if name == 'multistep':
        if isinstance(gamma, list): gamma = gamma[0]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=tau, gamma=gamma)
    raise ValueError(f'Optimizer option [{name}] not available!')

@param('train.lr')
@param('train.optimizer', 'name')
def get_optimizer(parameters, lr, name='adam'):
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr)

@param('train.seed')
def set_seed(seed, deterministic = False):
    """Set all random seeds."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # if want pure determinism could uncomment below: but slower
        if deterministic:
            torch.backends.cudnn.deterministic = True

@param('run.on_cpu')
@param('run.gpu_id')
def get_device(on_cpu=False, gpu_id=0):
    """Return the correct device"""
    return torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() and not on_cpu else "cpu")

def get_model_device(model):
    """Return the device on which a model is."""
    return next(model.parameters()).device

def get_n_param(model):
    """Return the number of parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nParams = sum([np.prod(p.size()) for p in model_parameters])
    return nParams

