"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import pathlib
import os
from uuid import uuid4

import fastargs
from fastargs.decorators import param
import numpy as np
import pickle as pkl

import utils

WANDB_MODES = ['run', 'offline', 'dryrun', 'off']

class DataLogger(object):
    """Main data-logging object, logs both offline and online to W&B.
    """

    @param('log.wandb_mode')
    @param('log.wandb_key')
    @param('log.wandb_allow_val_change')
    def __init__(
        self, write_dir, uid=None, wandb_mode='run', wandb_key=None, wandb_allow_val_change=False, **kwargs):
        self.write_dir = write_dir
        project, group, run_name = self.write_dir.parts[-3:]
        if wandb_mode not in WANDB_MODES:
            err = 'Incorrect value log.wandb_mode = {}. Please choose {}'
            raise ValueError(err.format(wandb_mode, WANDB_MODES))

        #Initialize W&B.
        self.use_wandb = wandb_mode != 'off'
        if self.use_wandb:
            import wandb
            wandb.Api({'base_url': 'https://api.wandb.ai'})
            _ = os.system(f'wandb login {wandb_key}')
            os.environ['WANDB_API_KEY'] = wandb_key
            os.environ['WANDB_MODE'] = wandb_mode
            wandb_dir = write_dir
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(
                id=uid,
                resume='allow',
                project=project,
                group=group,
                name=run_name,
                dir=wandb_dir
            )
            config = fastargs.get_current_config()
            config_dict = utils.get_config_dict(config)
            wandb.config.update(
                config_dict, allow_val_change=wandb_allow_val_change)

        #Initialize offline logging.
        self.log_files = {}
        self.call_iter = 0

    def log_summary(self, log_summary, log_key=None, include_as_config=True):
        if self.use_wandb:
            import wandb
            config_dict = {}
            for key, item in log_summary.items():
                if not (isinstance(item, np.ndarray) or isinstance(item, list)) or isinstance(item, tuple):
                    sp_item = item if item is not None else -1
                    wandb.run.summary[key] = sp_item
                    config_dict[key] = sp_item
            if include_as_config:
                wandb.config.update(config_dict)
        pkl.dump(log_summary, open(self.write_dir/'{}log_summary.pkl'.format(log_key + '_' if log_key is not None else ''), 'wb'))                

    def log(self, log_dict, is_img='none', log_key=None, commit=True):
        if isinstance(is_img, str):
            if is_img == 'none':
                is_img = {key: False for key in log_dict.keys()}
            elif 'all' in is_img:
                exclusion_needed = '-' in is_img
                if exclusion_needed:
                    exclude = is_img.split('-')[1:]
                else:
                    exclude = []
                is_img = {key: True for key in log_dict.keys()}
                for ex_key in exclude:
                    is_img[ex_key] = False
            else:
                raise ValueError(f'is_img has to be [none, all, <dict>], but received {is_img}.')

        self.call_iter += 1
        if self.use_wandb:
            import wandb
            log_dict['call_iter'] = self.call_iter
            is_img['call_iter'] = False
        
        if log_key:
            log_dict = {f'{log_key}_{key}':item for key, item in log_dict.items()}
            is_img = {f'{log_key}_{key}':item for key, item in is_img.items()}

        if self.use_wandb:
            wandb.log(
                {
                    key: wandb.Image(item) if is_img[key] else item for key, item in log_dict.items()
                }, commit=commit)

        for key, item in log_dict.items():
            if not is_img[key]:
                if key not in self.log_files:
                    self.log_files[key] = self.write_dir/f'{key}_log.csv'
                    with open(self.log_files[key], 'a') as file:
                        file.writelines(f'index,{key}\n')
                with open(self.log_files[key], 'a') as file:
                    file.writelines(f'{self.call_iter},{item}\n')


