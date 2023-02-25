"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
#%%
#Load libraries.
import logging
import json
import os
os.chdir('/private/home/karroth/Projects/dent/iclr_visualizations')
import sys
sys.path.insert(0, '..')
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm
import wandb

import datasets
import dent
import parameters
import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--chkpt_paths', nargs='+', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--source_group', type=str, required=True)
parser.add_argument('--target_group', type=str, required=True)
parser.add_argument('--preemb', type=str, required=True)
opt = parser.parse_args()

#%%
### Compute correlation transfers.
def correlations_file_convert(dataset_name, group):
    if group == 'none':
        return group
    return f'../grid_searches/avail_correlations.yaml:{dataset_name}_{group}'

def metric_name_convert(metric_name):
    metric_name = metric_name.replace('full_', '')
    metric_name = metric_name.replace('instance_', '')
    metric_name = metric_name.replace('dci_d_disentanglement', 'dci_d')
    metric_name = metric_name.replace('dci_d_informativeness_test_errors', 'dci_i_test_errors')
    metric_name = metric_name.replace('dci_d_informativeness_train_errors', 'dci_i_train_errors')
    metric_name = metric_name.replace('dci_d_informativeness_test_scores', 'dci_i_test_scores')
    metric_name = metric_name.replace('dci_d_informativeness_train_scores', 'dci_i_train_scores')
    metric_name = metric_name.replace('dci_d_completeness', 'dci_c')
    metric_name = metric_name.replace('modularity_d', 'modularity')
    metric_name = metric_name.replace('sap_d', 'sap')
    metric_name = metric_name.replace('fos_fos', 'fos')
    metric_name = metric_name.replace('kld_kld', 'kld')
    return metric_name


# %%
import pickle as pkl
import torch

device = torch.device('cuda')
metrics_to_compute = ['dci_d', 'mig', 'sap_d', 'modularity_d', 'rand_fos', 'rand_kld']
disable_tqdm = False

for chkpt_path in opt.chkpt_paths:
    chkpt = torch.load(chkpt_path)
    metadata = chkpt['metadata']
    if 'inv' in opt.target_group:
        config = utils.insert_config('constraints.correlation_distribution', 'inv_traeuble')    
    config = utils.overwrite_config(metadata)
    project = chkpt_path.split('/')[-3]
    project_seed = config['train.seed']

    savename = f'corr_transfer_results/{opt.preemb}__{opt.dataset_name}__{opt.source_group}__{opt.target_group}_seed-{project_seed}.pkl'
    if not os.path.exists(savename):
        #Initialize model.
        model = dent.model_select(device, name=config['model.name'], img_size=config['data.img_size'])
        model.load_state_dict(chkpt['model'])
        _ = model.to(device)    
        _ = model.eval()

        constraints_filepath = None
        correlations_filepath = correlations_file_convert(opt.dataset_name, opt.target_group.replace('inv', ''))

        correlated_dataloader, _ = datasets.get_dataloaders(
            dataset=opt.dataset_name,
            shuffle=True, 
            device=device, 
            batch_size=2048,
            return_pairs=False, 
            root=f'../data/{opt.dataset_name}',
            k_range=config['data.k_range'], 
            constraints_filepath=constraints_filepath,
            correlations_filepath=correlations_filepath)

        with torch.no_grad():
            #Initialize metric computer.
            metric_group = dent.metrics.utils.MetricGroup(device=device, metric_names=metrics_to_compute)
            metric_out = metric_group.compute(correlated_dataloader, model, disable_tqdm=False)
            aggregated_metrics = {}
            if metric_out['computed_metrics']:
                metric_out['computed_metrics'] = {key: item.item() for key, item in metric_out['computed_metrics'].items()}
                aggregated_metrics.update({f'full_{key}': item for key, item in metric_out['computed_metrics'].items()})
            if metric_out['computed_instance_metrics']:
                metric_out['computed_instance_metrics'] = {key: item.item() for key, item in metric_out['computed_instance_metrics'].items()}            
                aggregated_metrics.update({f'instance_{key}': item for key, item in metric_out['computed_instance_metrics'].items()})
        aggregated_metrics = {metric_name_convert(key): item for key, item in aggregated_metrics.items()}        

        pkl.dump(aggregated_metrics, open(savename, 'wb'))