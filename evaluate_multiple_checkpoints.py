"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
#%%
#Load Libraries.
import os

import argparse
import numpy as np
import pandas as pd
import submitit
import tqdm

#%%
#Get input arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    '--chkpt_folder', '-c', type=str, required=True, 
    help='Path to folder containing checkpoint file.')
parser.add_argument(
    '--eval_metrics', nargs='+', type=str, default=['fos', 'kld', 'dci_d', 'mig', 'modularity_d', 'sap_d', 'reconstruction_error'])
parser.add_argument(
    '--job_name', type=str, default='job', help=''
)
parser.add_argument(
    '--slurm_partition', '-p', type=str, default='partition_name', help=''
)
parser.add_argument(
    '--timeout_min', type=int, default=60, help=''
)
opt = parser.parse_args()

#%%
### Allow for evaluation of single checkpoint.
chkpt_files = []
for chkpt_folder in tqdm.tqdm(os.listdir(opt.chkpt_folder), desc='Aggregating checkpoint paths...'):
    chkpt_folders_path = os.path.join(opt.chkpt_folder, chkpt_folder)
    if os.path.isdir(chkpt_folders_path):
        chkpt_files.append(chkpt_folders_path)
chkpt_files = sorted(chkpt_files)

#%%
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

#%%
### (Re-)computing selected metrics for train/test splits.
files_to_check = np.arange(len(chkpt_files))
project_results = {}
failed_checkpoints = []    
chkpt_files_to_check = [chkpt_files[i] for i in files_to_check]

log_folder = "/checkpoint/karroth/jobs/%j"
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    name=opt.job_name,
    timeout_min=opt.timeout_min,
    slurm_partition=opt.slurm_partition,
    tasks_per_node=1,
    nodes=1,
    gpus_per_node=1,
    cpus_per_task=8
)

jobs = []
with executor.batch():
    with tqdm.tqdm(total=len(chkpt_files_to_check), desc='Submitting runs...') as progress_bar:
        for chkpt_file in chkpt_files_to_check:
            cmds = ["python", "evaluate_single_checkpoint.py", "-c", chkpt_file]
            func = submitit.helpers.CommandFunction(cmds, verbose=True)
            job = executor.submit(func)
            jobs.append(job)
            progress_bar.update(1)