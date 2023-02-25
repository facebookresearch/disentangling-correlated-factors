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
import pandas as pd
import pickle
import torch
import tqdm

import datasets
import dent
import utils

#%%
#Get input arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    '--chkpt_folder', '-c', type=str, required=True, 
    help='Path to folder containing checkpoint file.')
parser.add_argument(
    '--eval_metrics', nargs='+', type=str, default=['fos', 'kld', 'dci_d', 'mig', 'modularity_d', 'sap_d', 'reconstruction_error'])
parser.add_argument(
    '--dont_group_by_seed', action='store_true', help='If set, will NOT group results by seeds.'
)
parser.add_argument(
    '--batch_size', type=int, default=2048, help='Batchsize to use when computing image embeddings.'
)
opt = parser.parse_args()
opt.group_by_seed = not opt.dont_group_by_seed

#%%
### Allow for evaluation of single checkpoint.
chkpt_files = []
if opt.group_by_seed:
    opt.chkpt_folder = opt.chkpt_folder.replace(' ', '')
    if os.path.isdir(opt.chkpt_folder):
        for chkpt_folder in tqdm.tqdm(os.listdir(opt.chkpt_folder), desc='Aggregating checkpoint paths...'):
            chkpt_folders_path = os.path.join(opt.chkpt_folder, chkpt_folder)
            seed_chkpt_path = os.path.join(chkpt_folders_path, 'chkpt.pth.tar')
            if os.path.exists(seed_chkpt_path):
                chkpt_files.append(seed_chkpt_path)
    else:
        raise Exception()
else:
    chkpt_folders_path = os.path.join(opt.chkpt_folder, 'chkpt.pth.tar')
    assert os.path.exists(chkpt_folders_path), f'Checkpoint {chkpt_folders_path} does not exist!'
    chkpt_files = [chkpt_folders_path]
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
device = torch.device('cuda')
project_seeds = []
result_summary = {}
failed_checkpoints = []
max_num_seeds = -1
project = 'None'
if max_num_seeds == -1:
    max_num_seeds = len(chkpt_files)
if len(chkpt_files):
    for chkpt_path in tqdm.tqdm(chkpt_files[:max_num_seeds], desc='Evaluating chkpts...'):
        try:
            chkpt = torch.load(chkpt_path)
            metadata = chkpt['metadata']
            config = utils.overwrite_config(metadata)
            project = chkpt_path.split('/')[-3]
            project_seed = config['train.seed']
            
            eval_path = chkpt_path.replace('chkpt.pth.tar', 'eval_metrics_log.csv')
            eval_results = {}
            if os.path.exists(eval_path):
                temp_eval_results = eval(
                    '{' + str(pd.read_csv(eval_path, delimiter=' '
                    )['index,eval_metrics']).split(",{")[-1].split('}\nName')[0] + '}'
                )
                for key in temp_eval_results.keys():
                    if key[:5] != 'train' and key[:4] != 'test':
                        eval_results[f'test_{metric_name_convert(key)}'] = temp_eval_results[key]
                        
            constraints_filepath = config['constraints.file']
            if constraints_filepath == 'none':
                constraints_filepath = None
            correlations_filepath = config['constraints.correlations_file']
            if correlations_filepath == 'none':
                correlations_filepath = None

            correlated_train_loader, _ = datasets.get_dataloaders(
                shuffle=True, 
                device=device, 
                batch_size=opt.batch_size,
                return_pairs='pairs' in config['train.supervision'], 
                root=config['data.root'],
                k_range=config['data.k_range'], 
                constraints_filepath=constraints_filepath,
                correlations_filepath=correlations_filepath)

            train_loader, _ = datasets.get_dataloaders(
                dataset=config["data.name"], shuffle=False, batch_size=opt.batch_size,
                num_workers=8, device=device, root=f'data/{config["data.name"]}'
            )
            
            #Initialize model.
            model = dent.model_select(device, name=config['model.name'], img_size=config['data.img_size'])
            model.load_state_dict(chkpt['model'])
            _ = model.to(device)    
            _ = model.eval()
            
            #Metrics to compute.
            metrics_to_compute = {
                'train': [metric for metric in opt.eval_metrics if f'train_{metric}' not in eval_results],
                'test': [metric for metric in opt.eval_metrics if f'test_{metric}' not in eval_results]
            }
            #Extract embeddings.
            with torch.no_grad():
                for mode, dataloader in zip(['train', 'test'], [correlated_train_loader, train_loader]):
                    #Initialize metric computer.
                    metric_group = dent.metrics.utils.MetricGroup(device=device, metric_names=metrics_to_compute[mode])
                    metric_out = metric_group.compute(dataloader, model, disable_tqdm=True)
                    aggregated_metrics = {}
                    if metric_out['computed_metrics']:
                        metric_out['computed_metrics'] = {key: item.item() for key, item in metric_out['computed_metrics'].items()}
                        aggregated_metrics.update({f'full_{key}': item for key, item in metric_out['computed_metrics'].items()})
                    if metric_out['computed_instance_metrics']:
                        metric_out['computed_instance_metrics'] = {key: item.item() for key, item in metric_out['computed_instance_metrics'].items()}            
                        aggregated_metrics.update({f'instance_{key}': item for key, item in metric_out['computed_instance_metrics'].items()})
                    aggregated_metrics = {f'{mode}_{metric_name_convert(key)}': item for key, item in aggregated_metrics.items()}
                    eval_results.update(aggregated_metrics)
                    
            for metric_name, metric_value in eval_results.items():
                if metric_name not in result_summary:
                    result_summary[metric_name] = []
                result_summary[metric_name].append(metric_value)
            project_seeds.append(project_seed)     
        except:
            failed_checkpoints.append(chkpt_path)
    results = {
        'project': project, 
        'seeds': project_seeds, 
        'results': result_summary, 
        'failed': failed_checkpoints
    }

    os.makedirs('posthoc_evals', exist_ok=True)
    pickle.dump(
        results, open(f'posthoc_evals/metrics__{opt.chkpt_folder.replace("/", "__")}.pkl', 'wb')
    )
else:
    print(f'No chkpt files available in {opt.chkpt_folder}!')
