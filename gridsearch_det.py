"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import copy
import itertools as it

import numpy as np
import yaml
import submitit
import tqdm
import os

# hack to set default slurmdir based on environment variables
# DEFAULT_SLURMDIR = '/checkpoint/karroth/jobs'
USER = os.getenv('USER', 'karroth')
DEFAULT_SLURMDIR = os.getenv('SLURMDIR', '/checkpoint/' + USER + '/jobs')

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', '-cfg', type=str, required=True)
parser.add_argument('--gridfile', '-g', type=str, required=True)
parser.add_argument('--overwrite', '-ov', nargs='+', type=str, default=[])
parser.add_argument('--script', '-s', type=str, default='base_main.py')
parser.add_argument('--jobname', '-j', type=str, default='job')
parser.add_argument(
    '--dont_shorten_group', '-dsg', action='store_true', 
    help='If set: Dont shorten log group name (but can potentially exceed max. char limit for W&B.')
#SLURM-specific paremters
parser.add_argument('--partition', '-p', type=str, default='learnlab')
parser.add_argument('--slurmdir', '-sd', type=str, default=DEFAULT_SLURMDIR)
parser.add_argument('--cpus_per_task', '-nc', type=int, default=8)
parser.add_argument('--gpus_per_node', '-ng', type=int, default=1)
parser.add_argument('--timeout_min', '-to', type=int, default=300)
#TODO: FINISH MULTISUBMIT
args = parser.parse_args()

#Parse yaml to convert 'start-end' to [start, start+1, ..., end].
def dict_generator(indict, pre=None):
    #Taken from https://stackoverflow.com/a/12507546.
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in dict_generator(v, pre + [key]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield pre + [indict]

def convert(dict_list, is_grid=True):
    command_collect = {}
    for command in dict_list:
        command, value = '.'.join(command[:-1]), command[-1]
        if is_grid:
            #If value is str (and not a search list), convert to corresponding list.
            if isinstance(value, str):
                if '-' in value:
                    start, end = [int(x) for x in value.split('-')]
                    value = list(range(start, end+1))
            if command not in command_collect:
                command_collect[command] = []
            if not isinstance(value, list):
                value = [value]
            command_collect[command].extend(value)
        else:
            if command in command_collect:
                if not isinstance(command_collect[command], list):
                    command_collect[command] = [command_collect[command]]
                command_collect[command].extend([value])
            else:        
                command_collect[command] = value
    return command_collect

#Load Gridsearch and Config YAML.
with open(args.gridfile) as gridfile:
    yaml_file = yaml.load(gridfile, Loader=yaml.FullLoader)
with open(args.config_file) as gridfile:
    config_file = yaml.load(gridfile, Loader=yaml.FullLoader)

base_config = convert(dict_generator(config_file), is_grid=False)
grid_items_list_coll = []
grid_keys_coll = []
grid_commands_coll = []
for x in yaml_file:
    new_grid_commands = convert(dict_generator(x), is_grid=True)
    grid_elements = []
    grid_keys = list(new_grid_commands.keys())
    grid_commands = ['--' + key + '=' for key in grid_keys]
    grid_items_list = list(new_grid_commands.values())
    grid_items_list = list(it.product(*grid_items_list))
    grid_keys_coll.append(grid_keys)
    grid_items_list_coll.append(grid_items_list)
    grid_commands_coll.append(grid_commands)

#Initialize Submitter.
executor = submitit.AutoExecutor(folder=args.slurmdir)
executor.update_parameters(
    name=args.jobname,
    timeout_min=args.timeout_min,
    slurm_partition=args.partition,
    tasks_per_node=1,
    nodes=1,
    gpus_per_node=args.gpus_per_node,
    cpus_per_task=args.cpus_per_task
)

#Base command to call that is overwritten based on args.overwrite.
base_cmd = [
    "python", 
    args.script, 
    f"--config-file={args.config_file}"
]

print(f'Scheduling Gridsearch for {args.config_file}')
if len(args.overwrite):
    print(f'Overwriting config with: {args.overwrite}')
counts = len([x for y in grid_items_list_coll for x in y])
print(f'Submitting {counts} jobs.')

# Submit gridjobs as arrays to avoid overloading the scheduler.
jobs = []
with tqdm.tqdm(total=counts) as progress_bar:
    with executor.batch():
        for grid_keys, grid_commands, grid_items_list in zip(grid_keys_coll, grid_commands_coll, grid_items_list_coll):
            for grid_items in grid_items_list:
                aux_cmds = []
                overwrite_list = copy.deepcopy(args.overwrite)
                for i, grid_item in enumerate(grid_items):
                    aux_cmds.append(grid_commands[i]+f'{grid_item}')

                # Adjust the name of the logging group based on the gridded hyperparameter pair.
                group_cmd = None
                cfg_has_group = any(['log.group' in x for x in base_config.keys()])
                if cfg_has_group:
                    group_cmd = f'--log.group={base_config["log.group"]}'
                set_new_group = any(['log.group' in x for x in overwrite_list])
                if set_new_group:
                    group_idx = np.where(['log.group' in x for x in overwrite_list])[0][0]
                    group = overwrite_list[group_idx]
                    group_cmd = f'--{group}'
                    overwrite_list.pop(group_idx)

                overwrite_cmds = [f'--{x}' for x in overwrite_list]

                if args.dont_shorten_group:
                    #Append auxiliary commands to group-name.
                    group_cmd = group_cmd + '__grid' + ''.join(x for x in aux_cmds if '.seed' not in x)
                    #Append overwrite commands to group-name
                    if len(overwrite_cmds):
                        group_cmd = group_cmd + '__ov' + ''.join(x for x in overwrite_cmds if '.seed' not in x)
                else:
                    #Append auxiliary commands to group-name.
                    group_cmd = group_cmd + '--grid_' + '_'.join(x.split('=')[-1].split(':')[-1] for x in aux_cmds if '.seed' not in x)
                    #Append overwrite commands to group-name
                    if len(overwrite_cmds):
                        group_cmd = group_cmd + '--ov_' + '_'.join(x.split('=')[-1] for x in aux_cmds if '.seed' not in x)
                group_cmds = [] if group_cmd is None else [group_cmd]

                if len(group_cmd[12:]) > 128:
                    raise Exception(f'W&B group name [{group_cmd}] exceeds 128 character limit!')

                cmds = base_cmd + aux_cmds + group_cmds + overwrite_cmds
                for i in range(len(cmds)):
                    if 'log.group' in cmds[i]:
                        cmds[i] = cmds[i].replace(':', '-')
                        cmds[i] = cmds[i].replace('/', '-')
                #Submit each grid-job.
                func = submitit.helpers.CommandFunction(cmds, verbose=True)
                job = executor.submit(func)
                jobs.append(job)
                progress_bar.update(1)

print("Finished scheduling!")
