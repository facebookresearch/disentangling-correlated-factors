"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse

import submitit

parser = argparse.ArgumentParser()
parser.add_argument('--jobs', '-j', type=str, required=True)
parser.add_argument('--jobname', '-jn', type=str, default='job')
#SLURM-specific paremters
parser.add_argument('--partition', '-p', type=str, default='learnfair')
parser.add_argument('--slurmdir', '-sd', type=str, default='/checkpoint/karroth/jobs')
parser.add_argument('--cpus_per_task', '-nc', type=int, default=8)
parser.add_argument('--gpus_per_node', '-ng', type=int, default=1)
parser.add_argument('--timeout_min', '-to', type=int, default=4000)
args = parser.parse_args()

def read_file(path):
    with open(path) as f:
        read_lines = f.readlines()

    temp_lines = []
    for read_line in read_lines:
        if '#' not in read_line and read_line != '\n':
            temp_lines.append(read_line.rstrip())
    
    lines = []
    base_line = ''
    for temp_line in temp_lines:
        if temp_line[:6] == 'python':
            if base_line != '':
                lines.append(base_line)
            base_line = temp_line
        else:
            base_line += temp_line
    lines.append(base_line)

    return lines

#Read Jobs to submit
jobs_to_submit = read_file(args.jobs)

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

print(f'Submitting from : {args.jobs}')
print(f'Number of submitted jobs: {len(jobs_to_submit)}')

#Submit jobs as arrays to avoid overloading the scheduler.
with executor.batch():
    for job_str in jobs_to_submit:
        # Submit each job.
        job = job_str.split(' ')
        job = [x.replace('\\', '') for x in job]
        func = submitit.helpers.CommandFunction(job, verbose=True)
        job = executor.submit(func)

print("Finished scheduling!")