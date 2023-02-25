"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import copy
import glob
import logging
import os

import fastargs
from fastargs.decorators import param
import pickle as pkl
import numpy as np
from PIL import Image
import time
import torch
from tqdm import tqdm
import yaml

import datasets.base
import utils

COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS = [
    'mnist', 'fashionmnist', 'dsprites', 'celeba', 'chairs', 'shapes3d', 'mpi3d', 'mpi3d_real', 'mpi3d_real_complex', 'cars3d', 'smallnorb'
]

def get_dataset(dataset):
    """Selection function to assign a respective dataset to a query string.
    """
    dataset = dataset.lower()

    if dataset not in DATASETS:
        raise NotImplementedError(f'Unknown datasets {dataset}!')

    if dataset == 'celeba':
        import datasets.celeba
        return datasets.celeba.CelebA
    if dataset == 'chairs':
        import datasets.chairs
        return datasets.chairs.Chairs        
    if dataset == 'dsprites':
        import datasets.dsprites
        return datasets.dsprites.DSprites
    if dataset == 'fashionmnist':
        import datasets.fashionmnist
        return datasets.fashionmnist.FashionMNIST
    if dataset == 'mnist':
        import datasets.mnist
        return datasets.mnist.MNIST
    if dataset == 'mpi3d':
        import datasets.mpi3d
        return datasets.mpi3d.MPI3D
    if dataset == 'mpi3d_real':
        import datasets.mpi3d_real
        return datasets.mpi3d_real.MPI3D_real
    if dataset == 'mpi3d_real_complex':
        import datasets.mpi3d_real_complex
        return datasets.mpi3d_real_complex.MPI3D_real_complex
    if dataset == 'shapes3d':
        import datasets.shapes3d
        return datasets.shapes3d.Shapes3D
    if dataset == 'cars3d':
        import datasets.cars3d
        return datasets.cars3d.Cars3D
    if dataset == 'smallnorb':
        import datasets.smallnorb
        return datasets.smallnorb.SmallNORB

def get_img_size(dataset):
    """Return the correct image size."""
    return fastargs.get_current_config()['data.img_size']

def get_background(dataset):
    """Return the image background color."""
    return fastargs.get_current_config()['data.background_color']

@param('data.name', 'dataset')
@param('train.batch_size')
@param('data.num_workers')
def get_dataloaders(
    dataset, shuffle=False, device=torch.device('cuda'), logger=logging.Logger(__name__), root='n/a', 
    batch_size=256, num_workers=4, return_pairs=False, k_range=[1, -1], constraints_filepath=None,
    correlations_filepath=None):
    """A generic data loader

    Parameters
    ----------
    dataset : {"mnist", "fashion", "dsprites", "celeba", "chairs"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = True if device.type != 'cpu' else False
    log_summary = {}

    if root == 'n/a':
        dataset = get_dataset(dataset)(logger=logger)
    else:
        dataset = get_dataset(dataset)(root=root, logger=logger)

    if constraints_filepath is not None and constraints_filepath != 'none':
        constrain_summary = constrain(
            dataset=dataset, constraints_filepath=constraints_filepath)
        log_summary.update({f'constraint.{key}': item for key, item in constrain_summary.items()})

    correlation_sampler = None
    correlation_weights = None
    if correlations_filepath is not None and correlations_filepath != 'none':
        if shuffle:
            correlation_sampler, correlation_weights = provide_correlation_sampler(
                dataset=dataset, correlations_filepath=correlations_filepath)
            # Turn shuffle of when using a custom sampler.
            shuffle = False

    if return_pairs:
        assert_str = f"Can not return sample pairs for {dataset} as "
        "no ground truth factors of variation are given."
        assert hasattr(dataset, 'lat_sizes'), assert_str
        dataset = PairedDataset(
            dataset=dataset, 
            k_range=k_range,
            correlation_weights=correlation_weights)  

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        sampler=correlation_sampler), log_summary


def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    This function was taken from https://github.com/YannDubs/disentangling-vae.

    Parameters
    ----------
    root : string
        Root directory of all images.
    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.
    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.
    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


#------------- CONSTRAINED SETTINGS ---------------
def compare(comp_vals, constr, op, hash_map):
    if op == '==':
        return hash_map[constr]        
    if op == '>=':
        return comp_vals >= constr
    if op == '<=':
        return comp_vals <= constr
    if op == '>':
        return comp_vals > constr
    if op == '<':
        return comp_vals < constr

def extract_constraint_yaml(constraints_filepath):
    name = None
    if ':' in constraints_filepath:
        constraints_filepath, name = constraints_filepath.split(':')

    with open(constraints_filepath) as constraint_file:
        yaml_file = yaml.load(constraint_file, Loader=yaml.FullLoader)
        if name is not None:
            yaml_file = yaml_file[name] 
    connect = 'and' if 'connect' not in yaml_file else yaml_file['connect']
    constraints = yaml_file['constraints']
    repeat = ['none'] if 'repeat' not in yaml_file else yaml_file['repeat']
    return constraints, connect, repeat

@param('constraints.allow_overshoot')
@param('constraints.max_overshoot')
def compute_constrained_elements(dataset, constraints_filepath, allow_overshoot=True, max_overshoot=0.1):
    lat_names = np.array(dataset.lat_names)

    constraints, connect, repeat = extract_constraint_yaml(constraints_filepath)

    all_ops = []
    stop_repeat = False
    base_remove = np.zeros(len(dataset)).astype(bool)

    unique_lat_vals_list = [np.unique(dataset.lat_values[..., i]) for i in range(dataset.lat_values.shape[-1])]

    #PRECACHE using np.unique!
    index_hash = {}
    iterator = tqdm(
        enumerate(unique_lat_vals_list), desc='Creating hashmap...', total=len(unique_lat_vals_list), leave=False)
    for i, unique_lat_vals in iterator:
        index_hash[i] = {}
        for unique_lat_val in unique_lat_vals:
            index_hash[i][unique_lat_val] = (dataset.lat_values[..., i] == unique_lat_val).astype(bool)

    #Iterate over every axis and apply constraints   
    iterm = 0 
    old_count = 0
    not_iterative = False

    if repeat[0] != 'none':
        if 'coverage' in repeat[0]:
            num_samples_to_remove = int(len(dataset.lat_values) * float(repeat[1]))        
        elif repeat[0] == 'count':
            num_samples_to_remove = int(repeat[1])

        #If only single holes are removed, skip the smart selection step below.
        if len(constraints) == 1 and 'all' in constraints:
            if constraints['all'][0] == '==' and constraints['all'][1] == 'random':
                if repeat[0] == 'coverage' or repeat[0] == 'count':
                    remove_idcs = sorted(np.random.choice(len(base_remove), num_samples_to_remove, replace=False))
                elif repeat[0] == 'coverage_with_hullmin':
                    eligible_idcs = np.logical_or(dataset.lat_values == 0., dataset.lat_values == 1.)
                    eligible_idcs = eligible_idcs.sum(axis=-1) >= repeat[2]
                    assert_str = f'Not enough eligible datapoints to provide coverage of {repeat[1] * 100}%!'
                    assert np.sum(eligible_idcs) * 1. / len(eligible_idcs) > repeat[1], assert_str
                    remove_idcs = np.random.choice(np.where(eligible_idcs)[0], num_samples_to_remove, replace=False)
                else:
                    raise ValueError(f'No coverage option [{repeat[0]}] available!')
                base_remove[remove_idcs] = True
                stop_repeat = True
                not_iterative = True
        if not not_iterative:
            pbar = tqdm(
                total=num_samples_to_remove, desc='Finding holes...', leave=False)
    else:
        not_iterative = True

    while not stop_repeat:   
        commands = ['all', 'random']
        used_commands = []
        already_filtered = []
        base_operations = []
        to_remove = np.ones(len(dataset)) if connect == 'and' else np.zeros(len(dataset))        
        to_remove = to_remove.astype(bool)

        #Start by removing factors of variation whos values match defined hard constraints.
        for name, constraint in constraints.items():
            if name.split('_')[0] in commands:
                used_commands.append([name, constraint])
            else:
                already_filtered.append(name)
                constraint_idx = np.where(lat_names == name)[0][0]
                comp_vals = dataset.lat_values[:, constraint_idx]
                unique_lat_vals = unique_lat_vals_list[constraint_idx]

                if len(constraint) == 2:
                    if constraint[0] == 'or':
                        constraint_list = constraint[1]
                    elif constraint[0] == 'random':
                        sub_idx = np.random.choice(len(constraint[1]))
                        constraint_list = [constraint[1][sub_idx]]
                    else:
                        constraint_list = [constraint]
                elif len(constraint) == 3:
                    assert_str = f'Not enough constraint options ({len(constraint[1])}) for {constraint[2]}-tuple generation!'
                    assert constraint[2] <= len(constraint[1]), assert_str
                    assert_str = f'Only exact matching ("==") allowed for constraint subsetting!'
                    assert constraint[0] == '==', assert_str
                    assert_str = f'Constraint subset size has to be at least 2 (given: {constraint[2]})!'
                    assert constraint[2] >= 2, assert_str
                    constraint_list = []
                    for i in range(len(constraint[1]) - constraint[2] + 1):
                        constraint_list.append([['==', x] for x in constraint[1][i:i+constraint[2]]])
                    sub_idx = np.random.choice(len(constraint_list))
                    constraint_list = constraint_list[sub_idx]

                constraint_true = np.zeros(len(to_remove))
                for constraint in constraint_list:
                    if constraint[1] == 'random':
                        constr = np.random.choice(unique_lat_vals)
                    else:
                        if isinstance(constraint[1], list):
                            sel_idx = np.random.choice(len(constraint[1]))
                            constr = constraint[1][sel_idx]
                            if isinstance(constr, list):
                                constr = np.random.choice(constr)
                            elif constr == 'random':
                                constr = np.random.choice(unique_lat_vals)
                        else:
                            constr = constraint[1] if not isinstance(constraint[1], str) else eval(constraint[1])
                    #If querying for a numerical value which is not in the available FoV values, 
                    #we search for values within a numerical error of 0.001.
                    #If an exact query is needed ("=="), and no match can be found, an error is thrown.
                    if constr not in unique_lat_vals and not isinstance(constr, str):
                        atol = 0.001
                        if len(unique_lat_vals) > 1./atol:
                            dataset.logger.warning(
                                f'Input error tolerance ({atol}) may not be suitable for available number of ground truth FoV values ({len(unique_lat_vals)})!')
                        has_match = np.isclose(unique_lat_vals, constr, atol=atol)
                        if np.sum(has_match):
                            has_match = np.where(has_match)[0][0]
                            constr = unique_lat_vals[has_match]
                        else:
                            if constraint[0] == '==':
                                raise ValueError(
                                    f"No matching FoV ground truth value for: {constraint}. Available values: {unique_lat_vals}.")

                    op = constraint[0] if not isinstance(constraint[0], list) else np.random.choice(constraint[0])
                    constraint_true = np.logical_or(
                        constraint_true, compare(comp_vals, constr, op, index_hash[constraint_idx]))
                    # constraint_true = compare(comp_vals, constr, op)
                    base_operations.append([name, op, constr])
                if connect == 'and':
                    to_remove = np.logical_and(to_remove, constraint_true)
                elif connect == 'or':
                    to_remove = np.logical_or(to_remove, constraint_true)

        #For every factor not influenced by hard constraints, apply filtering commands (random/all).
        additional_operations = []  
        already_filtered_it = already_filtered[:]        
        for command_name, constraint in used_commands:
            remaining_lats = [x for x in lat_names if x not in already_filtered_it]    
            if len(remaining_lats) and 'random' in command_name:
                remaining_lats = [np.random.choice(remaining_lats)]
                already_filtered_it.extend(remaining_lats)
            for name in remaining_lats:
                constraint_idx = np.where(lat_names == name)[0][0]
                comp_vals = dataset.lat_values[:, constraint_idx]
                unique_lat_vals = unique_lat_vals_list[constraint_idx]

                if len(constraint) == 2:
                    if constraint[0] == 'or':
                        constraint_list = constraint[1]
                    elif constraint[0] == 'random':
                        sub_idx = np.random.choice(len(constraint[1]))
                        constraint_list = [constraint[1][sub_idx]]
                    else:
                        constraint_list = [constraint]
                elif len(constraint) == 3:
                    assert_str = f'Not enough constraint options ({len(constraint[1])}) for {constraint[2]}-tuple generation!'
                    assert constraint[2] <= len(constraint[1]), assert_str
                    assert_str = f'Only exact matching ("==") allowed for constraint subsetting!'
                    assert constraint[0] == '==', assert_str
                    assert_str = f'Constraint subset size has to be at least 2 (given: {constraint[2]})!'
                    assert constraint[2] >= 2, assert_str
                    constraint_list = []
                    for i in range(len(constraint[1]) - constraint[2] + 1):
                        constraint_list.append([['==', x] for x in constraint[1][i:i+constraint[2]]])
                    sub_idx = np.random.choice(len(constraint_list))
                    constraint_list = constraint_list[sub_idx]

                constraint_true = np.zeros(len(to_remove))
                for constraint in constraint_list:                
                    if constraint[1] == 'random':
                        constr = np.random.choice(unique_lat_vals)
                    else:
                        if isinstance(constraint[1], list):
                            sel_idx = np.random.choice(len(constraint[1]))
                            constr = constraint[1][sel_idx]
                            if isinstance(constr, list):
                                constr = np.random.choice(constr)
                            elif constr == 'random':
                                constr = np.random.choice(unique_lat_vals)
                        else:
                            constr = constraint[1] if not isinstance(constraint[1], str) else eval(constraint[1])
                    #If querying for a numerical value which is not in the available FoV values, 
                    #we search for values within a numerical error of 0.001.
                    #If an exact query is needed ("=="), and no match can be found, an error is thrown.
                    if constr not in unique_lat_vals and not isinstance(constr, str):
                        atol = 0.001
                        if len(unique_lat_vals) > 1./atol:
                            dataset.logger.warning(
                                f'Input error tolerance ({atol}) may not be suitable for available number of ground truth FoV values ({len(unique_lat_vals)})!')
                        has_match = np.isclose(unique_lat_vals, constr, atol=atol)
                        if np.sum(has_match):
                            has_match = np.where(has_match)[0][0]
                            constr = unique_lat_vals[has_match]
                        else:
                            if constraint[0] == '==':
                                raise ValueError(
                                    f"No matching FoV ground truth value for: {constraint}. Available values: {unique_lat_vals}.")

                    op = constraint[0] if not isinstance(constraint[0], list) else np.random.choice(constraint[0])
                    constraint_true = np.logical_or(
                        constraint_true, compare(comp_vals, constr, op, index_hash[constraint_idx]))
                    additional_operations.append([name, op, constr])
                    if connect == 'and':
                        to_remove = np.logical_and(to_remove, constraint_true)
                    elif connect == 'or':
                        to_remove = np.logical_or(to_remove, constraint_true)

        all_ops.extend(additional_operations)

        iterm += 1
        if 'coverage' in repeat[0]:
            rem_count = np.sum(to_remove)
            if rem_count * 1./len(base_remove) <= float(repeat[1]):
                base_remove = np.logical_or(base_remove, to_remove)                            
                count = np.sum(base_remove)
                perc = count * 1./len(base_remove)                
                pbar.update(int(count - old_count))
                old_count = count                
                if perc >= float(repeat[1]):
                    stop_repeat = True
        elif repeat[0] == 'count':
            rem_count = np.sum(to_remove)
            if rem_count <= int(repeat[1]):
                base_remove = np.logical_or(base_remove, to_remove)
                count = np.sum(base_remove)
                pbar.update(int(count - old_count))                
                old_count = count
                if count >= int(repeat[1]):
                    stop_repeat = True
        else:
            base_remove = np.logical_or(base_remove, to_remove)
            stop_repeat = True

    if not not_iterative:
        pbar.close()

    if repeat[0] != 'none' and not not_iterative:
        #Allow up to <max_overshoot>% more samples than indicated via coverage (i.e. 11% instead of 10%) 
        #before removing.
        allowed_overshoot = int(num_samples_to_remove * max_overshoot) if allow_overshoot else 0
        num_overshoot = count - num_samples_to_remove
        if num_overshoot > allowed_overshoot:
            reset_idcs = sorted(
                np.random.choice(np.where(base_remove)[0], num_overshoot - allowed_overshoot, replace=False))
            base_remove[reset_idcs] = False

    constrain_summary = {
        'num_removed_samples': np.sum(base_remove),
        'total_num_samples': len(base_remove),
        'perc_removed_entries': np.sum(base_remove) * 1. / len(base_remove),
        'latent_names': dataset.lat_names,
        'value_dist_pre_constrain': [np.unique(dataset.lat_values[..., i], return_counts=True)[-1] for i in range(len(dataset.lat_names))],
        'value_dist_post_constrain': [np.unique(dataset.lat_values[np.logical_not(base_remove), i], return_counts=True)[-1] for i in range(len(dataset.lat_names))],
        'target_percentage': repeat[1] if 'coverage' in repeat[0] else None,
        'target_count': repeat[1] if 'count' in repeat[0] else None,
        'hullmin': None if 'hullmin' not in repeat[0] else repeat[2]
    }
    return base_remove, constrain_summary

@param('constraints.file', 'constraints_filepath')
def constrain(dataset, constraints_filepath):
    """Constrain a dataset based on constraints-yaml.

    This function changes dataset in-place.

    Parameters
    ----------
    dataset : datasets.base.DisentangledDataset
        Base dataset (e.g. for dsprites or cars3d) to constrain.
    constraints_filepath : str
        path to constraints-yaml that contains the names of FoVs and respective values
        to be removed from training. 
        Example constraints.yaml which says that for every objType == 1, remove all entries
        with wallCol > 0.75:
        - shapes3d:
            constraints:
                objType: ['==', 4/4.]
                wallCol: ['>', 0.75]
            connect: and        
    """
    to_remove, constrain_summary = compute_constrained_elements(
        dataset, constraints_filepath)
    percentage_removed = np.sum(to_remove) * 1. / len(to_remove)
    to_keep = np.logical_not(to_remove) 
    dataset.lat_values = dataset.lat_values[to_keep]
    dataset.imgs = dataset.imgs[to_keep]
    dataset.logger.info(
        f'[Constraints] Removed {int(np.sum(to_remove))}/{len(to_remove)} | {percentage_removed * 100}% of data entries.')
    constrain_summary['removed_idcs'] = to_remove
    return constrain_summary
    
#------------- CORRELATION SETTINGS ---------------
def extract_correlations_yaml(correlations_filepath):
    name = None
    if ':' in correlations_filepath:
        correlations_filepath, name = correlations_filepath.split(':')
    with open(correlations_filepath) as correlation_file:
        yaml_file = yaml.load(correlation_file, Loader=yaml.FullLoader)
        if name is not None:
            yaml_file = yaml_file[name] 
    repeat = ['none'] if 'repeat' not in yaml_file else yaml_file['repeat']
    return yaml_file['correlations'], repeat

@param('constraints.correlations_file', 'correlations_filepath')
@param('constraints.correlation_distribution')
def provide_correlation_sampler(dataset, correlations_filepath, correlation_distribution='traeuble'):
    """Provide a sampler for a PyTorch DataLoader which samples from a standard dataset using correlated FoVs (see [1]).
    
    Parameters
    ----------
    dataset : datasets.base.DisentangledDataset
        Base dataset (e.g. for dsprites or cars3d) to convert to a paired variant.
    correlations_filepath : str
        path to yaml with correlation constraints. Of shape <path_to_file.yaml:name_of_constraint>.
    correlation_distribution : str
        name of distribution to use for correlations. Uses 'traeuble' (see [1]) by default.

    Returns
    -------
    torch.utils.data.WeightedRandomSampler : 
        Is weighted such that it correctly accounts for pairwise correlations. Inserted into the DataLoader, and
        can thus easily be used with both constrained and weakly supervised training.

    References
    ----------
    [1] Traeuble et al., "On Disentangled Representations Learned from Correlated Data"
    """
    if correlation_distribution == 'traeuble':
        #Note that we assume that each factor c_i is already normalized to [0...1].
        f_corr = lambda c_1, c_2, sigma: np.exp(-(c_1 - c_2)**2/(2 * sigma ** 2))
    elif correlation_distribution == 'inv_traeuble':
        #Note that we assume that each factor c_i is already normalized to [0...1].
        f_corr = lambda c_1, c_2, sigma: np.exp(-(c_1 - (1 - c_2))**2/(2 * sigma ** 2))
    else:
        raise ValueError(f'No correlation distribution [{correlation_distribution}] available!')
        
    correlations, repeat = extract_correlations_yaml(correlations_filepath)
    correlations = {eval(key):item for key, item in correlations.items()}

    temp_correlations = {}
    all_pairs = []
    for lat_name_1 in dataset.lat_names:
        for lat_name_2 in dataset.lat_names:
            if lat_name_1 != lat_name_2 and (lat_name_2, lat_name_1) not in all_pairs:
                all_pairs.append((lat_name_1, lat_name_2))
    if repeat[0] == 'none':
        num_correlation_passes = 1
    else:
        if 'count' in repeat[0]:
            num_correlation_passes = repeat[1]
            if repeat[1] == 'max_num_single':
                num_correlation_passes = len(dataset.lat_names) - 1
            if repeat[1] == 'max_num_pairs':
                num_correlation_passes = len(all_pairs)
        elif repeat[0] == 'coverage':
            num_correlation_passes = int(repeat[1] * len(all_pairs))

    correlations_list = [correlations for _ in range(num_correlation_passes)]
    for correlations in correlations_list:
        for corr_pair in correlations.keys():
            if not len(all_pairs):
                raise ValueError('Enforcing more correlation pairs than available!')
            if corr_pair == ('random', 'random'):
                pair_idx = np.random.choice(len(all_pairs))
                adj_corr_pair = all_pairs[pair_idx]
            elif corr_pair[0] == 'random':
                avail_pairs_idcs = [i for i, x in enumerate(all_pairs) if x[1] == corr_pair[1] or x[1] == corr_pair[1]]
                if not len(avail_pairs_idcs):
                    raise ValueError('Enforcing more correlation pairs than available!')
                adj_corr_pair = all_pairs[np.random.choice(avail_pairs_idcs)]
            elif corr_pair[1] == 'random':
                avail_pairs_idcs = [i for i, x in enumerate(all_pairs) if x[0] == corr_pair[0] or x[1] == corr_pair[0]]
                if not len(avail_pairs_idcs):
                    raise ValueError('Enforcing more correlation pairs than available!')
                adj_corr_pair = all_pairs[np.random.choice(avail_pairs_idcs)]
            else:
                adj_corr_pair = corr_pair
            if adj_corr_pair in all_pairs:
                all_pairs.remove(adj_corr_pair)
            elif tuple(reversed(adj_corr_pair)) in all_pairs:
                all_pairs.remove(tuple(reversed(adj_corr_pair)))
            else:
                raise ValueError(f'The correlation pair {adj_corr_pair} is not possible!')
            corr_val = correlations[corr_pair]
            if isinstance(corr_val, list):
                val_idx = np.random.choice(len(corr_val))
                corr_val = corr_val[val_idx]
            temp_correlations[tuple(adj_corr_pair)] = corr_val
    correlations = temp_correlations

    lat_name_to_idx = {lat_name: i for i, lat_name in enumerate(dataset.lat_names)}
    correlations = {tuple(lat_name_to_idx[x] for x in key): item for key, item in correlations.items()}
    
    weights_coll = []
    for (idx_1, idx_2), sigma in correlations.items():
        if sigma != 'none':
            weights = f_corr(dataset.lat_values[..., idx_1], dataset.lat_values[..., idx_2], sigma)
            weights_coll.append(weights)
    if len(weights_coll):
        correlation_weights = np.prod(weights_coll, axis=0)
        # correlation_weights = np.mean(np.stack(weights_coll, axis=1), axis=-1)
    else:
        correlation_weights = np.ones(len(dataset))
    
    return torch.utils.data.WeightedRandomSampler(correlation_weights, len(dataset), replacement=True), correlation_weights


#------------- WEAKLY SUPERVISED SETTINGS ---------------
class PairedDataset(torch.utils.data.Dataset):
    @param('data.k_range')
    @param('data.pair_index')
    def __init__(self, dataset, k_range=[1, 1], pair_index='locatello', correlation_weights=None):
        """Convert a standard dataset to a Paired Variants (see [1]).

        Parameters
        ----------
        dataset : datasets.base.DisentangledDataset
            Base dataset (e.g. for dsprites or cars3d) to convert to a paired variant.
        k_range : List[int]
            Number of UNshared factors of variation, where k_range[0] is the minimal and
            k_range[1] the maximal number of non-shared factors of variation. If 
            k_range[1] = -1, this value goes up to latent_dim - 1, i.e. only sharing a single
            factor of variation.
        pair_index: str
            Denotes the method used to sample the number of unshared FoVs - k.
            Choose from ["locatello", "uniform", "uniform_fixed"], where the former follows the 
            implementation in https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/methods/weak/train_weak_lib.py#L41-L57
            which differs from the original paper description, going beyond just uniformly sampling k.
            "uniform" is the same as "locatello", but only uniformly samples k. And finally, 
            "uniform_fixed" is closest to what it should be based on the paper description: Uniformly sample
            k AND ENSURE that for unshared factors of variation the respective entries are differing.

        References
        ----------
        [1] Locatello et al., "Weakly Supervised Disentanglement without Compromises"

        """
        self._align_attributes(dataset)
        self.k_range = k_range
        self.k_min = k_range[0]
        self.k_max = k_range[1] if k_range[1] != -1 else len(dataset.lat_sizes) - 1
        self.dataset = dataset

        self.correlation_weights = correlation_weights

        self._compute_map_and_index_hash(dataset)

        if pair_index == 'uniform':
            self._sample_shared_idcs = self._sample_shared_idcs_uniformly
        elif pair_index == 'uniform_fixed':
            self._sample_shared_idcs = self._sample_shared_idcs_uniformly_fixed
        elif pair_index == 'locatello':
            self._sample_shared_idcs = self._sample_shared_idcs_locatello

    def _align_attributes(self, dataset):
        self.__dict__.update(type(dataset).__dict__)
        self.__dict__.update(dataset.__dict__)

    def _compute_map_and_index_hash(self, dataset):
        self.unique_lat_values = {}
        start = time.time()
        for factor_id in range(dataset.lat_values.shape[-1]):
            self.unique_lat_values[factor_id] = np.unique(dataset.lat_values[:, factor_id])
        self.lats_to_idx = {tuple(lat_vals): i for i, lat_vals in enumerate(dataset.lat_values)}
        dataset.logger.info('Computed latent -> idx map in {}s'.format(
            time.time() - start
        ))

        self.index_hash = {}
        iterator = tqdm(
            enumerate(self.unique_lat_values.keys()), desc='Creating hashmap...', total=len(self.unique_lat_values), leave=False)
        for i, key in iterator:
            self.index_hash[i] = {}
            for unique_lat_val in self.unique_lat_values[key]:
                self.index_hash[i][unique_lat_val] = (dataset.lat_values[..., i] == unique_lat_val).astype(bool)

    def _sample_shared_idcs_uniformly(self, idx):
        num_factors_of_v = len(self.lat_sizes)
        lat_vals = self.dataset.lat_values[idx]
        k = np.random.choice(range(self.k_min, self.k_max + 1), 1, replace=False)[0]
        shared_factor_idcs = sorted(np.random.choice(num_factors_of_v, num_factors_of_v - k, replace=False))

        avail_lat_idcs = np.ones(len(self.dataset.lat_values), dtype=bool)
        for i in shared_factor_idcs:
            avail_lat_idcs = np.logical_and(avail_lat_idcs, self.index_hash[i][lat_vals[i]])

        if self.correlation_weights is not None:
            weights = self.correlation_weights[avail_lat_idcs]
            sampling_p = weights/np.sum(weights)
            pair_idx = np.random.choice(len(sampling_p), p=sampling_p)
        else:
            pair_idx = np.random.choice(int(np.sum(avail_lat_idcs)))
        pair_lat_vals = self.dataset.lat_values[avail_lat_idcs][pair_idx]
    
        # pair_lat_vals = []
        # while tuple(pair_lat_vals) not in self.lats_to_idx:
        #     pair_lat_vals = []            
        #     for i in range(num_factors_of_v):
        #         if i in shared_factor_idcs:
        #             pair_lat_vals.append(lat_vals[i])
        #         else:
        #             pair_lat_vals.append(np.random.choice(self.unique_lat_values[i]))
        pair_lat_vals = tuple(pair_lat_vals)
        pair_idx = self.lats_to_idx[pair_lat_vals]
        shared_factor_idcs_1h = [1 if i in shared_factor_idcs else 0 for i in range(num_factors_of_v)]
        return pair_idx, shared_factor_idcs_1h

    def _sample_shared_idcs_uniformly_fixed(self, idx):
        num_factors_of_v = len(self.lat_sizes)
        lat_vals = self.dataset.lat_values[idx]
        k = np.random.choice(range(self.k_min, self.k_max + 1), 1, replace=False)[0]
        shared_factor_idcs = sorted(np.random.choice(num_factors_of_v, num_factors_of_v - k, replace=False))
        non_shared_factor_idcs = [i for i in range(num_factors_of_v) if i not in shared_factor_idcs]
       
        #Find subset of latent vectors which share factors.
        avail_lat_idcs = np.ones(len(self.dataset.lat_values), dtype=bool)
        for i in shared_factor_idcs:
            avail_lat_idcs = np.logical_and(avail_lat_idcs, self.index_hash[i][lat_vals[i]])
        #Ensure no overlap with non-shared FoV values.
        for i in non_shared_factor_idcs:
            avail_lat_idcs = np.logical_and(avail_lat_idcs, np.logical_not(self.index_hash[i][lat_vals[i]]))

        if self.correlation_weights is not None:
            weights = self.correlation_weights[avail_lat_idcs]
            sampling_p = weights/np.sum(weights)
            pair_idx = np.random.choice(len(sampling_p), p=sampling_p)
        else:
            pair_idx = np.random.choice(int(np.sum(avail_lat_idcs)))
        pair_lat_vals = self.dataset.lat_values[avail_lat_idcs][pair_idx]

        # pair_lat_vals = []
        # while tuple(pair_lat_vals) not in self.lats_to_idx:
        #     pair_lat_vals = []            
        #     for i in range(num_factors_of_v):
        #         if i in shared_factor_idcs:
        #             pair_lat_vals.append(lat_vals[i])
        #         else:
        #             sample_set = list(set(self.unique_lat_values[i]) - {lat_vals[i]})
        #             pair_lat_vals.append(np.random.choice(sample_set))
        pair_lat_vals = tuple(pair_lat_vals)
        pair_idx = self.lats_to_idx[pair_lat_vals]
        shared_factor_idcs_1h = [1 if i in shared_factor_idcs else 0 for i in range(num_factors_of_v)]
        return pair_idx, shared_factor_idcs_1h

    def _sample_shared_idcs_locatello(self, idx):
        num_factors_of_v = len(self.lat_sizes)
        lat_vals = self.dataset.lat_values[idx]
        k = np.random.choice(range(self.k_min, self.k_max + 1), 1, replace=False)[0]
        #Locatello et al. perform a secondary sampling:
        #https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/methods/weak/train_weak_lib.py#L41-L57
        #This significantly increases the chance of receiving pairs with most factors shared!        
        k = np.random.choice([1, k])
        shared_factor_idcs = sorted(np.random.choice(num_factors_of_v, num_factors_of_v - k, replace=False))

        #Find subset of latent vectors which share factors.
        avail_lat_idcs = np.ones(len(self.dataset.lat_values), dtype=bool)
        for i in shared_factor_idcs:
            avail_lat_idcs = np.logical_and(avail_lat_idcs, self.index_hash[i][lat_vals[i]])

        if self.correlation_weights is not None:
            weights = self.correlation_weights[avail_lat_idcs]
            sampling_p = weights/np.sum(weights)
            pair_idx = np.random.choice(len(sampling_p), p=sampling_p)
        else:
            pair_idx = np.random.choice(int(np.sum(avail_lat_idcs)))
        pair_lat_vals = self.dataset.lat_values[avail_lat_idcs][pair_idx]

        # pair_lat_vals = []
        # for i in range(num_factors_of_v):
        #     if i not in shared_factor_idcs:
        #         pair_lat_vals.append(None)
        #     else:
        #         pair_lat_vals.append(lat_vals[i])
        # for i in range(num_factors_of_v):
        #     if i not in shared_factor_idcs:
        #         if self.correlations is not None:                    
        #             weight_coll = []
        #             for j in range(len(pair_lat_vals)):
        #                 #self.correlations is a dictionary of structure
        #                 #{(idx_of_fov_1, idx_of_fov_2): correlation_strength, ...}
        #                 # Decomposes p(fov_1, fov_2, ..., fov_n) into p(fov_1|fov_2, ..., fov_n) * p(fov_2 | fov_3, ..., fov_n) * p(fov_n)
        #                 # Where some fov_2 are given due to sharing, i.e. p({fov_i} | {fov_k}) where {fov_i} and {fov_k} denote the set of 
        #                 # Factors of Variation that are non-shared and shared, respectively.
        #                 sigma = None
        #                 if (i, j) in self.correlations:
        #                     sigma = self.correlations[(i, j)]
        #                 elif (j, i) in self.correlations:
        #                     sigma = self.correlations[(j, i)]                                
        #                 if sigma is not None:
        #                     if pair_lat_vals[j] is not None:                         
        #                         weight_coll.append(self.f_corr(self.unique_lat_values[i], pair_lat_vals[j], sigma))
        #             if len(weight_coll):
        #                 sampling_weights = np.prod(weight_coll, axis=0)
        #                 sampling_p = sampling_weights / np.sum(sampling_weights)
        #                 entry = np.random.choice(self.unique_lat_values[i], p = sampling_p)
        #             else:
        #                 entry = np.random.choice(self.unique_lat_values[i])
        #         else:
        #             entry = np.random.choice(self.unique_lat_values[i])
        #         pair_lat_vals[i] = entry

        # #Update remaining ones based on marginals.
        # pair_lat_vals = []
        # while tuple(pair_lat_vals) not in self.lats_to_idx:
        #     pair_lat_vals = []            
        #     for i in range(num_factors_of_v):
        #         if i in shared_factor_idcs:
        #             pair_lat_vals.append(lat_vals[i])
        #         else:
        #             pair_lat_vals.append(np.random.choice(self.unique_lat_values[i]))
        pair_lat_vals = tuple(pair_lat_vals)
        pair_idx = self.lats_to_idx[pair_lat_vals]
        shared_factor_idcs_1h = [1 if i in shared_factor_idcs else 0 for i in range(num_factors_of_v)]
        return pair_idx, shared_factor_idcs_1h

    def __getitem__(self, idx):
        pair_idx, shared_factor_idcs_1h = self._sample_shared_idcs(idx)
        sample, lat_values = self.dataset[idx]
        pair_sample, pair_lat_values = self.dataset[pair_idx]
        return (sample, pair_sample), (lat_values, pair_lat_values), shared_factor_idcs_1h

    def __len__(self):
        return len(self.dataset.imgs)