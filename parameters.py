"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import logging

from fastargs import Section, Param
from fastargs.validation import And, Anything, Checker, OneOf

import utils
import os

# hack to set default wandb key if environment variable is set:
WANDB_DEFAULT_KEY = os.getenv('WANDB_API_KEY', '<your_wandb_key>')

#------------ General parameters -------------------------------
Section('run', 'base run parameters').params(
    do=Param(utils.List(),
             'Provide list of things to do - train, eval, visualize.',
             default=['train', 'eval', 'visualize']),
    on_cpu=Param(utils.Bool(), 'If set to true, runs on CPU', default=False),
    restore_from=Param(str,
                       'Restore run from checkpoint folder. Set to "n/a" if not'
                       'restoring, <path_to_chkpt> to restore from checkpoint,'
                       'and "continue" to continue from standard folder name.'
                       'Set to "overwrite" to overwrite folder with same names.',
                       default='continue'),
    restore_with_config=Param(utils.Bool(), 'Flag. If set, will also load full chkpt fastargs config.', default=False),
    restore_to_new=Param(utils.Bool(), 
                         'Flag. If set, restores a new training in <restore_from> described by log.project & log.group.', 
                         default=False),
    gpu_id=Param(int, 'gpu-id to run on.', default=0))

Section('train', 'base training parameters').params(
    supervision=Param(str, 'Type of training supervision: none, pairs', default='none'),
    lr=Param(float, 'Learning rate', default=5e-4),
    seed=Param(int, 'seed value for reproducibility', default=0),
    optimizer=Param(str, 'Optimizer to use. Currently available: [adam]', default='adam'),
    batch_size=Param(int, 'batchsize', default=64),
    epochs=Param(int, 'number of training epochs', default=100),
    iterations=Param(int, 'number of training iterations. If set, will adapt train.epochs to a suitable value.', default=-1),
    checkpoint_every=Param(int, 'number of epochs for each ckpt.', default=30),
    checkpoint_first=Param(int, 'checkpoint each of the first N epochs.', default=0),
    loss=Param(str, 'loss name', default='burgess'),
    rec_dist=Param(str, 'rec_dist', default='bernoulli'),
    record_loss_every=Param(int,
                            '#iterations to record loss value at',
                            default=50))

Section('scheduler', 'scheduling parameters').params(
    name=Param(str, 'Name of learning rate scheduler. Currently available: none, multistep.', default='none'),
    tau=Param(utils.List(), 'Learning rate scheduling steps', default=[]),
    gamma=Param(utils.List(), 'Learning rate decay values', default=[]))

Section('eval', 'evaluation parameters').params(
    mode=Param(
        str,
        "Which checkpoints to use when 'run.restore_from'==continue or points "
        "to a folder. Option: <last>, <exact> checkpoint when available and set"
        "via <run.restore_from>,  or <all> - in this case, evaluates all "
        "checkpoints of form chkpt-N.pth.tar.",
        default='all'),
    metrics=Param(
        utils.List(),
        "Lists metrics to compute on the evaluation data. Options are :"
        "'mig', 'sap_d', 'dci_d', 'modularity_d', 'reconstruction_error'",
        default=['kld', 'fos', 'mig', 'sap_d', 'dci_d', 'modularity_d', 'reconstruction_error']),
    to_compute=Param(
        utils.List(),
        'What to compute - metrics and/or losses. Options: metrics, losses',
        default=['metrics']),
    no_test=Param(utils.Bool(),
                  'Flag. If true, run test after training',
                  default=False),
    batch_size=Param(int, 'Evaluation Batchsize', default=1000))

Section('model', 'all model-related parameters').params(
    name=Param(str,' model choice', default='vae_burgess'),
    encoder_decay=Param(float, 'L2 Decay on encoder', default=0.),
    decoder_decay=Param(float, 'L2 Decay on decoder', default=0.))

Section('data', 'all data-related parameters').params(
    name=Param(str, 'dataset name', default='shapes3d'),
    root=Param(str, 'optional root to dataset', default='n/a'),
    k_range=Param(utils.List(), 'number of UNshared factors of variation. k_range = [k_min, k_max]. If k_max = -1 -> k_max = num_factors_of_variation - 1', default=[1,-1]),
    pair_index=Param(str, 
                     'Method to select number of shared factors. By default, selects "locatello" - the method'
                     'implemented in https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/methods/weak/train_weak_lib.py#L41-L57'
                     'which differs from the purely uniform sampling described in the paper.'
                     'Other options: "uniform", which uniformly samples k, and "uniform_fixed", which'
                     'uniformly samples k AND ENSURES different FoV values for the unshared entries.', 
                     default='locatello'),
    subset=Param(float, 'optional subset of data for debugging. Uses all by default.', default=1),
    num_workers=Param(int, 'number of workers', default=8))

Section('log', 'all logging-related hyperparameters').params(
    level=Param(str, 'logging log level', default='info'),
    base_dir=Param(str,
                   'base directory to store run outputs s.a. chkps.',
                   default='results'),
    group=Param(
        str,
        'experimental group (groups multiple seeds). If "default" uses default naming convention.',
        default='default'),
    project=Param(
        str,
        'experimental project (groups multiple groups). If "default" uses default naming convention.',
        default='default'),
    wandb_mode=Param(str, "Denote wandb logging mode.", default='run'),
    wandb_allow_val_change=Param(utils.Bool(), "Flag - is set, allows config to be overwritten when continuing checkpoints.", default=False),
    wandb_key=Param(str, "Weights & Biases key.", default=WANDB_DEFAULT_KEY),
    new_wandb=Param(utils.Bool(), 
                    'Flag. If set, will not continue from checkpointed W&B logger but new one'
                    'Currently only relevant for separate evaluation/visualization.', 
                    default=False),
    printlevel=Param(int, "How much information to print. 1: Minimal, 2: All.", default=1))


#------------ Main_Viz visualization -------------------------------
Section('viz', 'external visualization parameters for main_viz').params(
    mode=Param(
        str,
        "Denotes which checkpoints to visualize. Has four options:"
        "<exact> For exact checkpoint, simply set <run.restore_from> pointing "
        "to a specific checkpoint, <last> for last checkpoint (chkpt.pth.tar), "
        "<all> for all checkpoints of form chkpt-N.pth.tar, <eval> to match "
        "the <eval.mode> setup.",
        default='eval'),
    plots=Param(
        utils.List(),
        "List of all plots to generate. `generate-samples`: random decoded "
        "samples. `data-samples` samples from the dataset. `reconstruct` first "
        "nrows//2 will be the original and rest will be the corresponding "
        "reconstructions. `traversals` traverses the most important nrows "
        "dimensions with ncols different samples from the prior or posterior. "
        "`reconstruct-traverse` first row for original, second are "
        "reconstructions, rest are traversals. `gif-traversals` grid of gifs "
        "where rows are latent dimensions, columns are examples, each gif shows"
        "posterior traversals. `all` runs every plot.",
        default=['all']),
    nrows=Param(int,
                '#Rows to visualize if applicable (corresponds to number of latents to visualize)', 
                default=10),
    ncols=Param(int, 
                '#Cols to visualize if applicable (corresponds to number of samples to visualize)', 
                default=7),
    reorder_latents_by_kl=Param(utils.Bool(),
                                'should latents be reordered based on kl',
                                default=True),
    n_per_latent=Param(int,
                        'number of points to include in latent traversal',
                        default=10),
    n_latents=Param(
        int,
        'number of latent dimensions to display. If -1 displays all.',
        default=-1),
    max_traversal=Param(
        int,
        "The maximum displacement induced by a latent traversal. Symmetrical "
        "traversals are assumed. If `m>=0.5` then uses absolute value traversal"
        ", if `m<0.5` uses a percentage of the distribution (quantile). E.g. "
        "for the prior the distribution is a standard normal so `m=0.45` "
        "corresponds to an absolute value of `1.645` because `2m=90%%` of a "
        "standard normal is between `-1.645` and `1.645`. Note in the case of "
        "the posterior, the distribution is not standard normal anymore.",
        default=2),
    upsample_factor=Param(
        float,
        "The scale factor with which to upsample the image (if applicable).",
        default=1),
    show_loss=Param(utils.Bool(), 'Display loss on figs.', default=False),
    is_posterior=Param(utils.Bool(),
                       'Traversers the posterior instead of the prior',
                       default=True),
    save_images=Param(utils.Bool(),
                      'Flag. If set, images are saved.',
                      default=True),
    loss_of_interest=Param(str, 'Loss to visualize', default='kl_loss_'),
    display_loss_per_dim=Param(
        utils.Bool(),
        "if the loss should be included as text next to the corresponding "
        "latent dimension images.",
        default=False),
    plot_sample_pairs=Param(utils.Bool(), 'Plot example pairs for weakly supervised training.', default=True),
    idcs=Param(
        utils.List(),
        'List of indices to of images to put at the begining of the samples.',
        default=[]))

#------------ Model-specific parameters -----------------------------
Section('vae', 'Parameter for a custom VAE.').params(
    latent_dim=Param(int, 'Latent dimensionality', default=10),
    encoder=Param(str, 'encoder to use: locatello, chen_mlp, burgess.', default='locatello'),
    decoder=Param(str, 'decoder to use: locatello, chen_mlp, burgess, sbd', default='locatello'))

Section('vae_burgess', 'Parameter for VAE taken from Burgess et al.').params(
        latent_dim=Param(int, 'Latent dimensionality', default=10))

Section('vae_locatello', 'Parameter for VAE taken from Locatello et al.').params(
        latent_dim=Param(int, 'Latent dimensionality', default=10))

Section('vae_chen_mlp', 'Parameter for MLP VAE taken from Chen et al.').params(
        latent_dim=Param(int, 'Latent dimensionality', default=10))


#------------ Loss-specific parameters -----------------------------
Section('betavae', 'higgins beta-VAE parameters').params(
    beta=Param(float, 'beta value', default=4),
    log_components=Param(
        utils.Bool(),
        'Flag. If set, logs kl-loss for each latent component.',
        default=False))

Section('annealedvae', 'Burgess annealed beta-VAE parameters').params(
    C_init=Param(float, 'initial annealed capacity.', default=0),
    C_fin=Param(float, 'final annealed capacity.', default=25),
    gamma=Param(float, 'Replaced standard beta to account for annealed capacity.', default=1000),
    anneal_steps=Param(int, 'reg_anneal', default=100000),    
    log_components=Param(
        utils.Bool(),
        'Flag. If set, logs kl-loss for each latent component.',
        default=False))

Section('factorvae', 'factorVAE parameters').params(
    gamma=Param(float, 'weight on total correlation loss', default=6),
    discr_neg_slope=Param(float, 'negative_slope', default=0.2),
    discr_lr=Param(float, 'learning rate discriminator', default=0.0001),
    discr_hidden_units=Param(int, 'number of hidden units/layer', default=1000),
    discr_latent_dim=Param(int, 'input latent dim, usually <latent_dim>.', default=10),
    discr_betas=Param(utils.List(), 'Adam parameters', default=[0.5, 0.9]),
    anneal_steps=Param(int, 'reg_anneal', default=100000),    
    log_components=Param(
        utils.Bool(),
        'Flag. If set, logs kl-loss for each latent component.',
        default=False))

Section('betatcvae', 'btcVAE parameters').params(
    alpha=Param(float, 'Weight on Mutual Information.', default=1),
    gamma=Param(float, 'Weight on dimension-wise KLD.', default=1),
    beta=Param(float, 'Weight on Total Correlation.', default=6),
    is_mss=Param(
        utils.Bool(),
        'Flag if minibatch stratified sampling should be used.',
        default=True),
    log_components=Param(
        utils.Bool(),
        'Flag. If set, logs kl-loss for each latent component.',
        default=False))

Section('adagvae', 'Adaptive Group-VAE parameters').params(
    annealing=Param(str, 
                    'Which method to use to anneal the KL-Divergence between posterior and normal prior.'
                    'Available: higgins, burgess.', 
                    default='higgins'),
    thresh_mode=Param(str, 'Adaptive threshold mode: kl, symmetric_kl, dist, sampled_dist', default='symmetric_kl'),
    average_mode=Param(str, 'Type of (shared) posterior averaging: gvae, mlvae', default='gvae'),
    thresh_ratio=Param(float, 'Thresholding for Adaptive thresholding.', default=0.5),
    beta=Param(float, 'If annealing==higgins: beta-VAE beta.', default=6),
    C_init=Param(float, 'If annealing==burgess: initial annealed capacity.', default=0),
    C_fin=Param(float, 'If annealing==burgess: final annealed capacity.', default=25),
    sanity_check=Param(utils.Bool(), 'If set, converts AdaGVAE to effectively a beta-VAE for a sanity check.', default=False),
    gamma=Param(float, 'If annealing==burgess: Replaced standard beta to account for annealed capacity.', default=100),    
    anneal_steps=Param(int, 'reg_anneal', default=100000),    
    log_components=Param(
        utils.Bool(),
        'Flag. If set, logs kl-loss for each latent component.',
        default=False))

Section('factorizedsupportvae', 'standard VAE with factorized support constraint').params(
    beta=Param(float, 'Weight on latent factorization regularization.', default=1),
    gamma=Param(float, 'Weight on factorized support regularization.', default=1),
    delta=Param(float, 'Weight on scale regularization.', default=0),
    btc_alpha=Param(float, 'Weight on Mutual Information.', default=1),
    btc_gamma=Param(float, 'Weight on dimension-wise KLD.', default=1),
    btc_beta=Param(float, 'Weight on Total Correlation.', default=6),
    is_mss=Param(utils.Bool(), 'Flag if minibatch stratified sampling should be used.', default=True),    
    reg_mode=Param(str, 'Scale regularizer for support. Available: minimal_support, variance', default='variance'),
    reg_range=Param(utils.List(), 'If reg_mode == minimal_support, this denotes [a, b].', default=[0., 1.]),
    matching=Param(str, 'Matching method to factorized support', default='hausdorff_hard'),
    use_rec=Param(int, 'If 0: Do not use reconstruction (likelihood) objective.', default=1),
    latent_select=Param(str, 'Whether to randomly subsample the full latents into latent pairs or not. Choose from <pair> and <all>.', default='pair'),
    factorized_support_estimation=Param(
        str, 
        'How to compute the distance against the factorized support, computing either the '
        '<full> factorized support set, or using <random> samples ', 
        default='full'),
    num_support_estimators=Param(
        int, 
        'If the factorized support is estimated at random, this value will denote the '
        'number of vectors to draw from the factorized support.', 
        default=100),
    num_latent_pairs=Param(
        int, 
        'Number of pairs of latent indices to generate for pairwise matching. '
        'Will be capped at the total number of latent pairings.', 
        default=25),
    temperature_1=Param(float, 'Softmax temperature for inner operation in Hausdorff approximations', default=1),
    temperature_2=Param(float, 'Softmax temperature for outer operation in Hausdorff approximations', default=1),
    inner_prob_samples=Param(int, 'Number of entries to sample in inner operation for prob. Hausdorff approx.', default=5),
    outer_prob_samples=Param(int, 'Number of entries to sample in outer operation for prob. Hausdorff approx.', default=20),
    log_components=Param(
        utils.Bool(),
        'Flag. If set, logs kl-loss for each latent component.',
        default=False))

#------------ Metric-specific parameters -----------------------------
Section('sap', 'SAP parameters').params(
        num_train=Param(int, 'Number of samples to train classifier on.', default=10000),
        num_test=Param(int, 'Number of samples to test classifier on.', default=5000),
        num_bins=Param(int, 'number of discretization bins', default=20))

Section('dci', 'DCI parameters').params(
        num_train=Param(int, 'Number of samples to train classifier on.', default=10000),
        num_test=Param(int, 'Number of samples to test classifier on.', default=5000),
        num_bins=Param(int, 'number of discretization bins.', default=20),
        backend=Param(str, 'GradientBoostingClassifier backend to use.', default='sklearn'))

Section('modularity', 'Modularity parameters').params(
        num_bins=Param(int, 'number of discretization bins', default=20))

Section('mig', 'MIG parameters').params(
        num_bins=Param(int, 'number of discretization bins', default=20))

Section('fos', 'FoS parameters').params(
        num_pairs=Param(int, 'number of pairs for pairwise approximation.', default=-1),
        batch_size=Param(int, 'batchsize to approximate factorized support.', default=128))

Section('kld', 'KL-Div. parameters').params(
        batch_size=Param(int, 'batchsize to approximate KLD-Divergence.', default=128))


#------------ Dataset-specific parameters -----------------------------
Section('constraints', 'Constrain support of ground truth factors based on conditions.').params(
    file=Param(str, 'path to yaml with constraint', default='none'),
    correlations_file=Param(str, 'path to yaml with correlations', default='none'),
    correlation_distribution=Param(str, 'distribution to use to correlate FoVs', default='traeuble'),
    allow_overshoot=Param(
        utils.Bool(), 
        'As holes are sampled iteratively, the number of holes may overshoot the limit indicated.'
        'This only happens when full chunks of samples are removed at once. In this case, if this flag'
        'is set to False or if the maximum allowed overshoot is passed, subsampling is performed'
        'This means that there may occur instances in which chunks are planned for removal, '
        'but a handful of samples are still retained.',
        default=True),
    max_overshoot=Param(
        float, 
        'Percentage of limit that can be overshot.'
        'Set to 0.1 by default - i.e. when the percentage of holes is limited to 10, '
        '10 * 1.1. = 11 are instead allowed when generated by accident.', 
        default=0.1)
)

Section('dsprites', 'DSprites-specific parameters').params(
    factors_to_use=Param(utils.List(), 'ground truth factors to include', default=['shape', 'scale', 'orientation', 'posX', 'posY']))


#------------ Sanity Checks ------------------
def sanity_checks(config, logger=None):
    if config['train.loss'] == 'factor':
        pass
        # if logger:
        #     logger.info(
        #         "FactorVae needs 2 batches per iteration. To replicate this "
        #         "behavior while being consistent, we double the batch size and "
        #         "the number of epochs.")
        # config.collect({
        #     'train.batch_size': config['train.batch_size'] * 2,
        #     'train.epochs': config['train.epochs'] * 2
        # })
