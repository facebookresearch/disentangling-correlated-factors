# Disentangling Correlated Factors [![Python 3.8+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

---
<img align="left" width="210" src=images/illustration.png>

This if the official repository for the code associated with the following paper:

[Disentanglement of Correlated Factors via Hausdorff Factorized Support](https://openreview.net/forum?id=OKcJhpQiGiX), **Karsten Roth, Mark Ibrahim, Zeynep Akata, Pascal Vincent, Diane Bouchacourt,** *International Conference on Learning Representations (ICLR 2023)*, 2023.

This code was written primarily by [Karsten Roth](https://karroth.com/) during a research internship at Meta/FAIR in 2022. 

The repository contains a general-purpose *pytorch-based* framework library and benchmarking suite to facilitate research on methods for learning disentangled representations. It was conceived especially for evaluating robustness under correlated factors, and contains all the code, benchmarks and method implementations used in the paper. 

If you find it useful or use it for your experiments, consider giving this repository a star, and please cite our paper [  [bibtex]  ](#citing-our-work)


## Quick overview

Check out [***Implementations***](#implemented-methods-and-metrics) for a list of everything implemented, and [***Complete Examples***](#complete-examples) for a collection of singular run-and-evaluation examples for a quick start.

To find an implementation for our _Hausdorff Factorized Support_, simply check out for example `dent/losses/factorizedsupportvae.py`, which applies Hausdorff Support Factorization on a simple β-VAE.

To replicate large-scale literature gridsearch results on disentanglement under correlations, please refer to [***Running a large-scale gridsearch over correlated FOVs***](#running-a-large-experimental-gridsearch) which assumes a SLURM-compatible compute system.


## What we built on

This repository contains and adapts some elements from other great frameworks in the space of disentangled representation learning:

- Yann Dubois et al.'s [Disentangling VAE](https://github.com/YannDubs/disentangling-vae) provided initial codebase structure, base backbone & VAE architecture, base loss function formulation.
- Google research's [disentanglement_lib](https://github.com/google-research/disentanglement_lib) was used for AdaGVAE implementation, metrics implementation, and general crosscheck of architectural details.
- Nathan Juraj Michlo's [disent](https://github.com/nmichlo/disent) was used as a reference PyTorch implementation of disentanglement metrics.

The majority of **Disentangling Correlated Factors** is licensed under the MIT license, however portions of the project are available under separate license terms: https://github.com/google-research/disentanglement_lib is licensed under the Apache 2.0 license.



---
**Table of Contents**:

- [Disentangling correlated factors]()
  - [What we built on](#what-we-built-on)
  - [Installation and Requirements](#installation-and-requirements)
  - [Implemented Methods and Metrics](#implemented-methods-and-metrics)
    - [Training Objectives and Architectures](#training-objectives-and-architectures)
    - [Metrics](#metrics)
    - [Benchmarks](#benchmarks)
    - [Additional Features](#additional-features)
  - [Training](#training)
      - [Key Files](#key-files)
  - [Checkpointing](#checkpointing)
      - [Key Files](#key-files-1)
  - [Evaluation \& Visualization](#evaluation--visualization)
      - [Key Files](#key-files-2)
  - [Outputs](#outputs)
  - [Adding new methods](#adding-new-methods)
    - [New Backbone Architectures](#new-backbone-architectures)
    - [New Losses](#new-losses)
    - [New Metrics](#new-metrics)
    - [New Datasets](#new-datasets)
  - [Gridsearches and multiple jobs](#gridsearches-and-multiple-jobs)
    - [Gridsearches](#gridsearches)
    - [Running multiple jobs](#running-multiple-jobs)
  - [Run experiments with correlated factors of variation](#run-experiments-with-correlated-factors-of-variation)
  - [Run experiments with constrained factors of variation](#run-experiments-with-constrained-factors-of-variation)
  - [Complete Examples](#complete-examples)
    - [Training and Evaluating a BetaTCVAE](#training-and-evaluating-a-betatcvae)
    - [Training a BetaVAE with Hausdorff Factorized Support](#training-a-betavae-with-hausdorff-factorized-support)
    - [Training a pair-supervised AdaGVAE](#training-a-pair-supervised-adagvae)
    - [Training an AdaGVAE with correlated factors of variation](#training-an-adagvae-with-correlated-factors-of-variation)
    - [Training an AdaGVAE with constrained factors of variation](#training-an-adagvae-with-constrained-factors-of-variation)
    - [Running a large experimental gridsearch](#running-a-large-experimental-gridsearch)
  - [Running a large-scale gridsearch evaluating disentanglement methods over correlated FOVs](#running-a-large-scale-gridsearch-evaluating-disentanglement-methods-over-correlated-fovs)
  - [Introduction to fastargs](#introduction-to-fastargs)
  - [Citation](#citation)

---

## Installation and Requirements

This repository was tested and evaluated using

- Python 3.9
- PyTorch >=1.10 on GPU

and a suitable conda environment can be set up by running `conda env create -f environment.yaml`

If you wish to utilize logging with Weights & Biases (highly recommended!), simply visit https://wandb.ai to set up an account. Then simply run `wandb login` and enter your passkey. Alternatively, enter your key using `--log.wandb_key=<your_key>` when training, or simply set it directly in `parameters.py/WANDB_DEFAULT_KEY`.

If you want to make use of the automatic data download, you will also have to install `curl` on your machine, e.g. via `sudo apt install curl` on a `linux` machine.

---

## Implemented Methods and Metrics

This repository contains code (training / metrics / plotting) to train and evaluate various [VAE training objectives](#architecture-and-losses), measuring disentanglement using a [collection of metrics](#metrics) over a [collection of standard benchmarks](#benchmarks).

### Training Objectives and Architectures

This repository contains the following approaches to learn disentangled representation spaces, which can be loaded e.g. via `dent.loss_select(<loss_name>, **kwargs)` or the respective loss class in `dent/losses`:

- **Standard VAE Loss** from [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), see `dent/losses/betavae.py`.
- **Standard β-VAE<sub>H</sub>** from [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl), see `dent/losses/betavae.py`, but with higher `--betavae.beta`.
- **Annealed β-VAE<sub>B</sub>** from [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599), see `dent/losses/annealedvae.py`.
- **FactorVAE** from [Disentangling by Factorising](https://arxiv.org/abs/1802.05983), see `dent/losses/factorvae.py`.
- **β-TCVAE** from [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942), see `dent/losses/betatcvae.py`.
- **AdaGVAE** from [Weakly-Supervised Disentanglement Without Compromises
](https://arxiv.org/abs/2002.02886), see `dent/losses/adagvae.py`, with `--adagvae.average_mode=gvae`.
- **AdaMLVAE** from [Weakly-Supervised Disentanglement Without Compromises
](https://arxiv.org/abs/2002.02886), see `dent/losses/adagvae.py`, with `--adagvae.average_mode=mlvae`.

These objectives can be matched with the following default architectures, by passing the respective name via `--model.name=<model_name>`, or explicitly initializing via `dent.model_select(<model_name>, **kwargs)` / the respective class in `dent/models`:

- **MLP-based VAE** as used in [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942), see `dent/models/vae_chen_mlp.py`.
- **Convolutional VAE** as used in [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl), see `dent/models/vae_burgess.py`.
- **Convolutional VAE** as used in [Weakly-Supervised Disentanglement Without Compromises
](https://arxiv.org/abs/2002.02886), see `dent/models/vae_locatello.py`.
- **Convolutional VAE** as used in [Lost in Latent Space: Disentangled Models and the Challenge of Combinatorial Generalisation
](https://arxiv.org/abs/2002.02886), see `dent/models/vae_montero_small/large.py`, depending on the model size to use (they use larger models on average).
- **Convolutional VAE with Spatial Basis Decoder SBD**, e.g. with `vae_locatello`, as also used in e.g. [Lost in Latent Space: Disentangled Models and the Challenge of Combinatorial Generalisation
](https://arxiv.org/abs/2002.02886), see `dent/models/vae_locatello_sbd.py`.
- **Custom Convolutional VAE**, see `vae.py`. Allows simple mix-and-match between various default and custom encoder and decoder types.

### Metrics

The following metrics are provided to evaluate each method:

- **MIG:** Mutual Information Gap as proposed in [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942), with currently three differing implementations, based on continuous setups or discrete approximations. By default, uses MIG as computed in [`disentanglement_lib`](https://github.com/google-research/disentanglement_lib) (discrete). See `dent/metrics/mig.py`.
- **SAP:** Separated Attribute Predictability as proposed in [Variational Inference of Disentangled Latent Concepts from Unlabeled Observations](https://arxiv.org/abs/1711.00848). Has an implementation based on continuous and discrete approximations. Currently uses the discrete approximation as used in [`disentanglement_lib`](https://github.com/google-research/disentanglement_lib). See `dent/metrics/sap_d.py`.
- **DCI:** Disentanglement (or modularity), Completeness & Informativeness as proposed in [A Framework for the Quantitative Evaluation of Disentangled Representations](https://openreview.net/forum?id=By-7dz-AZ). Estimates mutual information of latent importance to predict a respective ground truth FoV, provided through some predictive model.
Runtime can differ between model backbones used. The default uses `sklearn.ensemble.GradientBoostingClassifier` as in [`disentanglement_lib`](https://github.com/google-research/disentanglement_lib), but is painfully slow. See `dent/metrics/dci_d.py`.
- **Modularity**: Computes modularity as proposed in [Learning deep disentangled embeddings with the f-statistic loss](https://arxiv.org/abs/1802.05312). Two differing implementations, either with discretization or not. By default, uses the former. See `dent/metrics/modularity_d.py`.
- **Simple Reconstruction Error**: Computes a straighforward, sample-averaged reconstruction error, see `dent/metrics/reconstruction.py`.

All metrics are defined as classes, which can be called directly or via `dent.metrics.utils.select_metric(<metric_name>)`. In addition, `dent.metrics.utils.MetricGroup` offers a simple metric computation aggregator. Simply define `--eval.metrics="['list','of','metric','names']"` or pass it to `MetricGroup`, and it will compute all metrics as efficiently as possible together given a respective dataloader and model.

### Benchmarks

These methods can be trained and evaluated on the following benchmark datasets, with all ground-truth factors of variation available:

- **DSprites** following [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl). See `datasets/dsprites.py`.
- **3D-Shapes** following [Disentangling by Factorising](http://proceedings.mlr.press/v80/kim18b.html). See `datasets/shapes3d.py`.
- **SmallNORB** following https://cs.nyu.edu/~ylclab/data/norb-v1.0-small. See `datasets/smallnorb.py`.
- **MPI3D** following [On the Transfer of Inductive Bias from Simulation to the Real World: a New Disentanglement Dataset](https://arxiv.org/abs/1906.03292). See `datasets/mpi3d.py`.
- **Cars3D** following [Deep Visual Analogy-Making](https://papers.nips.cc/paper/2015/hash/e07413354875be01a996dc560274708e-Abstract.html). See `datasets/cars3d.py`.

with some or no ground-truth factors of variation available:

- **Chairs** following [Learning to Generate Chairs, Tables and Cars
with Convolutional Networks](https://arxiv.org/pdf/1411.5928.pdf). See `datasets/chairs.py`.
- **CelebA** following [Deep Learning Face Attributes in the Wild](https://arxiv.org/abs/1411.7766). See `datasets/celeba.py`.
- **MNIST** following http://yann.lecun.com/exdb/mnist/. See `datasets/mnist.py`.
- **FashionMNIST** following [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747). See `datasets/fashionmnist.py`.

Note that every dataset implementation comes with its own download functionality, so no need to download and setup each dataset externally!

### Additional Features

Beyond key implementations, this repository

- allows for checkpointing at specificed training stages, resumption from last or chosen checkpoints,
- evaluation on last, specific or all checkpoints,
- visualization of latent space traversals (built on https://github.com/YannDubs/disentangling-vae),
- synchronization of training, evaluation and visualizations to Weights and Biases,
- and use of yaml-configurations alongside [fastargs](https://github.com/GuillaumeLeclerc/fastargs) for easy extendability. See both the linked repo and the bottom of this README for a (quick) tutorial on the usage.

---

## Training

**[For explicit examples please see the [Examples](#complete-examples) section]** A simply training run with pre-set configurations (stored in `/configs`) can be done using

```bash
python base_main.py --config-file=/configs/<name_of_config.yaml>
```

The config file overwrites the complete parameter set stored `parameters.py`, which also contains __all__ parameters used in this repository. Here, each parameter group is given a `Section`, which contains group-specific parameters (such as `data` or `betatcvae` for loss-specific parameters).

If one wishes to overwrite config parameters, this can be done through the command line:

```bash
python base_main.py --config-file=/configs/<name_of_config.yaml> --run.do="['train','visualize']" --log.wandb_mode=off --data.num_workers=12
```

which only includes training and visualization (as specified in `--run.do`), does not use W&B logging and using 12 workers. If only training should be done, simply use `--run.do="['train']"`.

In general, the following overwrite order exists: `env variable` > `cli argument` > `config yamls` > `parameters.py` where `>` denotes "overwrites".

#### Key Files

- `base_main.py`: Main scipt which calls the `fastargs` parameters (`parameters.py`) that can be then references throughout the repository using `from fastargs.decorators import param; @param('train.lr')`. In addition, calls training, evaluation and visualization scripts.
- `parameters.py`: Contains all `fastargs` parameters divided into parameter sections. Each parameter has a quick explanation for its usage. Similar to the `argparse.Namespace`, after loading the parameters in the main script (`base_main.py > import parameters`), every parameter can be access from everywhere using `@param('section.parametername')`. In addition to that, within a function, `config = fastargs.get_current_config()` can be utilized to get a dictionary with **all** parameters. If one wishes to update the config parameters internally, simply change the `config` dictionary and call `utils.helpers.overwrite_config(config)` to globally overwrite the fastargs parameters.
- `dent/trainers/basetrainer.py`: This is the main trainer file, which is called in `base_main.py`. It performs training iterations as well as storing respective checkpoints.

---

## Checkpointing
By default, a training run keeps a running checkpoint (`ckpt.pth.tar`) that corresponds to the most recent epoch. In addition, `train.checkpoint_every=N` can be specified to checkpoint every `N` epochs (`chkpt-<epoch>.pth.tar`) alongsige `train.checkpoint_first=N` to checkpoint the first `N` epochs (`chkpt-<epoch>.pth.tar`).

Using the `--run.restore_from` flag, the following flags allow for different restoration:

- `continue`: Use the most recent checkpoint to restore training. Also restores weights and biases logging if turned on (i.e. through `log.wandb_mode=run`).
- `n/a`: Don't restore even if a correct checkpoint is available.
- `<path_to_chkpt> `: Specific the exact checkpoint from which training should be restored.

The checkpointing ties in with how data is saved. Here, one can specify
`--log.project` to determine the overall experiment project folder (e.g. `architecture_evaluation`), and `log.group` to determine the specific experiment (e.g. `betavae_lr-0.5`). Finally, `log.group` also aggregates runs with differing seed values to make sure they are grouped together and also jointly visualized in Weights & Biases if turned on.

Note that if `--log.project` and `--log.group` are not specified explicitly, a custom storage name is generated based on the dataset, model and key training parameters used.

#### Key Files

- `dent/trainers/basetrainer.py > save_checkpoint`: Checkpointing is performed through the trainer class.
- `dent/trainers/basetrainer.py > initialize_from_checkpoint`: given a path to a checkpoint, the trainer class is reinitialized using the respective checkpoint data.

---

## Evaluation & Visualization

After training, evaluation is done on the stored checkpoints - this is either done directly after training when `--run.do="['train','eval']"` is used, or by using `--run.do="['eval']"` with `--run.restore_from=<path_to_chkpt or continue>` set respectively.

Evaluation is then done in sequence for all checkpoints of the form `chkpt-N.pth.tar` if `--eval.mode=all` is used, or only on `chkpt.pth.tar` if `--eval.mode=last` is used. If Weights&Biases is turned on, results are logged online.

For visualization, the exact same holds, by using `--run.do="['visualize']"` or `--run.do="['train','eval','visualize']".

#### Key Files

- `dent/evaluators/baseevaluator.py`: Contains the main evaluator class computing evaluation metrics and evaluation losses.
- `dent/metrics/utils.py > MetricGroup`: Main metric aggregator that collects all metrics of interest, computes shared requirements for each metric and then distributes these amongst each metric class. The metrics used are determined via `eval.metrics="['metric1','metric2',...]"`.
- `dent/utils/visualize.py`: Contains all visualization code. The type of visualization to include is determined via `--viz.plots`, which defaults to `all`, i.e. plotting all possible plots.

---

## Outputs
The following files are generated when (or after) training:

- `chkpt.pth.tar`, `chkpt-<epoch>.pth.tar`: Running checkpoint and checkpoint for respective epochs.,
- `specs.json`: `JSON` format of utilized parameters.
- `train_{}_log.csv`: log files for various training metrics that should be logged.

The following files are generated when (after) evaluation:

- `evaluation_results.txt`: summary of all evaluation results for each checkpoint,.
- `eval_{}_log.csv`: Various evaluation log files for metrics, call iteration, chkpt epoch and more.

The following files are generated when (after) visualization:

- `data_samples_{chkpt}.png`: Data sample visualization.
- `posterior_traversals_{chkpt}.gif/png`: Sample posterior traversals, saved as gif and png.
- `reconstruct_{chkpt}.png`: Sample reconstructions.

In addition, every function/class may also write to `foldername/wandb` if logging to Weights & Biases is turned on.

---

## Adding new methods

This section provides details on how to extend this codebase with new backbone models, VAE losses, evaluation metrics and datasets.

### New Backbone Architectures

Relevant files and folders:

- `dent/models`: This folder should contain key overall architectural setups such as `vae_burgess.py`, containing the convolutional VAE using by Burgess et al., or `vae_chen_mlp.py` - the MLP-based VAE using in Chen et al. Make sure to include the respective classes in `__init__.py` so they can be loaded more intuitively. Note that these models get their encoders (and optional decoders) from:
- `dent/encoders`: Main encoder architectures. Make sure to update `dent/encoders/__init__.py`.
- `dent/decoders`: Main decoder architectures. Make sure to update `dent/decoders/__init__.py`.

### New Losses

Relevant files and folders:

- `dent/losses`: Contains all loss/regularization to the base backbone models in `dent/models`. Each loss in this folder should ideally borrow from `dent/losses/baseloss.py > BaseLoss` (c.f. other examples in this folder). Make sure to include a call handle for each loss in `dent/losses/__init__.py`.
- `dent/losses/utils.py`: Contains standard losses such as reconstruction or KLD losses.

### New Metrics

Relevant files and folders:

- `dent/metrics`: To include a new metric, simply inherit from `basemetric/BaseMetric` (c.f. e.g. `mig.py`). Note that each metric has to return required inputs (stuff that should be computed or extracted beforehand such as embeddings (`samples_qzx` or `stats_qzx`) or ground truth factors (`gt_factors`)). Make sure to update `__init__/METRICS` and `__init__/select` with a respective metric handle.
- `dent/metrics/utils.py`: Contains the `MetricGroup` class, which is the main metric superclass - it loads all the metrics specified in `dent/metrics` and passed via `--eval.metrics`, aggregates requirements for each metrics, computes those and distributes these between metrics.

### New Datasets

Relevant files and folders:

- `datasets`: Contains all benchmark datasets. Compare to e.g. `dsprites.py` or `shapes3d.py` for datasets with given ground truth factors or `celeba.py` for datasets without. They should be generated differently, as the former has to provide access to information about the ground truth factors for the computation of various metrics.

---

## Gridsearches and multiple jobs

### Gridsearches

Assuming a `submitit`-compatible system (s.a. SLURM), this repository provides a simple way to perform gridsearches.

First, create a `.yaml`-file (such as `grid_searches/sample.yaml`) that lists each parameter and respective list of parameters to parse, s.a.:

```yaml
train:
  seed: '0-5' #Alternatively one can also use [0, 1, 2, 3, 4, 5].
  lr: [0.1, 0.01, 0.001]
```

Then, simply run

```bash
python gridsearch_det.py -g <path_to_gridsearch_yaml> -cfg <path_to_base_config_yaml> -ov [list of parameters to overwrite for ALL gridsearch elements]
```

For an example, refer to [***Running a large experimental gridsearch***](#running-a-large-experimental-gridsearch).
To understand available hyperparameters for the slurm submissions, refer to `gridsearch_det.py` and help-texts for each argument.

### Running multiple jobs

To schedule a long list of jobs (stores in some text-file s.a. `job_lists/jobs.txt`), simply run:

```bash
python multischedule.py --jobs example_job_list/jobs.txt
```

To understand available hyperparameters that help configure multiple sequential slurm submissions, make sure to look at help-texts for each argument in `multischedule.py`.

---

## Run experiments with correlated factors of variation

Given a standard unsupervised training setup such as

```bash
python base_main.py --config-file=configs/examples/betavae_shapes3d.yaml
```

or even a weakly-supervised approach such as

```bash
python base_main.py --config-file=configs/examples/adagvae_shapes3d.yaml
```

introducing correlations on the ground truth factors of variation trained on is very straightforward - simply define a correlations-`yaml`-file, such as `constraints/avail_correlations.yaml`, containing blocks like

```yaml
shapes3d_single_1_01:
  correlations:
    ('floorCol', 'wallCol'): 0.1  
```

which, in this particular case correlates the ground truth factors `floorCol` and `wallCol` with correlation strength `0.1`. The correlation strength follows directly from the correlation formulation in [On Disentangled Representations Learned From Correlated Data](https://arxiv.org/abs/2006.07886)

$$
\begin{equation}
p(c_1,c_2) \propto \exp\left(-\frac{(c_1 - c_2)^2}{2\sigma^2}\right)
\end{equation}
$$

with factors of variation $c_1$ and $c_2$. Given this block, we call this set of constraints `shapes3d_single_1_01`. A correlations-`.yaml`-file can contain multiple possible correlations, and can be used to adjust training protocols with one simple additional command:

```bash
python base_main.py --config-file=configs/examples/montero_betavae_shapes3d.yaml \
--constraints.correlations_file=constraints/avail_correlations.yaml:shapes3d_single_1_01
```

which looks at the `shapes3d_single_1_01`-block in `constraints/avail_correlations.yaml`. Note the use of `--constraints.correlations_file` to introduce correlations, which stands in constrast to the removal of specific combinations using `--constraints.file` described in the next section.

In general, you can choose any arbitrary set and number of correlations, with the correlations simply stacked on top of each other, such as for example

```yaml
shapes3d_three_1_mix:
  correlations:
    ('floorCol', 'wallCol'): 0.2
    ('objSize', 'objAzimuth'): 0.2
    ('objCol', 'objType'): 0.1  
```

which creates three pairs of correlations FOVs with varying degrees of correlation. If you want one factor (in this case `objType` to be correlated with every remaining one, simply utilize

```yaml
shapes3d_objType_confounder_01:
  correlations:
    ('objType', 'random'): 0.1
  repeat: ['count', 'max_num_single']
```

Generally, if you want to adjust the correlation behaviour further, simply check out `datasets/utils.py - get_dataloaders()` which starts from a standard PyTorch Dataloader and incrementally adds FOV constraining, correlating and pairing, depending on what is required. For correlations in particular, `provide_correlation_sampler()` will return a `WeightedRandomSampler` which is used to randomly sample specific FOV combinations based on the degree of correlation. The function also highlights other extended options to correlated FOVs.

For most of the experiments conducted in the paper, the set of utilized correlations are available in `constraints/avail_correlations`.

**Note** that this only impacts the data encountered during training. *Test data (i.e. the data evaluation metrics are computed on) is left unaltered*.

For a quick-run example, simply check out [***Training an AdaGVAE with constrained factors of variation***](#training-an-adagvae-with-constrained-factors-of-variation).

---

## Run experiments with constrained factors of variation

Given a standard unsupervised training setup such as

```bash
python base_main.py --config-file=configs/examples/betavae_shapes3d.yaml
```

or even a weakly-supervised approach such as

```bash
python base_main.py --config-file=configs/examples/betavae_shapes3d.yaml \
--train.loss=adagvae --train.supervision=pairs
```

constraining the support of ground truth factors of variation trained on is very straightforward - simply define a constraint `yaml`-file, such as `constraints/montero.yaml`, containing blocks like

```yaml
shapes3d_v1:
  constraints:
    objType: ['==', 4/4.]
    objCol: ['>=', 0.5]
  connect: and
```

which, in this particular case removes `objTypes == 1` with `objCol >= 0.5` from the respective training data. In this case, we call this set of constraints `shapes3d_v1`. A `constraints.yaml`-file can contain multiple possible constraints, and can be used to adjust training protocols with one simple additional command:

```bash
python base_main.py --config-file=configs/examples/betavae_shapes3d.yaml \
--constraints.file=constraints/montero.yaml:shapes3d_v1
```

which looks at the `shapes3d_v1`-block in `constraints/montero.yaml`. There are also a lot of other ways to remove specific type of FOV combinations from the training data, with a large set of examples available in `constraints/recombinations.yaml`, containing a quick explanation of all the different possible recombination settings, such as

```yaml
# Have 10% holes, no constraint to be connected.
to_element_random_constraint_repeats_perc_01:
  constraints:
    all: ['==', 'random']
  connect: and
  repeat: ['coverage', 0.1] 

# Have 10% holes, but some can be connected! 
to_elementrange_random_constraint_repeats_perc_01:
  constraints:
    all: [['==', '>=', '<='], 'random']
  connect: and
  repeat: ['coverage', 0.1] 
```

which removes $10\%%$ of factor combinations either completely at random, are such that some can be connected to a random degree. It also highlights one key element - whenever you utilize a list for a specific argument, the parser will randomly sample from this list. In this example, for `all` available FOVs, it will thus first randomly select a threshold value (`random`), and then whether remaining FOVs should be `==`, `>=` or `<=` said threshold value.

You may even select some specific factors to adjust and change the rest at random:

```yaml
to_element_some_fixed_some_random:
  constraints: 
    objType: ['==', 3/3.]
    objCol: ['==', 0.5]
    all: ['==', 'random']
  connect: and
```

or only select factor values from the hull of the combination hypercube:

```yaml
to_element_hull_random_range_1:
  constraints:
    random_1: ['==', [0, 1]]
    random_2: ['==', [0, 1]]
    random_3: ['==', [0, 1]]
    random_4: ['==', [0, 1]]
    random_5: ['==', [0, 1]]
    random_6: ['<=', 1]                    
  connect: and
```

**Note** that this only impacts the data encountered during training. *Test data (i.e. the data evaluation metrics are computed on) is left unaltered*.

For a quick-run example, simply check out [***Training an AdaGVAE with constrained factors of variation***](#training-an-adagvae-with-constrained-factors-of-variation).

---

## Complete Examples

### Training and Evaluating a BetaTCVAE

This section showcases how to use the existing codebase to train an unsupervised beta-TCVAE model on the 3D-Shapes datasets using Weights & Biases logging.
Here, all relevant hyperameters are packaged in the respective `.yaml`-config, but can of course be similarly alterated through the command line, which takes priority:

```bash
python base_main.py --config-file=configs/examples/betatcvae_shapes3d.yaml
```

By default, this fully trains the models, and follows it up with an evaluation of all available checkpoints while providing all possible visualization.

If every step should be done in sequence, simply call

```bash
python base_main.py --config-file=configs/examples/betatcvae_shapes3d.yaml \
--run.do="['train']"
python base_main.py --config-file=configs/examples/betatcvae_shapes3d.yaml \
--run.do="['eval','visualize']"
```

in order. Given the default `--run.restore_from='continue'` parameter, the second call simply looks for a matching checkpoint folder name and resumes from `chkpt.pth.tar`.

### Training a BetaVAE with Hausdorff Factorized Support

Given a default BetaVAE run

```bash
python base_main.py --config-file=configs/examples/betavae_shapes3d.yaml
```

you may either adjust the config-file directly, or simply extend it via

```bash
python base_main.py --config-file=configs/examples/betavae_shapes3d.yaml --train.loss=factorizedsupportvae --factorizedsupportvae.beta=0 --factorizedsupportvae.gamma=300
```

where `--factorizedsupportvae.beta` turns of the standard prior-matching term in the BetaVAE formulation, and `--factorizedsupportvae.gamma` adjusts the degree of support factorization.

### Training a pair-supervised AdaGVAE

To train a weakly-supervised (pair-supervision) AdaGVAE (or AdaMLVAE) is a very straightforward extension to the unsupervised setup, as some key parameters can be simply overwritting when using e.g. the `betatcvae_shapes3d.yaml`-config. Simply run

```bash
python base_main.py --config-file=configs/examples/betatcvae_shapes3d.yaml \
--train.loss=adagvae --train.supervision=pairs
```

with the standard train/eval/visualize setup. Alternatively of course, one can create an AdaGVAE-specific config file, such as `examples/adagvae_shapes3d.yaml`, and simply run

```bash
python base_main.py --config-file=configs/examples/adagvae_shapes3d.yaml
```

### Training an AdaGVAE with correlated factors of variation

As in this work we are interested in understanding how disentanglement methods perform when ground-truth factors of variation are correlated, we can also easily include this in the training process - simply take some default weakly-supervised (or unsupervised) command and run

```bash
python base_main.py --config-file=configs/examples/adagvae_shapes3d.yaml --constraints.correlations_file=constraints/avail_correlations.yaml:shapes3d_single_1_01
```

which uses the specific set of correlations dubbed `shapes3d_single_1_01` (following the correlation protocol introduced in [On Disentangled Representations Learned From Correlated Data](https://arxiv.org/abs/2006.07886)) and noted down in `constraints/prob_corrs_shapes3d.yaml`, and which correlations the factors `floorCol` and `wallCol` with strength `0.1` (see also [Run experiments with correlated factors of variation](#run-experiments-with-correlated-factors-of-variation)). In this case, the standard `adagvae_shapes3d.yaml` is augmented with specific correlations on the factors of variation available during training.

### Training an AdaGVAE with constrained factors of variation

To extend AdaGVAE training on training data with a constrained set of factors of variation, in which specific ground-truth factor combinations have been excluded, as used e.g. in [Lost in Latent Space: Disentangled Models and the Challenge of Combinatorial Generalisation](https://arxiv.org/abs/2002.02886), simply take the default weakly-supervised (or unsupervised) command and run

```bash
python base_main.py --config-file=configs/examples/adagvae_shapes3d.yaml \
--constraints.file=constraints/montero.yaml:shapes3d_v1
```

which uses the constraint-block `shapes3d_v1` defined in `constraints/montero.yaml` (see [Run experiments with constrained factors of variation](#run-experiments-with-constrained-factors-of-variation)). In this case, the standard `adagvae_shapes3d.yaml` is augmented with specific constraints on the factors of variation available during training.

### Running a large experimental gridsearch

For a hands-on example on running a large-scale gridsearch, we here replicate key comparisons from [Weakly-Supervised Disentanglement Without Compromises
](https://arxiv.org/abs/2002.02886), looking at the performance differences of weakly-supervised AdaGVAE versus unsupervised approaches on various metrics and benchmark datasets.

To do so, simply check out `grid_searches/locatello_weak_exps`, which contains a `base.yaml`, comprising the base configuration for all experiments, and `grid.yaml`, comprising a list of parameters to iterate through.
Given these to, simply run

```bash
python gridsearch_det.py \
-cfg grid_searches/locatello_weak_exps/base.yaml \
-g grid_searches/locatello_weak_exps/grid.yaml
```

which runs a large hyperparamter gridsearch over three benchmark datasets and multiple seeds, logging everything to a shared Weights & Biases folder.

---

## Running a large-scale gridsearch evaluating disentanglement methods over correlated FOVs

To re-run our large-scale gridsearch of unsupervised disentanglement baselines and those augmentation with support factorization as listed in our paper [Disentanglement of Correlated Factors via Hausdorff Factorized Support](https://openreview.net/forum?id=OKcJhpQiGiX), proceed as noted above, but instead utilize the following gridsearch-files:

```bash
# Run standard unsupervised disentanglement methods + HFS on Shapes3D
python gridsearch_det.py \
-cfg grid_searches/correlation_benchmark/base.yaml \
-g grid_searches/correlation_benchmark/shapes3d_grid.yaml

# Run standard unsupervised disentanglement methods + HFS on MPI3D
python gridsearch_det.py \
-cfg grid_searches/correlation_benchmark/base.yaml \
-g grid_searches/correlation_benchmark/mpi3d_grid.yaml

# Run standard unsupervised disentanglement methods + HFS on DSprites
python gridsearch_det.py \
-cfg grid_searches/correlation_benchmark/base.yaml \
-g grid_searches/correlation_benchmark/dsprites_grid.yaml
```

In these cases, for the setting in which Hausdorff Factorized Support is applied on top of existing frameworks, a very coarse framework- and dataset-independent gridsearch over the factorization weights $\gamma$ in the Hausdorff Factorized Support is performed - i.e. $\gamma\in[0.1, 1, 10, 100, 1000, 10000]$ (on top of e.g. $\beta\in[0, 1, 2, 4, 6, 8, 10, 12, 16]$), as different degrees of explicit disentanglement in e.g. $\beta$-VAE or $\beta$-TCVAE benefit from different degrees of explicit support factorization. However, even this coarse gridsearch should give strong performance already out-of-the-box.

These results can then be further refined for example using $\gamma\in[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]$ to compete with the much more finegrained parameter gridsearches done in other frameworks.

After training is completed, all results can be simply downloaded from the associated W&B repo or extracted from each checkpoint (though the former is much more straightforward), for example via

```python
import pickle
import tqdm
import wandb

def get_data(project):
    api = wandb.Api()
    runs = api.runs(project)
    info_list = []
    for run in tqdm.tqdm(runs, desc='Downloading data...'):
        config = {k:v for k,v in run.config.items() if not k.startswith('_')}
        info_dict = {'metrics':run.history(), 'config':config}
        info_list.append((run.name,info_dict))
    return info_list

load_from_disk = False
downloaded_data  = get_data("name_of_wandb_repo")
pickle.dump(downloaded, open('name_of_wandb_repo.pkl', 'wb'))
```

which returns a large collection of results that can be evaluated locally however desired.

---

**Finally**, with checkpoints saved, one can then also re-evaluate those however desired using `evaluate_multiple_checkpoints.py` (or `evaluate_single_checkpoint.py` for a singular checkpoint:

```bash
python evaluate_multiple_checkpoints.py --chkpt_folder=<path_to_checkpoint_folder>
```

which submits, for each checkpoint, a respective run to a SLURM cluster using `evaluate_single_checkpoint.py`.

**In addition**, to evaluate the transfer performance of given checkpoints trained with a specific training-time correlation against other test-time correlations, check out `correlation_transfer.py` and `correlation_transfer_single.py`.

Specifically, `correlation_transfer_single.py` takes a single checkpoint, or checkpoint folder with multiple seed variants of a checkpoint, a `--source_group`, which denotes the training correlation name as used in e.g. `constraints/avail_correlations.yaml`, as well as `--target_group`, which denotes the test correlation available in e.g. `constraints/avail_correlations.yaml`, where the model should be transferred to.

```bash
python correlation_transfer_single.py --chkpt_paths=<path_to_chkpt_folder> --dataset=<name_of_dataset> --source_group=<name_of_source_correlation_type> --target_group=<name_of_target correlation_type>
```

With this setup, and assuming a dictionary, here called `chkpt_folder_paths_dict` which has the structure 

```python
chkpt_folder_paths_dict = {
  'source_group': ['all_available_checkpoints_associated_with_said_source_group']
}
```

a collection of transfer experiments can then be run via e.g. (assuming again a SLURM compute system)

```python
import submitit
disable_tqdm = False

# Set up SLURM submitter.
log_folder = "log_test/%j"
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(
    name='job',
    timeout_min=300,
    slurm_partition='partition_name',
    tasks_per_node=1,
    nodes=1,
    gpus_per_node=1,
    cpus_per_task=10
)

# Aggregate all jobs.
jobs = []
with executor.batch():
    for dataset_name in dataset_names:
        avail_groups = list(chkpt_folder_paths_dict[dataset_name].keys())
        for i, source_group in tqdm.tqdm(enumerate(avail_groups), total=len(avail_groups), position=0, desc='Submitting Source Groups...', disable=disable_tqdm):
            for t, target_group in tqdm.tqdm(enumerate(avail_groups), total=len(avail_groups), position=1, desc='Submitting Target Groups', disable=True):               
                chkpt_paths = chkpt_folder_paths_dict[dataset_name][source_group]
                already_run = []
                for chkpt_path in chkpt_paths:
                    project_seed = chkpt_path.split('_s-')[-1].split('/')[0]
                    savename = f'corr_transfer_results/bvae__{dataset_name}__{source_group}__{target_group}_seed-{project_seed}.pkl'
                    already_run.append(not os.path.exists(savename))
                if any(already_run):
                    cmds = [
                        "python", "correlation_transfer_single.py", "--dataset_name", dataset_name, "--chkpt_path"]
                    cmds.extend(chkpt_paths)
                    cmds.extend(["--source_group", source_group, "--target_group", target_group, "--preemb", 'bvae'
                    ])
                    func = submitit.helpers.CommandFunction(cmds, verbose=True)
                    job = executor.submit(func)
                    jobs.append(job)    
print(f'Submitted {len(jobs)} jobs!')
```

---

**Quick Note:** A similar gridsearch for recombination-based experiments extending those done in [Lost in Latent Space: Disentangled Models and the Challenge of Combinatorial Generalisation](https://arxiv.org/abs/2002.02886) on e.g. Shapes3D can also be done via

```bash
python gridsearch_det.py \
-cfg grid_searches/montero_lost_exps/base.yaml \
-g grid_searches/montero_lost_exps/grid.yaml
```

where more and more complex recombination settings can be included by utilizing e.g. those defined in `constraints/recombinations.yaml`.

---

## Introduction to fastargs

[Fastargs](https://github.com/GuillaumeLeclerc/fastargs) is an argument & configuration management library used e.g. in https://github.com/libffcv/ffcv which as proven to be very useful for passing arguments and extending upon large codebases without worrying about through which order of functions and classes parameters have to be passed.

In particular, **all defineable parameters** used in repository are provided, alongside a quick explainer, in `parameters.py`. 
Here, every parameter is grouped inside of a `Section('section_name', 'section_explaination').params(...parameters...)` to more easily separate between different arguments. For a parameter `param` inside of a section `section`, the parameter can be adjusted from the command line via `--section.param=<value>`, or through various options within the scripts.

These special ways of parameter accessibility in scripts and functions with `fastargs` makes the attractiveness.
Given that `parameters.py` has been loaded once (`import parameters`) in the main scripts (i.e. e.g. in `base_main.py`), all parameters are easily accessible in each (nested) script and function.
For that, see e.g. `dent/losses/adagvae.py`. Here, we access each respective parameter using the `@param`-handle:

```python
from fastargs.decorators import param

class Loss(...):
  @param('adagvae.thresh_mode')
  @param('adagvae.beta')
  ...
  def __init__(self, thresh_mode, beta, ..., **kwargs):
    ...
```

Using this setup, the loss can be imported without having to explicitly pass arguments. In `base_main.py`, we would thus e.g. have 

```python
loss_f = dent.losses.adagvae.Loss()
```

which will also use arguments updated through the command line or a config-yaml.

Another way to access parameters is by simply doing

```python
import fastargs
config = fastargs.get_current_config()
# Can also be converted to a namespace:
config_namespace = config.get()
param_of_interest = config['section.param_of_interest']
```

which collects all available parameters under the current status. 
After some parameters have been globally updated, this will retrieve the updated set of parameters.

To update or add configuration parameters is slightly more complicated, but not much. In particular, this repo provides a very straightforward way to do so using `overwrite_config()` found in `utils/helpers.py`:

```python
import fastargs
config = fastargs.get_current_config()
pars_to_update = {
  'section_name.param_name': <value>
}

import utils.helpers
config = utils.helpers.overwrite_config(pars_to_update)
```

This globally updates the given arguments and returns an updated config dictionary.
Note that it is also possible to pass in novel arguments of the form `section_name.param_name`!

---

## Citing our work

If you find this work useful, please consider citing it via  

[Disentanglement of Correlated Factors via Hausdorff Factorized Support](https://openreview.net/forum?id=OKcJhpQiGiX), **Karsten Roth, Mark Ibrahim, Zeynep Akata, Pascal Vincent, Diane Bouchacourt,** *International Conference on Learning Representations (ICLR 2023)*, 2023.

```bibtex
@inproceedings{
roth2023disentanglement,
title={Disentanglement of Correlated Factors via Hausdorff Factorized Support},
author={Karsten Roth and Mark Ibrahim and Zeynep Akata and Pascal Vincent and Diane Bouchacourt},
booktitle={International Conference on Learning Representations (ICLR)},
year={2023},
url={https://openreview.net/forum?id=OKcJhpQiGiX}
}
```
