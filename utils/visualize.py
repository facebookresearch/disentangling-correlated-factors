# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from math import ceil, floor
import random

from fastargs.decorators import param
import imageio
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from datasets import get_background
import utils

FPS_GIF = 12
TRAIN_FILE = "train_losses.log"
DECIMAL_POINTS = 3
GIF_FILE = "training.gif"
PLOT_NAMES = dict(generate_samples="samples.png",
                  data_samples="data_samples.png",
                  reconstruct="reconstruct.png",
                  traversals="traversals.png",
                  reconstruct_traverse="reconstruct_traverse.png",
                  gif_traversals="posterior_traversals.gif")

#### Base Visualization classes.
@param('data.name', 'dataset')
@param('viz.save_images')
@param('viz.loss_of_interest')
@param('viz.max_traversal')
@param('viz.upsample_factor')
class Visualizer():

    def __init__(self, model, write_dir, dataset, save_images, loss_of_interest,
                 max_traversal, upsample_factor):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.

        Parameters
        ----------
        model : disvae.vae.VAE

        dataset : str
            Name of the dataset.

        write_dir : str
            The directory that the model is saved to and where the images will
            be stored.

        save_images : bool, optional
            Whether to not save images and return a tensor.

        loss_of_interest : str, optional
            The loss type (as saved in the log file) to order the latent dimensions by and display.

        max_traversal: float, optional
            The maximum displacement induced by a latent traversal. Symmetrical
            traversals are assumed. If `m>=0.5` then uses absolute value traversal,
            if `m<0.5` uses a percentage of the distribution (quantile).
            E.g. for the prior the distribution is a standard normal so `m=0.45` c
            orresponds to an absolute value of `1.645` because `2m=90%%` of a
            standard normal is between `-1.645` and `1.645`. Note in the case
            of the posterior, the distribution is not standard normal anymore.

        upsample_factor : floar, optional
            Scale factor to upsample the size of the tensor
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.latent_dim = self.model.latent_dim
        self.max_traversal = max_traversal
        self.save_images = save_images
        self.write_dir = write_dir
        self.dataset = dataset
        self.upsample_factor = upsample_factor
        if loss_of_interest is not None:
            self.losses = read_loss_from_file(
                os.path.join(self.write_dir, TRAIN_FILE), loss_of_interest)

    def _get_traversal_range(self, mean=0, std=1):
        """Return the corresponding traversal range in absolute terms."""
        max_traversal = self.max_traversal

        if max_traversal < 0.5:
            max_traversal = (1 - 2 * max_traversal) / 2  # from 0.45 to 0.05
            max_traversal = stats.norm.ppf(max_traversal, loc=mean,
                                           scale=std)  # from 0.05 to -1.645

        # symmetrical traversals
        return (-1 * max_traversal, max_traversal)

    def _traverse_line(self, idx, n_samples, data=None):
        """Return a (size, latent_size) latent sample, corresponding to a traversal
        of a latent variable indicated by idx.

        Parameters
        ----------
        idx : int
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and idx = 7, then the 7th dimension
            will be traversed while all others are fixed.

        n_samples : int
            Number of samples to generate.

        data : torch.Tensor or None, optional
            Data to use for computing the posterior. Shape (N, C, H, W). If
            `None` then use the mean of the prior (all zeros) for all other dimensions.
        """
        if data is None:
            # mean of prior for other dimensions
            samples_qzx = torch.zeros(n_samples, self.latent_dim)
            traversals = torch.linspace(*self._get_traversal_range(),
                                        steps=n_samples)

        else:
            if data.size(0) > 1:
                raise ValueError(
                    "Every value should be sampled from the same posterior, but {} datapoints given."
                    .format(data.size(0)))

            with torch.no_grad():
                post_mean, post_logvar = self.model.encoder(data.to(self.device))['stats_qzx'].unbind(-1)
                samples_qzx = self.model.reparameterize(post_mean, post_logvar)['samples_qzx']
                samples_qzx = samples_qzx.cpu().repeat(n_samples, 1)
                post_mean_idx = post_mean.cpu()[0, idx]
                post_std_idx = torch.exp(post_logvar / 2).cpu()[0, idx]

            # travers from the gaussian of the posterior in case quantile
            traversals = torch.linspace(*self._get_traversal_range(
                mean=post_mean_idx, std=post_std_idx), steps=n_samples)

        for i in range(n_samples):
            samples_qzx[i, idx] = traversals[i]

        return samples_qzx

    def _save_or_return(self, to_plot, size, filename, is_force_return=False):
        """Create plot and save or return it."""
        to_plot = F.interpolate(to_plot, scale_factor=self.upsample_factor)

        if size[0] * size[1] != to_plot.shape[0]:
            raise ValueError("Wrong size {} for datashape {}".format(
                size, to_plot.shape))

        # `nrow` is number of images PER row => number of col
        kwargs = dict(nrow=size[1],
                      pad_value=(1 - get_background(dataset=self.dataset)))
        return_dict = {}
        if self.save_images and not is_force_return:
            filename = os.path.join(self.write_dir, filename)
            save_image(to_plot, filename, **kwargs)
            return_dict['path'] = filename
        else:
            return_dict['image'] = make_grid_img(to_plot, **kwargs)
        return return_dict

    def _decode_latents(self, latent_samples):
        """Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = latent_samples.to(self.device)
        return self.model.decoder(latent_samples)['reconstructions'].cpu()

    def generate_samples(self, size=(8, 8)):
        """Plot generated samples from the prior and decoding.

        Parameters
        ----------
        size : tuple of ints, optional
            Size of the final grid.
        """
        prior_samples = torch.randn(size[0] * size[1], self.latent_dim)
        generated = self._decode_latents(prior_samples)
        return self._save_or_return(generated.data, size,
                                    PLOT_NAMES["generate_samples"])

    def data_samples(self, data, size=(8, 8)):
        """Plot samples from the dataset

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints, optional
            Size of the final grid.
        """
        data = data[:size[0] * size[1], ...]
        return self._save_or_return(data, size, PLOT_NAMES["data_samples"])

    def reconstruct(self,
                    data,
                    size=(8, 8),
                    is_original=True,
                    is_force_return=False):
        """Generate reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints, optional
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even when `is_original`, so that upper
            half contains true data and bottom half contains reconstructions.contains

        is_original : bool, optional
            Whether to exclude the original plots.

        is_force_return : bool, optional
            Force returning instead of saving the image.
        """
        if is_original:
            if size[0] % 2 != 0:
                raise ValueError(
                    "Should be even number of rows when showing originals not {}"
                    .format(size[0]))
            n_samples = size[0] // 2 * size[1]
        else:
            n_samples = size[0] * size[1]

        with torch.no_grad():
            originals = data.to(self.device)[:n_samples, ...]
            recs = self.model(originals)['reconstructions']

        originals = originals.cpu()
        recs = recs.view(-1, *self.model.img_size).cpu()

        to_plot = torch.cat([originals, recs]) if is_original else recs
        return self._save_or_return(to_plot,
                                    size,
                                    PLOT_NAMES["reconstruct"],
                                    is_force_return=is_force_return)

    def traversals(self,
                   data=None,
                   n_per_latent=8,
                   n_latents=None,
                   loss_per_latent=None,
                   is_force_return=False):
        """Plot traverse through all latent dimensions (prior or posterior) one
        by one and plots a grid of images where each row corresponds to a latent
        traversal of one latent dimension.

        Parameters
        ----------
        data : bool, optional
            Data to use for computing the latent posterior. If `None` traverses
            the prior.

        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        is_force_return : bool, optional
            Force returning instead of saving the image.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim
        latent_samples = [
            self._traverse_line(dim, n_per_latent, data=data) for dim in range(self.latent_dim)
        ]
        decoded_traversal = self._decode_latents(
            torch.cat(latent_samples, dim=0))

        if loss_per_latent is not None:
            n_images, *other_shape = decoded_traversal.size()
            n_rows = n_images // n_per_latent
            decoded_traversal = decoded_traversal.reshape(
                n_rows, n_per_latent, *other_shape)
            decoded_traversal = sort_list_by_other(decoded_traversal, loss_per_latent)
            decoded_traversal = torch.stack(decoded_traversal, dim=0)
            decoded_traversal = decoded_traversal.reshape(
                n_images, *other_shape)

        decoded_traversal = decoded_traversal[range(n_per_latent * n_latents),
                                              ...]

        size = (n_latents, n_per_latent)
        sampling_type = "prior" if data is None else "posterior"
        filename = "{}_{}".format(sampling_type, PLOT_NAMES["traversals"])

        return self._save_or_return(decoded_traversal.data,
                                    size,
                                    filename,
                                    is_force_return=is_force_return)

    def reconstruct_traverse(self,
                             data,
                             is_posterior=True,
                             n_per_latent=8,
                             n_latents=None,
                             loss_per_latent=None,
                             is_show_text=False):
        """
        Creates a figure whith first row for original images, second are
        reconstructions, rest are traversals (prior or posterior) of the latent
        dimensions.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        n_per_latent : int, optional
            The number of points to include in the traversal of a latent dimension.
            I.e. number of columns.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        is_posterior : bool, optional
            Whether to sample from the posterior.

        is_show_text : bool, optional
            Whether the KL values next to the traversal rows.
        """
        n_latents = n_latents if n_latents is not None else self.model.latent_dim

        reconstructions = self.reconstruct(data[:2 * n_per_latent, ...],
                                           size=(2, n_per_latent),
                                           is_force_return=True)['image']
        traversals = self.traversals(data=data[0:1, ...] if is_posterior else None,
                                     loss_per_latent=loss_per_latent,
                                     n_per_latent=n_per_latent,
                                     n_latents=n_latents,
                                     is_force_return=True)['image']

        concatenated = np.concatenate((reconstructions, traversals), axis=0)
        concatenated = Image.fromarray(concatenated)

        if is_show_text:
            losses = sorted(self.losses, reverse=True)[:n_latents]
            labels = ['orig', 'recon'
                      ] + ["KL={:.4f}".format(l) for l in losses]
            concatenated = add_labels(concatenated, labels)

        filename = os.path.join(self.write_dir,
                                PLOT_NAMES["reconstruct_traverse"])
        concatenated.save(filename)
        return {'path': filename}

    def gif_traversals(self, data, loss_per_latent=None, n_latents=None, n_per_gif=15):
        """Generates a grid of gifs of latent posterior traversals where the rows
        are the latent dimensions and the columns are random images.

        Parameters
        ----------
        data : bool
            Data to use for computing the latent posteriors. The number of datapoint
            (batchsize) will determine the number of columns of the grid.

        n_latents : int, optional
            The number of latent dimensions to display. I.e. number of rows. If `None`
            uses all latents.

        n_per_gif : int, optional
            Number of images per gif (number of traversals)
        """
        n_images, _, _, width_col = data.shape
        width_col = int(width_col * self.upsample_factor)
        all_cols = [[] for c in range(n_per_gif)]
        for i in range(n_images):
            grid = self.traversals(data=data[i:i + 1, ...],
                                   loss_per_latent=loss_per_latent,
                                   n_per_latent=n_per_gif,
                                   n_latents=n_latents,
                                   is_force_return=True)['image']

            height, width, c = grid.shape
            padding_width = (width - width_col * n_per_gif) // (n_per_gif + 1)

            # split the grids into a list of column images (and removes padding)
            for j in range(n_per_gif):
                all_cols[j].append(grid[:, [(j + 1) * padding_width +
                                            j * width_col + i
                                            for i in range(width_col)], :])

        pad_values = (1 - get_background(dataset=self.dataset)) * 255
        all_cols = [
            concatenate_pad(cols, pad_size=2, pad_values=pad_values, axis=1)
            for cols in all_cols
        ]

        filename = os.path.join(self.write_dir, PLOT_NAMES["gif_traversals"])
        imageio.mimsave(filename, all_cols, fps=FPS_GIF)
        return {'path': filename}


#### Functional Helpers
def plot_pair_samples(train_loader, write_dir, num_samples=16, log_to_wandb=False):
    out = next(iter(train_loader))
    sample_idcs = np.random.choice(len(out[0][0]), num_samples, replace=False)
    base_imgs = out[0][0][sample_idcs]
    pair_imgs = out[0][1][sample_idcs]
    base_fcts = out[1][0][sample_idcs]
    pair_fcts = out[1][1][sample_idcs]
    import torchvision.utils
    base_grid = torchvision.utils.make_grid(base_imgs, nrows=4)
    pair_grid = torchvision.utils.make_grid(pair_imgs, nrows=4)
    base_smry = f'{train_loader.dataset.lat_names}\n'
    for fcts in base_fcts:
        base_smry += f'{[np.round(x, 3) for x in list(fcts.numpy())]}'
        base_smry += ' | '
    pair_smry = f'{train_loader.dataset.lat_names}\n'
    for fcts in pair_fcts:
        pair_smry += f'{[np.round(x, 3) for x in list(fcts.numpy())]}'
        pair_smry += ' | '
    if log_to_wandb:
        import wandb        
        wandb.log({'base_references': wandb.Image(base_grid, caption=base_smry)})
        wandb.log({'pair_references': wandb.Image(pair_grid, caption=pair_smry)})
    f, axes = plt.subplots(2, 1)
    axes[0].imshow(base_grid.numpy().transpose(1, 2, 0))
    axes[1].imshow(pair_grid.numpy().transpose(1, 2, 0))
    f.set_size_inches(10, 6)
    f.tight_layout()
    f.savefig(write_dir/'pairwise_samples.png')
    plt.close()

def get_samples(dataset, num_samples, logger, idcs=[]):
    """ Generate a number of samples from the dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset.

    num_samples : int, optional
        The number of samples to load from the dataset

    idcs : list of ints, optional
        List of indices to of images to put at the begning of the samples.
    """
    idcs += random.sample(range(len(dataset)), num_samples - len(idcs))
    samples = torch.stack([dataset[i][0] for i in idcs], dim=0)
    logger.info("Selected idcs: {}".format(idcs))

    return samples


def sort_list_by_other(to_sort, other, reverse=True):
    """Sort a list by an other."""
    return [el for _, el in sorted(zip(other, to_sort), reverse=reverse)]


# TO-DO: clean
def read_loss_from_file(log_file_path, loss_to_fetch):
    """ Read the average KL per latent dimension at the final stage of training from the log file.
        Parameters
        ----------
        log_file_path : str
            Full path and file name for the log file. For example 'experiments/custom/losses.log'.

        loss_to_fetch : str
            The loss type to search for in the log file and return. This must be in the exact form as stored.
    """
    EPOCH = "Epoch"
    LOSS = "Loss"

    logs = pd.read_csv(log_file_path)
    df_last_epoch_loss = logs[logs.loc[:, EPOCH] == logs.loc[:, EPOCH].max()]
    df_last_epoch_loss = df_last_epoch_loss.loc[
        df_last_epoch_loss.loc[:, LOSS].str.startswith(loss_to_fetch), :]
    df_last_epoch_loss.loc[:,
                           LOSS] = df_last_epoch_loss.loc[:, LOSS].str.replace(
                               loss_to_fetch, "").astype(int)
    df_last_epoch_loss = df_last_epoch_loss.sort_values(LOSS).loc[:, "Value"]
    return list(df_last_epoch_loss)


def add_labels(input_image, labels):
    """Adds labels next to rows of an image.

    Parameters
    ----------
    input_image : image
        The image to which to add the labels
    labels : list
        The list of labels to plot
    """
    new_width = input_image.width + 100
    new_size = (new_width, input_image.height)
    new_img = Image.new("RGB", new_size, color='white')
    new_img.paste(input_image, (0, 0))
    draw = ImageDraw.Draw(new_img)

    for i, s in enumerate(labels):
        draw.text(xy=(new_width - 100 + 0.005,
                      int((i / len(labels) + 1 / (2 * len(labels))) *
                          input_image.height)),
                  text=s,
                  fill=(0, 0, 0))

    return new_img


def make_grid_img(tensor, **kwargs):
    """Converts a tensor to a grid of images that can be read by imageio.

    Notes
    -----
    * from in https://github.com/pytorch/vision/blob/master/torchvision/utils.py

    Parameters
    ----------
    tensor (torch.Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
        or a list of images all of the same size.

    kwargs:
        Additional arguments to `make_grid_img`.
    """
    grid = make_grid(tensor, **kwargs)
    img_grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    img_grid = img_grid.to('cpu', torch.uint8).numpy()
    return img_grid


def get_image_list(image_file_name_list):
    image_list = []
    for file_name in image_file_name_list:
        image_list.append(Image.open(file_name))
    return image_list


def arr_im_convert(arr, convert="RGBA"):
    """Convert an image array."""
    return np.asarray(Image.fromarray(arr).convert(convert))


def plot_grid_gifs(filename, grid_files, pad_size=7, pad_values=255):
    """Take a grid of gif files and merge them in order with padding."""
    grid_gifs = [[imageio.mimread(f) for f in row] for row in grid_files]
    n_per_gif = len(grid_gifs[0][0])

    # convert all to RGBA which is the most general => can merge any image
    imgs = [
        concatenate_pad([
            concatenate_pad(
                [arr_im_convert(gif[i], convert="RGBA") for gif in row],
                pad_size,
                pad_values,
                axis=1) for row in grid_gifs
        ],
                        pad_size,
                        pad_values,
                        axis=0) for i in range(n_per_gif)
    ]

    imageio.mimsave(filename, imgs, fps=FPS_GIF)


def concatenate_pad(arrays, pad_size, pad_values, axis=0):
    """Concatenate lsit of array with padding inbetween."""
    pad = np.ones_like(arrays[0]).take(indices=range(pad_size),
                                       axis=axis) * pad_values

    new_arrays = [pad]
    for arr in arrays:
        new_arrays += [arr, pad]
    new_arrays += [pad]
    return np.concatenate(new_arrays, axis=axis)
