# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess

import gzip
import h5py
import numpy as np
import sklearn.preprocessing
import tarfile
import torchvision
import tqdm

import datasets
import datasets.base

class SmallNORB(datasets.base.DisentangledDataset):
    """SmallNORB Dataset from https://cs.nyu.edu/~ylclab/data/norb-v1.0-small.

    SmallNORB contains 50 toys over 5 categories (such as airplanes, trucks, cars), generated 
    under varying lighting, elevation and azimuth conditions.
    NOTE: SmallNORB has an official train/test split available, which is merged here.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    """
    urls = {
        "train":
        [
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz",
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz",
        ]
    }
    files = {"train": "smallnorb.npy"}
    lat_names = ('category', 'instance', 'elevation', 'rotation', 'lighting')
    lat_sizes = np.array([5, 5, 9, 18, 6])
    img_size = (1, 64, 64)
    background_color = datasets.COLOUR_WHITE
    lat_values = {
        'category': np.linspace(0, 1, 5),
        'instance': np.linspace(0, 1, 5),
        'elevation': np.linspace(0, 1, 9),
        'rotation': np.linspace(0, 1, 18), #rotations from 0 - 340 deg in 20 deg steps.
        'lighting': np.linspace(0, 1, 6)
    }

    def __init__(self, root='data/smallnorb/', **kwargs):
        super().__init__(root, [torchvision.transforms.ToTensor()], **kwargs)

        self.imgs = np.load(self.train_data.replace('.npy', '_data.npy')).transpose(0, 2, 3, 1)
        self.lat_values = np.load(self.train_data.replace('.npy', '_labels.npy'))
        self.lat_values = sklearn.preprocessing.minmax_scale(self.lat_values)
        
        if self.subset < 1:
            n_samples = int(len(self.imgs) * self.subset)
            subset = np.random.choice(len(self.imgs), n_samples, replace=False)
            self.imgs = self.imgs[subset]
            self.lat_values = self.lat_values[subset]

    @staticmethod
    def _load_mat_gz_file(filename):
        """Read SmallNORB data in the .mat.gz format.

        Script borrowed and adapted from: 
        https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/data/ground_truth/norb.py#L112.
        """
        with gzip.open(filename, "rb") as f:
            s = f.read()
            magic = int(np.frombuffer(s, "int32", 1))
            ndim = int(np.frombuffer(s, "int32", 1, 4))
            eff_dim = max(3, ndim)
            raw_dims = np.frombuffer(s, "int32", eff_dim, 8)
            dims = []
            for i in range(0, ndim):
                dims.append(raw_dims[i])

            dtype_map = {
                507333717: "int8",
                507333716: "int32",
                507333713: "float",
                507333715: "double"
            }
            data = np.frombuffer(s, dtype_map[magic], offset=8 + eff_dim * 4)
            data = data.reshape(tuple(dims))
        return data

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root, exist_ok=True)

        files_to_remove = []
        extracted_files = {}
        for url in type(self).urls["train"]:
            filename = self.train_data.replace(
                '.npy', url.split('96x96')[-1].split('.')[0].replace('-', '_') + '.mat.gz')
            subprocess.check_call([
                "curl", "-L", url, "--output", filename
            ])
            extracted_files[filename.split('smallnorb_')[-1]] = self._load_mat_gz_file(filename)
            files_to_remove.append(filename)

        # Aggregate training and test samples.
        data = np.concatenate([extracted_files['training_dat.mat.gz'], extracted_files['testing_dat.mat.gz']])
        data = data[:, 0]
        categories = np.concatenate([extracted_files['training_cat.mat.gz'], extracted_files['testing_cat.mat.gz']])
        info = np.concatenate([extracted_files['training_info.mat.gz'], extracted_files['testing_info.mat.gz']])

        # Merge category information and general properties into one label tensor.
        labels = np.column_stack([categories, info])        
        labels[:, 3] = labels[:, 3] / 2

        # Resize SmallNORB data to 64x64.
        import PIL
        data_64x64 = np.zeros((len(data), 1, 64, 64))
        for i in tqdm.tqdm(range(len(data)), desc='Resizing SmallNORB: 96x96 -> 64x64...'):
            img_96x96 = PIL.Image.fromarray(data[i])
            data_64x64[i] = np.array(torchvision.transforms.functional.resize(img_96x96, [64, 64]))[None, :]
        data_64x64 = data_64x64.astype(np.uint8)

        # Remove old files.
        for file in files_to_remove:
            os.remove(file)

        # Save data.
        np.save(os.path.join(self.root, self.files['train'].replace('.npy', '_data.npy')), data_64x64)
        np.save(os.path.join(self.root, self.files['train'].replace('.npy', '_labels.npy')), labels)

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 5, that gives the value of each factor of variation.
        """
        return self.transforms(self.imgs[idx]), self.lat_values[idx]
