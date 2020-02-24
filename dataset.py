import pandas as pd
import numpy as np

import logging
import torch
from pathlib import Path

import PIL

import transforms as T

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)


class HCSData(torch.utils.data.Dataset):
    """
    High Content Screening Dataset (BBBC022)
    """

    def __init__(self, data, data_path="./data/raw/", transforms=None):
        """
        Args:
            data (DataFrame) : pandas dataframe of metadata
        """
        self.df = data
        self.df.columns = [
            'FileER',
            'FileHoechst',
            'FileMito',
            'FilePh',
            'FileSyto',
            'ROLE',
            'ID',
            'MMOL',
            'PLATE_MAP_NAME',
            'SMILES',
            'WELL',
            'PLATE',
            'COMPOUND',
            'SOURCE',
            'SITE',
            'TIME'
        ]
        self.root = Path(data_path)
        self.transforms = transforms

    @classmethod
    def from_csv(cls, csv_file, data_path):
        """
        Constructor to generate a dataset from a csv file
        """
        return cls(pd.read_csv(csv_file, index_col=0), data_path)

    @property
    def class_weights(self):
        a = len(self.df[self.df['ROLE'] == 'mock'])
        b = len(self.df[self.df['ROLE'] == 'compound'])
        max = np.max([a, b])
        weights = []
        for sample in self.df['ROLE']:
            if sample == 'mock':
                weights.append(max / a)
            else:
                weights.append(max / b)
        return weights

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Args:
            idx (int) : Index of item to grab
        """
        sample = self.df.iloc[idx]  # Grab sample at index idx

        x = self.__load_img__(sample)
        x = torch.from_numpy(x).type(torch.float)  # Load images
        if self.transforms is not None:
            x = self.transforms(x)

        y = sample['ROLE']  # Load label

        # Encode y labels to binary
        label_enc = {  # Dictionary of label encoding
            'mock': 0,
            'compound': 1
        }
        # Get enc from label, default = 'mock'
        y = label_enc.get(y, 0)

        return x, y

    def __one_hot_enc(self, label, n_labels):
        a = torch.zeros(n_labels, dtype=torch.int64)
        a[label] = 1
        return a.type(torch.float)

    def __crop_center__(self, img, cropx, cropy):
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]

    def __load_img__(self, sample):
        """
        Load each image channel for the given sample and stack them into a
        single 3-channel 2D image
        """
        plate = sample['PLATE']  # Get plate num.

        # Load each tiff file individually
        hoechst_path = self.root / \
            f"BBBC022_v1_images_{plate}w1/{sample['FileHoechst']}"
        hoechst = plt.imread(hoechst_path)
        er_path = self.root / \
            f"BBBC022_v1_images_{plate}w2/{sample['FileER']}"
        er = plt.imread(er_path)
        syto_path = self.root / \
            f"BBBC022_v1_images_{plate}w3/{sample['FileSyto']}"
        syto = plt.imread(syto_path)

        channels = [hoechst, er, syto]

        # crop and normalize each channel individually
        for i, c in enumerate(channels):
            c = self.__crop_center__(c, 512, 512)
            channels[i] = (c - c.mean()) / c.std()

        # Stack images in channel (x, y, c) dimension
        img = np.stack(channels).astype(np.float64())

        return img

    def split(self, ratio):
        """
        Split the dataset into two subsets defined by the ration
        ration (double) : dataset is split into sizes of [ration, 1-ratio]
        """
        try:
            assert (ratio < 1) & (ratio > 0), \
                "Train-test split should be greater than 0 and less than 1."

            # Stratify by class
            mock = self.df[self.df['ROLE'] == 'mock']
            compound = self.df[self.df['ROLE'] == 'compound']

            mock_idx = np.round(ratio * len(mock)).astype(np.int)
            comp_idx = np.round(ratio * len(compound)).astype(np.int)

            train = pd.concat([mock.iloc[:mock_idx], compound.iloc[:comp_idx]])
            test = pd.concat([mock.iloc[mock_idx:], compound.iloc[comp_idx:]])

            return self.__class__(train, self.root), self.__class__(test, self.root)

        except AssertionError as error:
            logging.exception(error)
