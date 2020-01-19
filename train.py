import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

"""
# MVP Training
Train a network to discriminate between treatment and non-treatment images
in the mini dataset.

# Todo:
- Data generator
- Train/test split
- Train using an encoder
- Validate

---
"""

"# Data Generator"


class HCSData(Dataset):
    """
    High Content Screening Dataset (BBBC022)
    """

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string) : file path to metadata csv
        """
        self.df = pd.read_csv(csv_file, index_col=0)
        self.root = Path("./data/")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Args:
            idx (int) : Index of item to grab
        """
        sample = self.df.iloc[idx]  # Grab sample at index idx

        x = self.__load_img__(sample)  # Load images
        y = sample['ROLE']  # Load label

        # Encode y labels to binary
        label_enc = {  # Dictionary of label encoding
            'mock': 0,
            'compound': 1
        }
        y = label_enc.get(y, 0)  # Get enc from label, default = 'mock'

        return x, y

    def __load_img__(self, sample):
        """
        Load each image channel for the given sample and stack them into a
        single 5-channel 2D image
        """
        plate = sample['PLATE']  # Get plate num.

        channels = {
            'FileHoechst': 'w1',
            'FileER': 'w2',
            'FileSyto': 'w3',
            'FilePh': 'w4',
            'FileMito': 'w5'
        }

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
        ph_path = self.root / \
            f"BBBC022_v1_images_{plate}w4/{sample['FilePh']}"
        ph = plt.imread(ph_path)
        mito_path = self.root / \
            f"BBBC022_v1_images_{plate}w5/{sample['FileMito']}"
        mito = plt.imread(mito_path)

        # Stack images in channel (x, y, c) dimension
        img = np.stack([hoechst, er, syto, ph, mito], axis=-1)

        return img


class HCSMini(HCSData):
    """
    A mini version of the BBBC022 dataset for rapid testing
    """

    def __init__(self, csv_file):
        """
        Modifies the root data directory so that it points to the mini dataset
        """
        super().__init__(csv_file)
        self.root = Path('./data/mini')


data = HCSMini('data/mini.csv')

idx = st.slider('Select sample:', 0, len(data))

x, y = data[idx]

n = st.slider('Select modality:', 0, 4)
plt.imshow(x[..., n])
st.pyplot()

"---"

"""
# Model
"""
