import streamlit as st
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path

import torch

import torchvision as tv


# """
# # MVP Training
# Train a network to discriminate between treatment and non-treatment images
# in the mini dataset.
#
# # Todo:
# - Data generator
# - Train/test split
# - Train using an encoder
# - Validate
#
# ---
# """
#
# "# Data Generator"


class HCSData(torch.utils.data.Dataset):
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

        x = torch.from_numpy(self.__load_img__(sample)).type(
            torch.float)  # Load images
        y = sample['ROLE']  # Load label

        # Encode y labels to binary
        label_enc = {  # Dictionary of label encoding
            'mock': 0,
            'compound': 1
        }
        # Get enc from label, default = 'mock'
        y = torch.tensor(label_enc.get(y, 0), dtype=torch.int64)

        return x, y

    def __load_img__(self, sample):
        """
        Load each image channel for the given sample and stack them into a
        single 5-channel 2D image
        """
        plate = sample['PLATE']  # Get plate num.

        # channels = {
        #     'FileHoechst': 'w1',
        #     'FileER': 'w2',
        #     'FileSyto': 'w3',
        #     'FilePh': 'w4',
        #     'FileMito': 'w5'
        # }

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
        img = np.stack([hoechst, er, syto, ph, mito])

        return img.astype(np.int64())


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


n_epochs = 1
batch_size = 4

data = HCSMini('data/mini.csv')
loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

# """
# # Model
# Find a basic pretrained model.
# """

label_classes = 2

net = tv.models.vgg16(pretrained=True, progress=True)
net.features[0] = torch.nn.Conv2d(5, 64, 3, stride=(1, 1), padding=(1, 1))
net.classifier[-1] = torch.nn.Linear(4096, label_classes, bias=True)

print(net)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

for epoch in range(n_epochs):
    for batch_n, (x, y) in tqdm(enumerate(loader)):

        optimizer.zero_grad()

        o = net(x)
        loss = criterion(o, y)

        loss.backward()
        optimizer.step()
