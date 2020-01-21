from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torchvision as tv

import logging

logging.basicConfig(level=logging.DEBUG)


class HCSData(torch.utils.data.Dataset):
    """
    High Content Screening Dataset (BBBC022)
    """

    def __init__(self, data):
        """
        Args:
            data (DataFrame) : pandas dataframe of metadata
        """
        self.df = data
        self.root = Path("./data/")

    @classmethod
    def from_csv(cls, csv_file):
        """
        Constructor to generate a dataset from a csv file
        """
        return cls(pd.read_csv(csv_file, index_col=0))

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

            return self.__class__(train), self.__class__(test)

        except AssertionError as error:
            logging.exception(error)


class HCSMini(HCSData):
    """
    A mini version of the BBBC022 dataset for rapid testing
    """

    def __init__(self, data):
        """
        Modifies the root data directory so that it points to the mini dataset
        """
        super().__init__(data)
        self.root = Path('./data/mini')


# Set up gpu/cpu device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
n_epochs = 1
batch_size = 4
label_classes = 2

# Dataset
data = HCSMini.from_csv('data/mini.csv')  # Load dataset
train, test = data.split(0.8)  # Split data into train and test

len(train)
len(test)

len(data)

train_loader = torch.utils.data.DataLoader(  # Generate a training data loader
    data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(  # Generate a testing data loader
    data, batch_size=batch_size, shuffle=True)

# Define Model
net = tv.models.vgg16(pretrained=True, progress=True)
net.features[0] = torch.nn.Conv2d(5, 64, 3, stride=(1, 1), padding=(1, 1))
net.classifier[-1] = torch.nn.Linear(4096, label_classes, bias=True)
# Move Model to GPU
if torch.cuda.device_count() > 1:  # If multiple gpu's
    net = torch.nn.DataParallel(net)  # Parallelize
net.to(device)  # Move model to device

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

# Training
for epoch in range(n_epochs):  # Iter through epochcs
    epoch_loss = 0
    for batch_n, (X, Y) in tqdm(enumerate(train_loader)):  # Iter through batch
        x, y = X.to(device), Y.to(device)  # Move batch samples to gpu

        optimizer.zero_grad()  # Reset gradients

        o = net(x)  # Forward pass

        loss = criterion(o, y)  # Compute Loss
        loss.backward()  # Propagate loss, compute gradients
        optimizer.step()  # Update weights

        epoch_loss += loss  # cumulative Loss

    logging.info(epoch_loss)
