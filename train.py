import pandas as pd
import numpy as np
from tqdm import tqdm
import click
import yaml

# import cli_args

import logging
import torchvision as tv
import torch
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)


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
        self.root = Path("./data/raw/")

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
        y = label_enc.get(y, 0)

        return x, y

    def __one_hot_enc(self, label, n_labels):
        a = torch.zeros(n_labels, dtype=torch.int64)
        a[label] = 1
        return a.type(torch.float)

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

        return img.astype(np.float64())

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


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True),
    default="./configs/params.yml"
)
def train(config_file='./data/mini.csv'):
    # Set up gpu/cpu device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # args = cli_args.parse()
    config_file = './configs/params.yml'
    # Parameters
    with open(config_file, 'r') as f:
        p = yaml.load(f)

    # Dataset
    data = HCSMini.from_csv('data/mini.csv')  # Load dataset
    train, test = data.split(0.8)  # Split data into train and test

    train_loader = torch.utils.data.DataLoader(  # Generate a training data loader
        train, batch_size=p['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(  # Generate a testing data loader
        test, batch_size=p['batch_size'], shuffle=False)

    # Define Model
    net = tv.models.vgg16(pretrained=False, progress=True)
    net.features[0] = torch.nn.Conv2d(5, 64, 3, stride=(1, 1), padding=(1, 1))
    net.classifier[-1] = torch.nn.Linear(4096, p['label_classes'], bias=True)
    # Move Model to GPU
    if torch.cuda.device_count() > 1:  # If multiple gpu's
        net = torch.nn.DataParallel(net)  # Parallelize
    net.to(device)  # Move model to device

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for (x, y) in data:
        print(y)

    tr_correct = 0
    tr_total = 0
    correct = 0
    total = 0
    # Training
    for epoch in range(p['n_epochs']):  # Iter through epochcs
        cum_loss = 0
        msg = f"Training epoch {epoch+1}: "
        ttl = len(train_loader)  # Iter through batches
        for batch_n, (X, Y) in enumerate(train_loader):
            x, y = X.to(device), Y.to(device)  # Move batch samples to gpu

            o = net(x)  # Forward pass
            optimizer.zero_grad()  # Reset gradients
            loss = criterion(o, y)  # Compute Loss
            loss.backward()  # Propagate loss, compute gradients
            scheduler.step()  # Update weights

            cum_loss += loss.item()

            # tqdm.write((
            #     f"Batch {batch_n+1}:"
            #     f"\tLoss: {loss.item():.4f}"
            #     f"\tPrediction: {o.argmax()}"
            #     f" \t Label: {y.item()}"
            # ))

            # PERFORM SOME VALIDATION METRIC
            _, predicted = torch.max(o.data, 1)
            tr_total += y.size(0)
            tr_correct += (predicted == y).sum().item()

        print(f"Training loss: {cum_loss:.2f}")

        with torch.no_grad():

            msg = f"Testing epoch {epoch+1}: "
            ttl = len(test_loader)  # Iter through batches
            for batch_n, (X, Y) in enumerate(test_loader):
                x, y = X.to(device), Y.to(device)  # Move batch samples to gpu
                o = net(x)  # Forward pass

                # PERFORM SOME VALIDATION METRIC
                _, predicted = torch.max(o.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            print(f"Epoch {epoch}:")
            if epoch % 10 == 0:
                print('Accuracy of the network on the train images: %d %%' % (
                    100 * tr_correct / tr_total))
                print('Accuracy of the network on the test images: %d %%' % (
                    100 * correct / total))
                correct = 0
                total = 0
                tr_correct = 0
                tr_total = 0


if __name__ == '__main__':
    train()
