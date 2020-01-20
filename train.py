import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path

import torch

import torchvision as tv

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
Find a basic pretrained model.
"""

net = tv.models.vgg16(pretrained=True, progress=True)

st.write(net)

"""
---
# CIFAR test
"""

transform = tv.transforms.Compose(
    [tv.transforms.ToTensor(),
     tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = tv.datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = tv.datasets.CIFAR10(root='./data', train=False,
                              download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(tv.utils.make_grid(images))
# print labels
st.pyplot()
st.write(' '.join('%5s' % classes[labels[j]] for j in range(4)))


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            break

print('Finished Training')
#
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# # print images
# imshow(tv.utils.make_grid(images))
# st.pyplot()
#
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# st.write('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                                  for j in range(4)))
