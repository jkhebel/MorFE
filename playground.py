import torchvision as tv
import torch
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

"# BBBC 022 Dataset"

"""
- mock are negative controls
- MVP: distinguish between treatment/mock
"""


"## Metadata"


@st.cache  # load the csv and cache it for fast reloading
def load_data(path):
    return pd.read_csv(path, index_col=0)


df = load_data("./data/BBBC022.csv")
df.columns = [
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

df

"Mock vs Compound"
st.write(
    len(df[df['ROLE'] == 'mock']),
    len(df[df['ROLE'] == 'compound'])
)

"Get mock from the same plate? How many mock per plate?"
for i in range(5):
    plate = df[df['PLATE'] == df.iloc[i]['PLATE']]
    st.write(
        f"Plate {i}:",
        len(plate[plate['ROLE'] == 'mock']) / 9,
        len(plate[plate['ROLE'] == 'compound']) / 9
    )

"There are consistently 1 mock per 5 compound wells (64, 320) on each plate."

"""
For an MVP, I want to discriminate between mock and treatment cells. In order
to balance classes it probably makes sense to select 1:1 of each per plate when
training.
"""

"## Mini dataset"
"Take the first treatment and non-treatment well from plate 20585"
plate = 20585
mini = df[df['PLATE'] == plate]
mini = pd.concat([
    mini[mini['WELL'] == mini[mini['ROLE'] == 'compound'].iloc[0]['WELL']],
    mini[mini['WELL'] == mini[mini['ROLE'] == 'mock'].iloc[0]['WELL']]
])
mini
# mini.to_csv('./data/mini.csv')

"### Now try loading images..."

sample = mini.iloc[0]
sample

"""
Folders and the metadata aren't the same so find out which folder (w's)
correpsond to which modality:
- w1 = Hoechst
- w2 = ER
- w3 = Syto
- w4 = Ph
- w5 = Mito
"""

ch = 'w1'
path = Path(
    f"./data/mini/BBBC022_v1_images_{plate}{ch}/{sample['FileHoechst']}")
path
img = plt.imread(path)
plt.imshow(img)
st.pyplot()

ch = 'w2'
path = Path(
    f"./data/mini/BBBC022_v1_images_{plate}{ch}/{sample['FileER']}")
path
img = plt.imread(path)
plt.imshow(img)
st.pyplot()

ch = 'w3'
path = Path(
    f"./data/mini/BBBC022_v1_images_{plate}{ch}/{sample['FileSyto']}")
path
img = plt.imread(path)
plt.imshow(img)
st.pyplot()

ch = 'w4'
path = Path(
    f"./data/mini/BBBC022_v1_images_{plate}{ch}/{sample['FilePh']}")
path
img = plt.imread(path)
plt.imshow(img)
st.pyplot()

ch = 'w5'
mito_path = Path(
    f"./data/mini/BBBC022_v1_images_{plate}{ch}/{sample['FileMito']}")
mito_path

img = plt.imread(mito_path)

plt.imshow(img)
st.pyplot()

"##"

"## Compounds"

st.write('No. SMILES:', len(df['SMILES'].dropna().unique()))
st.write('No. Compounds:', len(df['COMPOUND'].dropna().unique()))
st.write('No. IDs:', len(df['ID'].dropna().unique()))

"This is inconsistent..."

# for smiles in df['SMILES'].dropna().unique():
#     id = df[df['SMILES'] == smiles]['ID'].unique()
#     compounds = df[df['SMILES'] == smiles]['COMPOUND'].unique()
#     if len(id) > 1 or len(compounds) > 1:
#         st.write(smiles, ':\n\t', id, '\n\t', compounds, '\n\n')
#
# df[df['PLATE'] == 20585]

"""
# Todo:
- start training!
- determine whether to split on smiles, compound names, or IDs
"""

"---"


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

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(tv.utils.make_grid(images))
st.pyplot()

outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
