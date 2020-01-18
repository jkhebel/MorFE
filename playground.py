import streamlit as st
import numpy as np
import pandas as pd
import cv2
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
