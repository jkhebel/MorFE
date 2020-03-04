import streamlit as st
import pandas as pd

"# Hello World!"


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

df.iloc[0]
