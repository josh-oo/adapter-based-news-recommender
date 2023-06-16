import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


@st.cache_data
def fit_standardizer(embeddings):
    return StandardScaler().fit(embeddings)


@st.cache_data
def standardize_data(_scaler, embeddings):
    return _scaler.transform(embeddings)


@st.cache_data
def load_data(path):
    return np.load(path)


@st.cache_data
def load_headlines(_config):
    # TODO place recommender system here
    headline_path = _config['HeadlinePath']
    return pd.read_csv(headline_path, header=None, sep='\t').loc[:int(_config['NoHeadlines']), 3]
