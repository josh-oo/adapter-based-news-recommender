import numpy as np
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
