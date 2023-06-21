import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud


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


@st.cache_data
def load_normalized_category_frequencies(path, user_mapping):
    df = pd.read_csv(path)
    # load frequencies
    user_category_frequ = df.loc[df['user'].isin(user_mapping.keys())]
    del df
    # normalize
    user_category_frequ.iloc[:, 2:] = user_category_frequ.iloc[:, 2:] \
        .div(user_category_frequ
             .iloc[:, 2:].sum(axis=1), axis=0)

    return user_category_frequ


def get_mind_id_from_index(id):
    user_mapping = json.load(open(st.session_state.config['DATA']['IdMappingPath']))
    return user_mapping[id]


def generate_wordcloud(_config, labels, cluster_id):
    # Opening JSON file
    user_mapping = json.load(open(st.session_state.config['DATA']['IdMappingPath']))
    user_category_frequ = load_normalized_category_frequencies(st.session_state.config['DATA']['UserCategoriesPath'],
                                                               user_mapping)

    index_current_cluster_points = (labels == cluster_id).nonzero()[0]
    user_ids_current_cluster = [key for key in user_mapping if user_mapping[key] in index_current_cluster_points]

    category_freq_current_cluster = user_category_frequ.loc[user_category_frequ['user'].isin(user_ids_current_cluster)]
    freq = category_freq_current_cluster.iloc[:, 2:].sum()

    return WordCloud(width = 800,height = 600,
                     background_color="rgba(255, 255, 255, 0)", mode="RGBA")\
        .generate_from_frequencies(freq)
