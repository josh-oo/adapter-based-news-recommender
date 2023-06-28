import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

from src.clustering.utils import fit_reducer, umap_transform


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
    return pd.read_csv(headline_path, header=None, sep='\t').loc[:int(_config['NoHeadlines']), [1,3]]


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
    return list(user_mapping.keys())[list(user_mapping.values()).index(id)]


def generate_wordcloud_category(labels, cluster_id):
    # todo @Mara
    # Opening JSON file
    user_mapping = json.load(open(st.session_state.config['DATA']['IdMappingPath']))
    user_category_frequ = load_normalized_category_frequencies(st.session_state.config['DATA']['UserCategoriesPath'],
                                                               user_mapping)

    index_current_cluster_points = (labels == cluster_id).nonzero()[0]
    user_ids_current_cluster = [key for key in user_mapping if user_mapping[key] in index_current_cluster_points]

    category_freq_current_cluster = user_category_frequ.loc[user_category_frequ['user'].isin(user_ids_current_cluster)]
    freq = category_freq_current_cluster.iloc[:, 2:].sum()

    return WordCloud(width = 1600,height = 1200,
                     background_color="rgba(255, 255, 255, 0)", mode="RGBA")\
        .generate_from_frequencies(freq)


def generate_wordcloud_deviation(word_dict):
    # todo @Mara
    return WordCloud(width=800, height=600,
                     background_color="rgba(255, 255, 255, 0)", mode="RGBA") \
        .generate_from_frequencies(word_dict)

def generate_header():
    l_small, l_right = st.columns([1, 4])
    l_small.image('media/logo_dark.png', use_column_width='always')
    l_right.title('Balanced Article Discovery')
    l_right.title('through Playful User Nudging')


def load_preprocess_data():
    embedding_path = st.session_state['config']['DATA']['UserEmbeddingPath']
    test_path = st.session_state['config']['DATA']['TestUserEmbeddingPath']
    user_embedding = load_data(embedding_path)  # todo get_historic_user_embeddings
    test_embedding = load_data(test_path)
    # standardize data
    scaler = fit_standardizer(user_embedding)
    user_embedding = standardize_data(scaler, user_embedding)
    test_embedding = standardize_data(scaler, test_embedding)
    # transform data
    reducer = fit_reducer(st.session_state['config']['UMAP'], user_embedding)
    user_red = umap_transform(reducer, user_embedding)
    user_test_red = umap_transform(reducer, test_embedding)
    return user_red, user_test_red


def set_session_state(emergency_user):
    if 'cold_start' not in st.session_state:
        st.session_state['cold_start'] = emergency_user
    if 'user' not in st.session_state:
        st.session_state['user'] = st.session_state['cold_start']
    if 'article_mask' not in st.session_state:
        st.session_state['article_mask'] = np.array(
            [True] * (int(st.session_state.config['DATA']['NoHeadlines']) + 1))  # +1 because indexing in pandas is apparently different
