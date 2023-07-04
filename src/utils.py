import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
from collections import Counter
from wordcloud import STOPWORDS

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

@st.cache_data
def get_mind_id_from_index(id):
    user_mapping = json.load(open(st.session_state.config['DATA']['IdMappingPath']))
    return list(user_mapping.keys())[list(user_mapping.values()).index(id)]


def generate_wordcloud_from_user_category(labels, cluster_id):
    # todo @Mara
    # Opening JSON file
    user_mapping = json.load(open(st.session_state.config['DATA']['IdMappingPath']))
    user_category_frequ = load_normalized_category_frequencies(st.session_state.config['DATA']['UserCategoriesPath'],
                                                               user_mapping)

    index_current_cluster_points = (labels == cluster_id).nonzero()[0]
    user_ids_current_cluster = [key for key in user_mapping if user_mapping[key] in index_current_cluster_points]

    category_freq_current_cluster = user_category_frequ.loc[user_category_frequ['user'].isin(user_ids_current_cluster)]
    freq = category_freq_current_cluster.iloc[:, 2:].sum()

    return generate_wordcloud(freq)

@st.cache_data
def generate_wordcloud(word_dict):
    # todo @Mara
    return WordCloud(scale=3, min_word_length=3,
                     background_color="rgba(255, 255, 255, 0)", mode="RGBA") \
        .generate_from_frequencies(word_dict)

def generate_header():
    l_small, l_right = st.columns([1, 4])
    l_small.image('media/logo_dark.png', use_column_width='always')
    l_right.title('Balanced Article Discovery')
    l_right.title('through Playful User Nudging')


def set_session_state(emergency_user):
    if 'cold_start' not in st.session_state:
        st.session_state['cold_start'] = emergency_user
    if 'user' not in st.session_state:
        st.session_state['user'] = st.session_state['cold_start']
    if 'article_mask' not in st.session_state:
        st.session_state['article_mask'] = np.array(
            [True] * (int(st.session_state.config['DATA']['NoHeadlines']) + 1))  # +1 because indexing in pandas is apparently different

@st.cache_data
def get_words_from_attentions(word_deviations):
    STOPWORDS.update(",", ":", "-", "(", ")", "?")
    c_word_deviations = Counter()
    # todo speed up
    for i, headline_counter in enumerate(word_deviations):
        sorted_headline = Counter(headline_counter).most_common(3)
        sorted_headline = [(w, s) for (w, s) in sorted_headline if w not in STOPWORDS]
        c_word_deviations += dict(sorted_headline)
    return c_word_deviations


def extract_unread(headlines):
    unread_headlines_ind = np.nonzero(st.session_state.article_mask)[0]
    unread_headlines = list(headlines.loc[:, 3][st.session_state.article_mask])
    return unread_headlines_ind, unread_headlines

def get_wordcloud_from_attention(scores, word_deviations, personal_deviations):
    word_deviations = [word_dict for word_dict, score in zip(word_deviations, scores) if score > 0.5]

    c_word_deviations = get_words_from_attentions(word_deviations)
    return generate_wordcloud(c_word_deviations)