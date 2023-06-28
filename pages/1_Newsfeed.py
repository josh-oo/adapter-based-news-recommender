import configparser
import time

import plotly
import streamlit as st
import numpy as np

from src.clustering.algorithm_wrappers.AgglomerativeWrapper import AgglomorativeWrapper
from src.clustering.algorithm_wrappers.ClickPredictor import ClickPredictor, RankingModule
from src.clustering.algorithm_wrappers.KMeansWrapper import KMeansWrapper
from src.clustering.algorithm_wrappers.OpticsWrapper import OpticsWrapper
from src.clustering.utils import umap_transform, fit_reducer
import matplotlib.pyplot as plt
from src.utils import fit_standardizer, standardize_data, load_data, load_headlines, \
    generate_header, load_preprocess_data, generate_wordcloud_category, generate_wordcloud_deviation

### GENERAL PAGE INFO ###

st.set_page_config(
    page_title="badpun - Newsfeed",
    layout="wide"
)

generate_header()

config = st.session_state.config

#### SIDEBAR ######
st.sidebar.header('Options')
add_selectbox = st.sidebar.selectbox(
    'Choose a clustering algorithm:',
    ('KMeans', 'Agglomerative Clustering', 'OPTICS')
)

### DATA LOADING ###
click_predictor = ClickPredictor(huggingface_url="josh-oo/news-classifier", commit_hash="1b0922bb88f293e7d16920e7ef583d05933935a9")
ranking_module = RankingModule(click_predictor)

user_embedding = click_predictor.get_historic_user_embeddings()
# standardize data
scaler = fit_standardizer(user_embedding)
user_embedding = standardize_data(scaler, user_embedding)
# transform data
reducer = fit_reducer(st.session_state['config']['UMAP'], user_embedding)
user_red = umap_transform(reducer, user_embedding)

if 'cold_start' not in st.session_state:
    st.session_state['cold_start'] = user_red[3]

if 'user' not in st.session_state:
    st.session_state['user'] = st.session_state['cold_start']  # todo replace


if 'article_mask' not in st.session_state:
    st.session_state['article_mask'] = np.array(
        [True] * (int(config['DATA']['NoHeadlines']) + 1))  # +1 because indexing in pandas is apparently different

left_column, right_column = st.columns(2)

### 1. NEWS RECOMMENDATIONS ###
left_column.header('Newsfeed')


headlines = load_headlines(config['DATA'])
unread_headlines_ind = np.nonzero(st.session_state.article_mask)[0]
unread_headlines = list(headlines[st.session_state.article_mask])
article_recommendations = ranking_module.rank_headlines(unread_headlines_ind, unread_headlines)

current_article = article_recommendations[0][0]
current_index = article_recommendations[0][1]


def handle_article(article_index, headline, read=True):
    start = time.time()

    st.session_state.article_mask[article_index] = False
    click_predictor.update_step(headline, read)

    print(f"Update: {time.time()-start}")
    user = click_predictor.get_personal_user_embedding().reshape(1, -1)
    print(f"Replace: {time.time()-start}")
    user_std = standardize_data(scaler, user)
    print(f"Standardize: {time.time()-start}")

    user_rd = umap_transform(reducer, user_std)
    print(f"Transform: {time.time()-start}")

    st.session_state.user = user_rd[0]


left_column.button(current_article, use_container_width=True, type="primary",
                   on_click=handle_article, args=(current_index, current_article, True))


def read_later():
    pass


ll, lr = left_column.columns(2, gap='large')
ll.button('Maybe later', use_container_width=True, on_click=read_later)

lr.button('Skip', use_container_width=True, on_click=handle_article, args=(current_index, current_article, False))

lower_left, lower_right = st.columns(2)

### 2. CLUSTERING ####
lower_left.header('Clustering')
lower_left.write("Here you can see where you are in comparison to other users, and how your click behaviour "
                 "influences your position.")

model = None
if add_selectbox == 'KMeans':
    model = KMeansWrapper()
elif add_selectbox == 'Agglomerative Clustering':
    model = AgglomorativeWrapper()
elif add_selectbox == 'OPTICS':
    model = OpticsWrapper()
else:
    raise ValueError

model.train(user_red)
model.extract_representations(user_red)  # return tuple (clusterid, location)
prediction = model.predict(st.session_state.user)

lower_left.markdown(f"**You are assigned to cluster** {prediction}")
model.visualize(user_red, [("You", st.session_state.user), ("Initial profile", st.session_state.cold_start)])
lower_left.plotly_chart(model.figure, use_container_width=True)

# ### 2.2. INTERPRETING ###
lower_right.header('Interpretation')
#todo what to pass
scores, word_deviations, personal_deviations = click_predictor.calculate_scores(list(headlines))

#todo how to merge
from collections import Counter
c_word_deviations = Counter()
# todo speed up
for headline_counter in word_deviations:
    c_word_deviations += Counter(headline_counter)
# wordcloud = generate_wordcloud_category(model.labels, prediction)
wordcloud = generate_wordcloud_deviation(c_word_deviations)

# Display the generated image:
lower_right.image(wordcloud.to_array(), use_column_width="auto")
