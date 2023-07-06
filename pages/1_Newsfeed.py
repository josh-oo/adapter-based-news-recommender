import os
import time
import streamlit as st
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from src.clustering.algorithm_wrappers.AgglomerativeWrapper import AgglomorativeWrapper
from src.recommendation.ClickPredictor import ClickPredictor, RankingModule
from src.clustering.algorithm_wrappers.KMeansWrapper import KMeansWrapper
from src.clustering.utils import umap_transform, fit_reducer
from src.utils import fit_standardizer, standardize_data, load_headlines, \
    generate_header, set_session_state, extract_unread, \
    get_wordcloud_from_attention, remove_old_files

### GENERAL PAGE INFO ###

st.set_page_config(
    page_title="badpun - Newsfeed",
    layout="wide"
)

generate_header()
remove_old_files()
config = st.session_state.config

### LAYOUT ###
left_column, right_column = st.columns([3, 1])

news_tinder = left_column.container()

lower_left, lower_right = st.columns(2)

visualization = lower_left.container()
interpretation = lower_right.container()


### DATA LOADING ###
@st.cache_resource
def load_predictor():
    return ClickPredictor(huggingface_url="josh-oo/news-classifier",
                          commit_hash="1b0922bb88f293e7d16920e7ef583d05933935a9")

@st.cache_resource
def load_rm():
    return RankingModule(click_predictor)

@st.cache_resource
def load_kdtree():
    return KDTree(click_predictor.get_historic_user_embeddings())

click_predictor = load_predictor()
ranking_module = load_rm()
kdtree = load_kdtree()

user_embedding = click_predictor.get_historic_user_embeddings()
reducer = fit_reducer(st.session_state['config']['UMAP'], user_embedding)
user_embedding = umap_transform(reducer, user_embedding)

set_session_state(user_embedding[3])  # todo replace

### 1. NEWS RECOMMENDATIONS ###

headlines = load_headlines(config['DATA'])
unread_headlines_ind, unread_headlines = extract_unread(headlines)
article_recommendations = ranking_module.rank_headlines(unread_headlines_ind, unread_headlines, take_top_k=40)
current_article = article_recommendations[0][0]
current_index = article_recommendations[0][1]


def handle_article(article_index, headline, read=1):
    start = time.time()
    st.session_state.article_mask[article_index] = False
    click_predictor.update_step(headline, read)

    print(f"Update: {time.time() - start}")
    user = click_predictor.get_personal_user_embedding().reshape(1, -1)
    print(f"Replace: {time.time() - start}")

    # todo is this ok?
    _, neighbor = kdtree.query(user)
    user_rd = user_embedding[neighbor[0]].reshape(1, -1)
    print(f"Transform: {time.time() - start}")

    st.session_state.user = user_rd[0]


def read_later():
    st.session_state.article_mask[current_index] = False


news_tinder.subheader(f"[{headlines.loc[current_index, 1].capitalize()}] :blue[{current_article}]")


ll, lm, lr = news_tinder.columns(3, gap='large')

ll.button('Skip', use_container_width=True, on_click=handle_article, args=(current_index, current_article, 0))
lm.button('Maybe later', use_container_width=True, on_click=read_later)
lr.button('Read', use_container_width=True, on_click=handle_article, type="primary",
          args=(current_index, current_article, 1))

### 2. CLUSTERING ####
visualization.header('Clustering')
visualization.write("Here you can see where you are in comparison to other users, and how your click behaviour "
                    "influences your position.")


@st.cache_resource
def get_kmeans_model():
    if config['Clustering']['Dimensionality'] == 'low':
        embeddings = user_embedding
    elif config['Clustering']['Dimensionality'] == 'high':
        embeddings = click_predictor.get_historic_user_embeddings()
    else:
        raise ValueError("Not a valid input for config['Clustering']['Dimensionality']")
    model = KMeansWrapper()
    model.train(embeddings)
    model.extract_representations(embeddings)  # return tuple (clusterid, location)
    print(model.representants)
    return model

model = get_kmeans_model()
if config['Clustering']['Dimensionality'] == 'low':
    prediction = model.predict(st.session_state.user)
elif config['Clustering']['Dimensionality'] == 'high':
    prediction = model.predict(user=click_predictor.get_personal_user_embedding())
else:
    raise ValueError("Not a valid input for config['Clustering']['Dimensionality']")

visualization.markdown(f"**You are assigned to cluster** {prediction}")
model.visualize(user_embedding, [("You", st.session_state.user), ("Initial profile", st.session_state.cold_start)])
visualization.plotly_chart(model.figure, use_container_width=True)

# ### 2.2. INTERPRETING ###
interpretation.header('Interpretation')

results = click_predictor.calculate_scores(list(headlines.loc[:, 3]))

wordcloud = get_wordcloud_from_attention(*results)

# Display the generated image:
interpretation.image(wordcloud.to_array(), use_column_width="auto")
