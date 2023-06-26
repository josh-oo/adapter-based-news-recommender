import configparser
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

embedding_path = st.session_state['config']['DATA']['UserEmbeddingPath']
test_path = st.session_state['config']['DATA']['TestUserEmbeddingPath']
user_embedding = click_predictor.get_historic_user_embeddings()
# test_embedding = load_data(test_path)
# standardize data
scaler = fit_standardizer(user_embedding)
user_embedding = standardize_data(scaler, user_embedding)
# test_embedding = standardize_data(scaler, test_embedding)
# transform data
reducer = fit_reducer(st.session_state['config']['UMAP'], user_embedding)
user_red = umap_transform(reducer, user_embedding)
# user_test_red = umap_transform(reducer, test_embedding)

if 'user' not in st.session_state:
    st.session_state['user'] = user_red[3]  # todo replace

if 'user_old' not in st.session_state:
    st.session_state['user_old'] = st.session_state['user']

if 'article_mask' not in st.session_state:
    st.session_state['article_mask'] = np.array(
        [True] * (int(config['DATA']['NoHeadlines']) + 1))  # +1 because indexing in pandas is apparently different

left_column, right_column = st.columns(2)

### 1. NEWS RECOMMENDATIONS ###
left_column.header('Newsfeed')
left_column.write("Below, you see your personalized newsfeed.")


headlines = load_headlines(config['DATA'])
unread_headlines_ind = np.nonzero(st.session_state.article_mask)[0]
unread_headlines = list(headlines[st.session_state.article_mask])
article_recommendations = ranking_module.rank_headlines(unread_headlines_ind, unread_headlines)

import time
def button_callback(button_index, article_index, headline):
    start = time.time()
    # set article  and all previous as read
    for (unread_headline, i, s) in article_recommendations[:button_index]:
        st.session_state.article_mask[i] = False
        click_predictor.update_step(unread_headline, 0)

    # give postive feedback for clicked headline
    st.session_state.article_mask[article_index] = False
    click_predictor.update_step(headline, 1)
    # all previous non clicked articles are considered negative update steps
    # update user states
    st.session_state.user_old = st.session_state.user

    print(f"Update: {time.time()-start}")
    user = click_predictor.get_personal_user_embedding().reshape(1, -1)
    print(f"Replace: {time.time()-start}")
    user_std = standardize_data(scaler, user)
    print(f"Standardize: {time.time()-start}")

    user_rd = umap_transform(reducer, user_std)
    print(f"Transform: {time.time()-start}")

    st.session_state.user = user_rd[0]


article_fields = [left_column.button(article, use_container_width=True,
                                     on_click=button_callback,
                                     args=(button_index, article_index, article))
                  for button_index, (article, article_index, score) in
                  enumerate(article_recommendations)]  # sorry for ugly

### 2. CLUSTERING ####
right_column.header('Clustering')
right_column.write("Here you can see where you are in comparison to other users, and how your click behaviour "
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

right_column.markdown(f"**You are assigned to cluster** {prediction}")
model.visualize(user_red, [("You", st.session_state.user), ("Previous position", st.session_state.user_old)])
right_column.plotly_chart(model.figure)

# ### 2.2. INTERPRETING ###
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
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
right_column.pyplot(plt)
