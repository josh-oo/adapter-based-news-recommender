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
    generate_wordcloud

st.set_page_config(
    page_title="badpun - Newsfeed",
    layout="wide"
)

l_small, l_right = st.columns([1, 2])
l_small.image('media/logo.png')
l_right.title('Balanced Article Discovery through Playful User Nudging')
#### SIDEBAR ######
st.sidebar.header('Options')
add_selectbox = st.sidebar.selectbox(
    'Choose a clustering algorithm:',
    ('KMeans', 'Agglomerative Clustering', 'OPTICS')
)
left_column, right_column = st.columns(2)

### DATA LOADING ###
embedding_path = st.session_state.config['DATA']['UserEmbeddingPath']
test_path = st.session_state.config['DATA']['TestUserEmbeddingPath']

user_embedding = load_data(embedding_path)  # todo get_historic_user_embeddings
test_embedding = load_data(test_path)

# standardize data
scaler = fit_standardizer(user_embedding)
user_embedding = standardize_data(scaler, user_embedding)
test_embedding = standardize_data(scaler, test_embedding)

# transform data
reducer = fit_reducer(st.session_state.config['UMAP'], user_embedding)
user_red = umap_transform(reducer, user_embedding)
user_test_red = umap_transform(reducer, test_embedding)

if 'user' not in st.session_state:
    st.session_state['user'] = user_test_red[3] # todo replace

if 'user_old' not in st.session_state:
    st.session_state['user_old'] = st.session_state['user']

if 'article_mask' not in st.session_state:
    st.session_state['article_mask'] = np.array([True]*(int(st.session_state.config['DATA']['NoHeadlines'])+1)) # +1 because indexing in pandas is apparently different


### 1. NEWS RECOMMENDATIONS ###
left_column.header('Newsfeed')
left_column.write("Below, you see your personalized newsfeed.")

click_predictor = ClickPredictor("test")
ranking_module = RankingModule(click_predictor)

headlines = load_headlines(st.session_state.config['DATA'])
article_recommendations = ranking_module.rank_headlines(np.nonzero(st.session_state.article_mask)[0], list(headlines[st.session_state.article_mask]))

article_fields = [left_column.button(article, use_container_width=True) for index, article
                  in zip(article_recommendations[0], article_recommendations[1])] # wtf python

for index, article, button in zip(article_recommendations[0], article_recommendations[1], article_fields):
    if button:
        # todo negative clicks
        st.session_state.article_mask[index] = False
        click_predictor.update_step(article, 1)
        # todo replace
        # st.session_state.user = click_predictor.get_personal_user_embedding()
        st.session_state.user_old = st.session_state.user
        st.session_state.user = user_test_red[index]

### 2. CLUSTERING ####
right_column.header('Clustering')
right_column.write("Here you can see, where you are in comparison to other users, and how your click behaviour "
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

wordcloud = generate_wordcloud(st.session_state.config, model.labels, prediction)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
right_column.pyplot(plt)
