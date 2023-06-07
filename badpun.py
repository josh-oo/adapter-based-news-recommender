import configparser

import streamlit as st
import numpy as np
import umap

from src.clustering.algorithm_wrappers.AgglomerativeWrapper import AgglomorativeWrapper
from src.clustering.algorithm_wrappers.KMeansWrapper import KMeansWrapper
from src.clustering.algorithm_wrappers.OpticsWrapper import OpticsWrapper

l_small, l_right = st.columns([1, 2])
l_small.image('logo.png')
l_right.title('Balanced Article Discovery through Playful User Nudging')

#### SIDEBAR ######
st.sidebar.header('Options')
add_selectbox = st.sidebar.selectbox(
    'Choose a clustering algorithm:',
    ('KMeans', 'Agglomerative Clustering', 'OPTICS')
)

left_column, right_column = st.columns(2)

### DATA LOADING ###

@st.cache_data
def load_data(path):
    return np.load(path)

config = configparser.ConfigParser()
config.read('config.ini')
embedding_path = config['DATA']['UserEmbeddingPath']
test_path = config['DATA']['TestUserEmbeddingPath']
user_embedding = load_data(embedding_path)
test_embedding = load_data(test_path)
@st.cache_data
def umap_transform(train_data, test_data):
    N_NEIGHBORS= int(config['UMAP']['n_neighbors'])
    MIN_DIST= float(config['UMAP']['min_dist'])
    N_COMPONENTS= int(config['UMAP']['n_components'])
    METRIC=config['UMAP']['metric']
    fit = umap.UMAP(
        n_neighbors= N_NEIGHBORS,
        min_dist= MIN_DIST,
        n_components= N_COMPONENTS,
        metric=METRIC
    )
    user_reduced = fit.fit_transform(user_embedding)
    test_reduced = fit.transform(test_embedding)
    return user_reduced, test_reduced

user_red, user_test_red = umap_transform(user_embedding, test_embedding)

# TODO start newsfeed
left_column.header('Newsfeed')

### 2. CLUSTERING ####
right_column.header('Clustering')
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
prediction = model.predict(user_test_red[0])
right_column.markdown(f"**Your cluster**: {prediction}")
cluster_representant = model.interpret(prediction)
user_suggestion = model.suggest(cluster_representant)
right_column.markdown(f"Would you like to see a user from **Cluster {user_suggestion[0]}**?")
model.visualize(user_red, user_test_red[0], user_suggestion[1])
right_column.plotly_chart(model.figure)
