import configparser

import streamlit as st
from wordcloud import WordCloud

from src.clustering.algorithm_wrappers.AgglomerativeWrapper import AgglomorativeWrapper
from src.clustering.algorithm_wrappers.KMeansWrapper import KMeansWrapper
from src.clustering.algorithm_wrappers.OpticsWrapper import OpticsWrapper
from src.clustering.utils import umap_transform, fit_reducer

from src.utils import fit_standardizer, standardize_data, load_data

st.set_page_config(
    page_title="badpun - Newsfeed",
    layout="wide"

)

#### SIDEBAR ######
st.sidebar.header('Options')
add_selectbox = st.sidebar.selectbox(
    'Choose a clustering algorithm:',
    ('KMeans', 'Agglomerative Clustering', 'OPTICS')
)
number = st.sidebar.slider("Choose distance of suggestion to user:", 1, 100)
left_column, right_column = st.columns(2)

### DATA LOADING ###

config = configparser.ConfigParser()
config.read('config.ini')
embedding_path = config['DATA']['UserEmbeddingPath']
test_path = config['DATA']['TestUserEmbeddingPath']

user_embedding = load_data(embedding_path)
test_embedding = load_data(test_path)

# standardize data
scaler = fit_standardizer(user_embedding)
user_embedding = standardize_data(scaler, user_embedding)
test_embedding = standardize_data(scaler, test_embedding)

# transform data
reducer = fit_reducer(config['UMAP'], user_embedding)
user_red = umap_transform(reducer, user_embedding)
user_test_red = umap_transform(reducer, test_embedding)

if 'user' not in st.session_state:
    st.session_state['user'] = []

# todo remove
st.session_state.user = user_test_red[3]

### 1. NEWS RECOMMENDATIONS ###
left_column.header('Newsfeed')

@st.cache_data
def get_articles_for_user():
    # TODO place recommender system here
    headline_path = config['DATA']['HeadlinePath']
    import pandas as pd
    headlines = pd.read_csv(headline_path, header=None, sep ='\t')
    return headlines.loc[:20,3]

article_recommendations = get_articles_for_user()

article_fields = [left_column.button(article, use_container_width=True) for article in article_recommendations]

for i, button in enumerate(article_fields):
    if button:
        # todo send info back
        print(article_recommendations[i])
        del button

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
prediction = model.predict(st.session_state.user)
cluster_representant = model.interpret(prediction)
user_suggestion = model.suggest(cluster_representant, metric=number)

right_column.markdown(f"**Your cluster**: {prediction}")
right_column.markdown(f"Would you like to see a user from **Cluster {user_suggestion[0]}**?")
model.visualize(user_red, st.session_state.user, user_suggestion[1])
right_column.plotly_chart(model.figure)


### 2.2. INTERPRETING ###
wordcloud = WordCloud().generate_from_frequencies(user)

# Display the generated image:
right_column.imshow(wordcloud, interpolation='bilinear')
right_column.axis("off")
right_column.show()
st.pyplot()
