import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.clustering.algorithm_wrappers.ClickPredictor import ClickPredictor, RankingModule
from src.clustering.algorithm_wrappers.KMeansWrapper import KMeansWrapper
from src.clustering.utils import fit_reducer, umap_transform

from src.utils import load_headlines, \
    get_mind_id_from_index, generate_header, load_preprocess_data, fit_standardizer, \
    standardize_data, generate_wordcloud_category

### GENERAL PAGE INFO ###
st.set_page_config(
    page_title="badpun - Alternative Suggestion",
    layout="wide")

generate_header()

config = st.session_state['config']

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

if 'user' not in st.session_state:
    st.session_state['user'] = user_red[3] # todo

if 'article_mask' not in st.session_state:
    st.session_state['article_mask'] = np.array([True]*(int(config['DATA']['NoHeadlines'])+1)) # +1 because indexing in pandas is apparently different


### 1. CLUSTERING AND SUGGESTION ####

model = KMeansWrapper()

model.train(user_red)
model.extract_representations(user_red)  # return tuple (clusterid, location)
prediction = model.predict(st.session_state.user)
cluster_representant = model.interpret(prediction)
user_suggestion = model.suggest(cluster_representant, metric=int(config['Clustering']['SuggestionMetric']))

st.write(f"Your actual cluster is {prediction}. We recommend you to have a look at cluster {user_suggestion[0]}, "
         f"which is the feed you see by default. Choose any other "
         f"cluster below.")
number = st.number_input('Cluster', min_value=0, max_value=int(config['Clustering']['NoClusters']), value=user_suggestion[0])

# get represenatnt of cluster chosen by user
exemplar_embedding, exemplar_index = model.get_cluster_representant(number)
#todo get id from suggestion
id = get_mind_id_from_index(exemplar_index)

### 2. NEWS RECOMMENDATIONS ###


headlines = load_headlines(config['DATA'])
unread_headlines_ind = np.nonzero(st.session_state.article_mask)[0]
unread_headlines = list(headlines[st.session_state.article_mask])
article_recommendations = ranking_module.rank_headlines(unread_headlines_ind, unread_headlines, user_id=id)


### 3. Page Layout ###

left_column, right_column = st.columns(2)
left_column.header('Newsfeed')

### 3.1 Newsfeed ###

def button_callback_alternative(button_index, test):
    # set article  and all previous as read
    for (a, i, s) in article_recommendations[:button_index+1]:
        st.session_state.article_mask[i] = False


article_fields = [left_column.button(article, use_container_width=True,
                                     on_click=button_callback_alternative,
                                     args=(button_index, 0))
                  for button_index, (article, article_index, score) in
                  enumerate(article_recommendations)]  # sorry for ugly


### 3.2. INTERPRETING ###

right_column.header('Clustering')
# todo color whole recommended cluster
model.visualize(user_red, [("Actual you", st.session_state.user), ("Feed you are seeing", exemplar_embedding)])
right_column.plotly_chart(model.figure)

# todo these can be precaclulated
wordcloud = generate_wordcloud_category(config, model.labels, number)
right_column.image(wordcloud.to_array())
