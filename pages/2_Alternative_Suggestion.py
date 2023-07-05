import streamlit as st
import numpy as np
from src.recommendation.ClickPredictor import ClickPredictor, RankingModule
from src.clustering.algorithm_wrappers.KMeansWrapper import KMeansWrapper
from src.clustering.utils import fit_reducer, umap_transform

from src.utils import load_headlines, \
    get_mind_id_from_index, generate_header, fit_standardizer, \
    standardize_data, set_session_state, get_wordcloud_from_attention, extract_unread, remove_old_files

### GENERAL PAGE INFO ###
st.set_page_config(
    page_title="badpun - Alternative Suggestion",
    layout="wide")

generate_header()
remove_old_files()

config = st.session_state['config']

### DATA LOADING ###
click_predictor = ClickPredictor(huggingface_url="josh-oo/news-classifier", commit_hash="c70d86ab3598c32be9466c5303231f5c6e187a2f")
ranking_module = RankingModule(click_predictor)

user_embedding = click_predictor.get_historic_user_embeddings()
scaler = fit_standardizer(user_embedding)
user_embedding = standardize_data(scaler, user_embedding)
reducer = fit_reducer(st.session_state['config']['UMAP'], user_embedding)
user_embedding = umap_transform(reducer, user_embedding)

set_session_state(user_embedding[3]) # todo replace


### 1. CLUSTERING AND SUGGESTION ####

model = KMeansWrapper()

model.train(user_embedding)
model.extract_representations(user_embedding)  # return tuple (clusterid, location)
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
unread_headlines_ind, unread_headlines = extract_unread(headlines)
article_recommendations = ranking_module.rank_headlines(unread_headlines_ind, unread_headlines, take_top_k=40)[:10]


### 3. Page Layout ###

left_column, right_column = st.columns(2)
left_column.header('Newsfeed')

### 3.1 Newsfeed ###

def button_callback_alternative(article_index, test):
    st.session_state.article_mask[article_index] = False


article_fields = [left_column.button(f"[{headlines.loc[article_index, 1]}] {article}", use_container_width=True,
                                     on_click=button_callback_alternative,
                                     args=(article_index, 0))
                  for button_index, (article, article_index, score) in
                  enumerate(article_recommendations)]  # sorry for ugly


### 3.2. INTERPRETING ###

right_column.header('Clustering')
model.visualize(user_embedding, [("Actual you", st.session_state.user), ("Feed you are seeing", exemplar_embedding)])
right_column.plotly_chart(model.figure)

# todo these can be precaclulated
right_column.header('Interpretation')

results = click_predictor.calculate_scores(list(headlines.loc[:, 3]), user_id=id)

wordcloud = get_wordcloud_from_attention(*results)

# Display the generated image:
right_column.image(wordcloud.to_array(), use_column_width="auto")
