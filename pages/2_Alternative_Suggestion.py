import streamlit as st
import numpy as np
from src.recommendation.ClickPredictor import ClickPredictor, RankingModule
from src.clustering.algorithm_wrappers.KMeansWrapper import KMeansWrapper
from src.clustering.utils import fit_reducer, umap_transform

from src.utils import load_headlines, \
    get_mind_id_from_index, generate_header, fit_standardizer, \
    standardize_data, generate_wordcloud_category, set_session_state, get_words_from_attentions, \
    generate_wordcloud_deviation

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
original_user_embedding = user_embedding
scaler = fit_standardizer(user_embedding)
reducer = fit_reducer(st.session_state['config']['UMAP'], user_embedding)
user_embedding = umap_transform(reducer, user_embedding)

set_session_state(user_embedding[3]) # todo replace


### 1. CLUSTERING AND SUGGESTION ####

model = KMeansWrapper()

if model.dim_of_clustering == 'low_dim':
    model.train(user_embedding)
    model.extract_representations(user_embedding)  # return tuple (clusterid, location)
    prediction = model.predict(st.session_state.user)
else:
    model.train(original_user_embedding)
    model.extract_representations(original_user_embedding)  # return tuple (clusterid, location)
    #prediction = model.predict(user = click_predictor.get_personal_user_embedding().reshape(1, -1))
    prediction = model.predict(user = click_predictor.get_personal_user_embedding())

cluster_representant = model.interpret(prediction)
user_suggestion = model.suggest(cluster_representant, metric=int(config['Clustering']['SuggestionMetric']))

st.write(f"Your actual cluster is {prediction}. We recommend you to have a look at cluster {user_suggestion[0]}, "
         f"which is the feed you see by default. Choose any other "
         f"cluster below.")
number = st.number_input('Cluster', min_value=0, max_value=int(config['Clustering']['NoClusters'])-1, value=user_suggestion[0])

# get represenatnt of cluster chosen by user
exemplar_embedding, exemplar_index = model.get_cluster_representant(number)
#todo get id from suggestion
id = get_mind_id_from_index(exemplar_index)

### 2. NEWS RECOMMENDATIONS ###


headlines = load_headlines(config['DATA'])
unread_headlines_ind = np.nonzero(st.session_state.article_mask)[0]
unread_headlines = list(headlines.loc[:, 3][st.session_state.article_mask])
article_recommendations = ranking_module.rank_headlines(unread_headlines_ind, unread_headlines, user_id=id)


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
# todo color whole recommended cluster
model.visualize(user_embedding, [("Actual you", st.session_state.user), ("Feed you are seeing", exemplar_embedding)])
right_column.plotly_chart(model.figure)

# todo these can be precaclulated
right_column.header('Interpretation')
#todo what to pass
scores, word_deviations, personal_deviations = click_predictor.calculate_scores(list(headlines.loc[:, 3]), user_id=id)

c_word_deviations = get_words_from_attentions(word_deviations, personal_deviations)
# wordcloud = generate_wordcloud_category(model.labels, prediction)
wordcloud = generate_wordcloud_deviation(c_word_deviations)

# Display the generated image:
right_column.image(wordcloud.to_array(), use_column_width="auto")
