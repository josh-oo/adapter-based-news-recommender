import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.clustering.algorithm_wrappers.ClickPredictor import ClickPredictor, RankingModule
from src.clustering.algorithm_wrappers.KMeansWrapper import KMeansWrapper

from src.utils import load_headlines, \
    generate_wordcloud, get_mind_id_from_index, generate_header, load_preprocess_data

### GENERAL PAGE INFO ###
st.set_page_config(
    page_title="badpun - Alternative Suggestion",
    layout="wide")

generate_header()

config = st.session_state['config']

### DATA LOADING ###

user_red, user_test_red = load_preprocess_data()

if 'user' not in st.session_state:
    st.session_state['user'] = user_test_red[3] # todo

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

click_predictor = ClickPredictor("test")
ranking_module = RankingModule(click_predictor)

headlines = load_headlines(config['DATA'])
article_recommendations = ranking_module.rank_headlines(np.nonzero(st.session_state.article_mask)[0],
                                                        list(headlines[st.session_state.article_mask]),
                                                        user_id=id)


### 3. Page Layout ###

left_column, right_column = st.columns(2)
left_column.header('Newsfeed')

article_fields = [left_column.button(article, use_container_width=True) for index, article
                  in zip(article_recommendations[0], article_recommendations[1])] # wtf python

for index, article, button in zip(article_recommendations[0], article_recommendations[1], article_fields):
    if button:
        # todo negative clicks
        st.session_state.article_mask[index] = False

right_column.header('Clustering')
# todo color whole recommended cluster
model.visualize(user_red, [("Actual you", st.session_state.user), ("Feed you are seeing", exemplar_embedding)])
right_column.plotly_chart(model.figure)

### 2.2. INTERPRETING ###
wordcloud = generate_wordcloud(config, model.labels, number)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
right_column.pyplot(plt)
