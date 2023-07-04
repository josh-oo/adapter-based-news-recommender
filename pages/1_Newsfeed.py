import time

import streamlit as st

from src.clustering.algorithm_wrappers.AgglomerativeWrapper import AgglomorativeWrapper
from src.recommendation.ClickPredictor import ClickPredictor, RankingModule
from src.clustering.algorithm_wrappers.KMeansWrapper import KMeansWrapper
from src.clustering.algorithm_wrappers.OpticsWrapper import OpticsWrapper
from src.clustering.utils import umap_transform, fit_reducer
from src.utils import fit_standardizer, standardize_data, load_headlines, \
    generate_header, set_session_state, extract_unread, \
    get_wordcloud_from_attention

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

### LAYOUT ###
left_column, right_column = st.columns([3,1])

news_tinder = left_column.container()

lower_left, lower_right = st.columns(2)

visualization = lower_left.container()
interpretation = lower_right.container()

### DATA LOADING ###
click_predictor = ClickPredictor(huggingface_url="josh-oo/news-classifier", commit_hash="1b0922bb88f293e7d16920e7ef583d05933935a9")
ranking_module = RankingModule(click_predictor)

user_embedding = click_predictor.get_historic_user_embeddings()
scaler = fit_standardizer(user_embedding)
user_embedding = standardize_data(scaler, user_embedding)
reducer = fit_reducer(st.session_state['config']['UMAP'], user_embedding)
user_embedding = umap_transform(reducer, user_embedding)

set_session_state(user_embedding[3]) # todo replace

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

    print(f"Update: {time.time()-start}")
    user = click_predictor.get_personal_user_embedding().reshape(1, -1)
    print(f"Replace: {time.time()-start}")
    user_std = standardize_data(scaler, user)
    print(f"Standardize: {time.time()-start}")

    user_rd = umap_transform(reducer, user_std)
    print(f"Transform: {time.time()-start}")

    st.session_state.user = user_rd[0]


news_tinder.subheader(f"[{headlines.loc[current_index, 1].capitalize()}] :blue[{current_article}]")


def read_later():
    st.session_state.article_mask[current_index] = False


ll, lm, lr = news_tinder.columns(3, gap='large')

ll.button('Skip', use_container_width=True, on_click=handle_article, args=(current_index, current_article, 0))
lm.button('Maybe later', use_container_width=True, on_click=read_later)
lr.button('Read', use_container_width=True, on_click=handle_article, type="primary", args=(current_index, current_article, 1))

### 2. CLUSTERING ####
visualization.header('Clustering')
visualization.write("Here you can see where you are in comparison to other users, and how your click behaviour "
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

model.train(user_embedding)
model.extract_representations(user_embedding)  # return tuple (clusterid, location)
prediction = model.predict(st.session_state.user)

visualization.markdown(f"**You are assigned to cluster** {prediction}")
model.visualize(user_embedding, [("You", st.session_state.user), ("Initial profile", st.session_state.cold_start)])
visualization.plotly_chart(model.figure, use_container_width=True)

# ### 2.2. INTERPRETING ###
interpretation.header('Interpretation')

results = click_predictor.calculate_scores(list(headlines.loc[:, 3]))

wordcloud = get_wordcloud_from_attention(*results)

# Display the generated image:
interpretation.image(wordcloud.to_array(), use_column_width="auto")
