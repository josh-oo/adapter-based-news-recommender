import configparser
import time
import streamlit as st
import umap
from scipy.spatial import KDTree
from src.recommendation.ClickPredictor import ClickPredictor, RankingModule
from src.clustering.KMeansWrapper import KMeansWrapper
from src.utils import load_headlines, \
    generate_header, set_session_state, extract_unread, \
    get_wordcloud_from_attention, remove_old_files, get_mind_id_from_index

### GENERAL PAGE INFO ###

st.set_page_config(
    page_title="badpun",
    layout="wide"
)

generate_header()
remove_old_files()

if 'config' not in st.session_state:
    config = configparser.ConfigParser()
    config.read('config.ini')
    st.session_state['config'] = config[config['DEFAULT']['Dimensionality']]

config = st.session_state['config']

### DATA LOADING ###
@st.cache_resource
def load_predictor():
    return ClickPredictor(huggingface_url="josh-oo/news-classifier", commit_hash="c70d86ab3598c32be9466c5303231f5c6e187a2f")


@st.cache_resource
def load_rm():
    return RankingModule(click_predictor)


@st.cache_resource
def load_kdtree():
    return KDTree(click_predictor.get_historic_user_embeddings())


@st.cache_resource
def fit_reducer():
    user_embedding = click_predictor.get_historic_user_embeddings()
    fit = umap.UMAP(
        n_neighbors=int(config['n_neighbors']),
        min_dist=float(config['min_dist']),
        n_components=int(config['n_components']),
        metric=config['metric']
    )
    return fit.fit(user_embedding)


@st.cache_resource
def get_kmeans_model():
    if config['Dimensionality'] == 'low':
        embeddings = user_embedding
    elif config['Dimensionality'] == 'high':
        embeddings = click_predictor.get_historic_user_embeddings()
    else:
        raise ValueError("Not a valid input for config['Clustering']['Dimensionality']")
    model = KMeansWrapper(embeddings)
    print(model.repr_indeces)
    return model


click_predictor = load_predictor()
ranking_module = load_rm()
kdtree = load_kdtree()
reducer = fit_reducer()


@st.cache_data
def umap_transform():
    return reducer.transform(click_predictor.get_historic_user_embeddings())


user_embedding = umap_transform()
model = get_kmeans_model()

set_session_state(user_embedding[3])  # todo replace

headlines = load_headlines()
unread_headlines_ind, unread_headlines = extract_unread(headlines)
if config['Dimensionality'] == 'low':
    prediction = model.predict(st.session_state.user)
elif config['Dimensionality'] == 'high':
    prediction = model.predict(user=click_predictor.get_personal_user_embedding())
else:
    raise ValueError("Not a valid input for config]")

exemplars = user_embedding[model.repr_indeces]

##### TABS ####

cold_start_tab, recommendation_tab, alternative_tab = st.tabs(["Reset User", "Personalized Recommendation", "Alternative Feeds"])

with cold_start_tab:
    st.write('To start off, choose a user which matches your interest most:')
    columns = st.columns(3)

    def set_user():
        st.session_state['clean'] = False

    buttons = [column.button(f"User {i + 1}", use_container_width=True, on_click=set_user) for i, column in enumerate(columns)]

    # # todo initialize as 1 in proper dimension
    # if 'user' not in st.session_state:
    #     st.session_state['user'] = []

    # todo plug in when ready
    # for user, button in zip(users, buttons):
    #     if button:
    #         st.session_state.cold_start = user

with recommendation_tab:
    ### LAYOUT ###
    left_column, right_column = st.columns([3, 1])

    news_tinder = left_column.container()

    lower_left, lower_right = st.columns(2)

    visualization = lower_left.container()
    interpretation = lower_right.container()

    ### 1. NEWS RECOMMENDATIONS ###
    start = time.time()

    article_recommendations = ranking_module.rank_headlines(unread_headlines_ind, unread_headlines, take_top_k=2)

    print(f"Get recommendation: {time.time() - start}")
    current_article = article_recommendations[0][0]
    current_index = article_recommendations[0][1]


    def handle_article(article_index, headline, read=1):
        st.session_state.article_mask[article_index] = False
        click_predictor.update_step(headline, read)  # online learning only performed on positive sample

        user = click_predictor.get_personal_user_embedding().reshape(1, -1)

        # todo is this ok?
        # ok, alternatvie is in the report
        if config['Dimensionality'] == 'low':
            user_rd = reducer.transform(user)[0]
        elif config['Dimensionality'] == 'high':
            _, neighbor = kdtree.query(user)
            user_rd = user_embedding[neighbor[0]]

        st.session_state.user = user_rd


    def read_later():
        st.session_state.article_mask[current_index] = False


    news_tinder.subheader(f"[{headlines.loc[current_index, 1].capitalize()}] :blue[{current_article}]")

    ll, lm, lr = news_tinder.columns(3, gap='large')

    ll.button('Skip', use_container_width=True, on_click=handle_article, args=(current_index, current_article, 0))
    lm.button('Maybe later', use_container_width=True, on_click=read_later)
    lr.button('Read', use_container_width=True, on_click=handle_article, type="primary",
              args=(current_index, current_article, 1))

    ### 2. CLUSTERING ####
    visualization.header(f"You are assigned to cluster {prediction}")

    model.visualize(user_embedding, exemplars,
                    [("You", st.session_state.user), ("Initial profile", st.session_state.cold_start)])
    visualization.plotly_chart(model.figure, use_container_width=True)

    # ### 2.2. INTERPRETING ###
    interpretation.header('Interpretation')
    start = time.time()

    results = click_predictor.calculate_scores(list(headlines.loc[:, 3]))
    wordcloud = get_wordcloud_from_attention(*results)
    print(f"Words: {time.time() - start}")

    # Display the generated image:
    interpretation.image(wordcloud.to_array(), use_column_width="auto")

with alternative_tab:
    ### 1. CLUSTERING AND SUGGESTION ####
    left_column, right_column = st.columns(2)

    left_column.write(f"Your actual cluster is {prediction}. Choose any other cluster below.")
    number = right_column.number_input('Cluster', min_value=0, max_value=int(config['NoClusters']) - 1,
                             value=prediction)

    ### 2. PAGE LAYOUT ###
    left, middle, right = st.columns(3)

    ### 2.1 Newsfeed ###
    left.header('Newsfeed')

    # get represenatnt of cluster chosen by user

    # todo get id from suggestion
    id = get_mind_id_from_index(model.repr_indeces[number])

    def button_callback_alternative(article_index, test):
        st.session_state.article_mask[article_index] = False


    article_recommendations = ranking_module.rank_headlines(unread_headlines_ind, unread_headlines, user_id=id,
                                                            take_top_k=10)

    article_fields = [left.button(f"[{headlines.loc[article_index, 1]}] {article}", use_container_width=True,
                                         on_click=button_callback_alternative,
                                         args=(article_index, 0))
                      for button_index, (article, article_index, score) in
                      enumerate(article_recommendations)]  # sorry for ugly

    ### 2.2. Clustering ###

    middle.header('Clustering')
    model.visualize(user_embedding, exemplars,
                    [("Actual you", st.session_state.user), ("Feed you are seeing", user_embedding[model.repr_indeces[number]])])
    middle.plotly_chart(model.figure)

    ### 2.3. INTERPRETATION ###
    # todo these can be precaclulated
    right.header('Interpretation')

    results = click_predictor.calculate_scores(list(headlines.loc[:, 3]), user_id=id)

    wordcloud = get_wordcloud_from_attention(*results)

    # Display the generated image:
    right.image(wordcloud.to_array(), use_column_width="auto")
