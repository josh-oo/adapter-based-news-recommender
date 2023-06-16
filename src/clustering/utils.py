import streamlit as st
import umap

@st.cache_data
def fit_reducer(_config, train_data):
    N_NEIGHBORS= int(_config['n_neighbors'])
    MIN_DIST= float(_config['min_dist'])
    N_COMPONENTS= int(_config['n_components'])
    METRIC=_config['metric']
    fit = umap.UMAP(
        n_neighbors= N_NEIGHBORS,
        min_dist= MIN_DIST,
        n_components= N_COMPONENTS,
        metric=METRIC
    )
    return fit.fit(train_data)

@st.cache_data
def umap_transform(_reducer, embeddings):
    return _reducer.transform(embeddings)