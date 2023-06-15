import configparser
from wordcloud import WordCloud
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="badpun - Choose profile",
)

l_small, l_right = st.columns([1, 2])
l_small.image('logo.png')
l_right.title('Balanced Article Discovery through Playful User Nudging')

### COLD START ###

st.write('To start off, choose a user which matches your interest most:')

columns = st.columns(3)
buttons = [column.button(f"User {i+1}", use_container_width=True) for i, column in enumerate(columns)]

# todo load users
users = [['sport'],['politics'], ['lifestyle']]

# todo initialize as 1 in proper dimension
if 'user' not in st.session_state:
    st.session_state['user'] = []

for user, button in zip(users, buttons):
    if button:
        st.session_state.user = user

print(st.session_state.user)