import configparser
import streamlit as st

from src.utils import generate_header, remove_old_files

### GENERAL PAGE INFO ###

st.set_page_config(
    page_title="badpun - Choose profile",
    layout="wide"
)

generate_header()
remove_old_files()

config = configparser.ConfigParser()
config.read('config.ini')
if 'config' not in st.session_state:
    st.session_state['config'] = config

### COLD START ###

st.write('To start off, choose a user which matches your interest most:')

columns = st.columns(3)
buttons = [column.button(f"User {i+1}", use_container_width=True) for i, column in enumerate(columns)]

# todo load users
users = [['sport'],['politics'], ['lifestyle']]

# # todo initialize as 1 in proper dimension
# if 'user' not in st.session_state:
#     st.session_state['user'] = []

# todo plug in when ready
# for user, button in zip(users, buttons):
#     if button:
#         st.session_state.cold_start = user

# todo clean modell files