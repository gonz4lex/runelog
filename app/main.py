import sys
import os

# This adds the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

from app.components import render_sidebar

st.set_page_config(page_title="Runelog", page_icon="📜", layout="wide")

render_sidebar()

st.title("Welcome to Runelog!")
st.info("Select a view from the sidebar to get started.")
