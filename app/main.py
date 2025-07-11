import streamlit as st

from app.components import render_sidebar

st.set_page_config(page_title="Runelog", page_icon="📜", layout="wide")

render_sidebar()

st.title("Welcome to Runelog!")
st.info("Select a view from the sidebar to get started.")
