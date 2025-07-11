import streamlit as st

pg = st.navigation([
    st.Page("pages/explorer.py", title="Experiment Explorer", icon="🔬"),
    st.Page("pages/registry.py", title="Model Registry", icon="📚"),
])

# Set the initial page configuration
st.set_page_config(
    page_title="Runelog",
    page_icon="📜",
    layout="wide"
)

# Run the navigation
pg.run()