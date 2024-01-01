import streamlit as st
from web_function import load_data
from Tabs import home, predict, visualize

Tabs = {
    "Home" : home,
    "Prediction" : predict,
    "Visualization" : visualize
}

st.sidebar.title("Navigasi")

page = st.sidebar.radio("Pages", list(Tabs.keys()))

df, x, y = load_data()

if page in ["Prediction", "Visualization"]:
    Tabs[page].app(df,x,y)
else:
    Tabs[page].app()