import streamlit as st
import os

st.set_page_config(
    page_title='Welcome to the Llama Index Assistant',
    layout='wide',
    page_icon=':llama:'
)
style_head = "<style>h2 {text-align: center;} img {align: center;}</style>"
with st.columns(3)[1]:
    st.markdown(style_head, unsafe_allow_html=True)
    st.image(os.path.join("images", "header.jpg"), width=500)
    st.header("Welcome to your LlamaIndex Assistant")

st.text('Check refresh')
with st.sidebar:
    add_radio = st.radio(
        "Choose a data source",
        ("Langchain", "LlamaIndex")
    )
