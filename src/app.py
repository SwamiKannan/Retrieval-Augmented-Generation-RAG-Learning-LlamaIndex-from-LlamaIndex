import streamlit as st
import os
from llama_index.memory import ChatMemoryBuffer
from query import get_index

st.set_page_config(
    page_title='Welcome to your Llama Index Assistant',
    layout='centered',
    page_icon=':llama:',
    initial_sidebar_state='auto',

)
# style_head = "<style>h2 {text-align: center;} img {align: center;}</style>"
# with st.columns(3)[1]:
#     st.markdown(style_head, unsafe_allow_html=True)
st.image(os.path.join("images", "app", "header.jpg"), width=600)
st.header("Welcome to your LlamaIndex Assistant")

st.text('Check refresh')
with st.sidebar:
    add_radio = st.radio(
        "Choose a data source",
        ("Langchain (WIP)", "LlamaIndex")
    )

# Create index and chat_engine
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
index = get_index()
db = index.as_chat_engine(
    chat_mode='context',
    memory=memory,
    prompt='You are a librarian chatbot helping users answer questions based on the LlamaIndex documentation and provide code examples from the documentation when required.',
    verbose=True)
if 'chat_engine' not in st.session_state.keys():
    st.session_state['chat_engine'] = db
