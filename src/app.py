import streamlit as st
import os
from llama_index.memory import ChatMemoryBuffer
from query import get_index, qa, process_metadata
from css import page_bg_img

# Setting layout
st.set_page_config(
    page_title='Welcome to your Llama Index Assistant',
    layout='centered',
    page_icon=':llama:',
    initial_sidebar_state='auto',

)

st.markdown(page_bg_img, unsafe_allow_html=True)

st.image(os.path.join("images", "app", "header.jpg"), width=600)
st.header("Welcome to your LlamaIndex Assistant")

with st.sidebar:
    add_radio = st.radio(
        "Choose a data source",
        ("LlamaIndex", "Langchain (WIP)")
    )


def create_message(role, content):
    assert role == 'user' or role == "assistant"
    return {"role": role, "content": content}


# Create index and chat_engine
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
with st.spinner('Loading index....'):
    index = get_index()
    print('LLM:', index.service_context.llm)
db = index.as_chat_engine(
    chat_mode='context',
    memory=memory,
    prompt='You are a librarian chatbot helping users answer questions based on the LlamaIndex documentation and provide code examples from the documentation when required.',
    verbose=True)
if 'chat_engine' not in st.session_state.keys():
    st.session_state['chat_engine'] = db

if 'messages' not in st.session_state.keys():
    st.session_state['messages'] = [create_message(
        "assistant", "Ask me a question about Llama Index's open source library")]

query = st.chat_input("Enter question here...")

# Starting the chat session
if query:
    st.session_state.messages.append(create_message('user', query))

if st.session_state.messages[-1]['role'] == 'user':
    question = st.session_state.messages[-1]['content']
    with st.spinner('Thinking....'):
        response, sources, _ = qa(question, st.session_state.chat_engine)
        message_response = create_message('assistant', response)
        st.session_state.messages.append(message_response)
        metadata = process_metadata(sources)
        if metadata:
            message_metadata = create_message(
                'assistant', 'Here are the references for my data: \n'+'\n **** \n'.join(metadata))
            st.session_state.messages.append(message_metadata)

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])
