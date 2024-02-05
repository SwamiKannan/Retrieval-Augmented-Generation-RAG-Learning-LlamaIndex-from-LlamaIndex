from functools import cache
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceLLM
from llama_index.storage import StorageContext
from llama_index import VectorStoreIndex, ServiceContext
from dotenv import load_dotenv
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from sklearn.metrics import top_k_accuracy_score
from llama_index.memory import ChatMemoryBuffer
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms.openai import OpenAI
from llama_index.llms import LlamaCPP

import streamlit as st

load_dotenv()

# Parameters
# Parameters
MODEL = "E:\\models\\OpenHermes_Mistral_GGUF\\openhermes-2.5-mistral-7b.Q5_K_M.gguf"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CACHE_DIR = "E:\\Embeddings\\mistral"
tokenizer = HuggingFaceEmbedding(EMBED_MODEL, cache_folder=CACHE_DIR)

TEMPERATURE = 0


DATA_DIR = 'data_json'
# Create embeddings object
embed_model = HuggingFaceEmbedding(EMBED_MODEL, cache_folder=CACHE_DIR)

# load vectorstores
db = chromadb.PersistentClient(path="./llamassist_chroma")
chroma_collection = db.get_collection("llamaindex-assistant")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Setup llm
llm = OpenAI(model='gpt-3.5-turbo', temperature=TEMPERATURE)
# llm = LlamaCPP(
#     # You can pass in the URL to a GGML model to download it automatically
#     # model_url='https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/blob/main/openhermes-2.5-mistral-7b.Q5_K_M.gguf',
#     # optionally, you can set the path to a pre-downloaded model instead of model_url
#     model_path=MODEL,
#     temperature=0.1,
#     max_new_tokens=256,
#     # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
#     context_window=4096,
#     # kwargs to pass to __call__()
#     generate_kwargs={},
#     # kwargs to pass to __init__()
#     # set to at least 1 to use GPU
#     model_kwargs={"n_gpu_layers": -1},
#     # transform inputs into Llama2 format
#     messages_to_prompt=messages_to_prompt,
#     completion_to_prompt=completion_to_prompt,
#     verbose=True,
# )

# Contexts
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# setup index


@st.cache_resource(show_spinner=False)
def get_index(callback=False):
    if callback:
        debug_handler = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager(handlers=[debug_handler])
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model, callback_manager=callback_manager)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context, service_context=service_context)

    else:
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context, service_context=service_context
        )
    print('Data loaded... !')
    return index


def process_metadata(source_nodes):
    return [f"Reference {i+1}\nPage name: {item.metadata['name']}Link: {item.metadata['url']}" for i, item in enumerate(source_nodes)]


def qa(question, chat_engine, steps=False):
    response = chat_engine.chat(question)
    context = '\n'.join(
        [node.text for node in response.source_nodes]) if steps else None
    sources = response.source_nodes
    return response.response, sources, context


# Testing to see if the data is loaded correctly
if __name__ == "__main__":
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    index = get_index()
    chat = index.as_chat_engine(
        chat_mode='context',
        memory=memory,
        prompt='You are a librarian chatbot helping users answer questions based on the LlamaIndex documentation and provide code examples from the documentation when required.',
        verbose=True)

    question = "How to setup a Chroma index"
    # print('Retriever output:')
    # # responses = retriever.retrieve(question)
    # for i, response in enumerate(responses):
    #     print('Reference ', i+1)
    #     print(response.metadata['name'])
    #     print(response.metadata['url'], '\n')
    # print('\n\n\nLLM Output:')
    response, sources, _ = qa(question, chat)

    print(f'Answer: {response}\n')
    print('Sources')
    # for i, source in enumerate(sources):
    #     print(f'Reference {i+1}:')
    #     print(f'Page name: {source.metadata["name"]}')
    #     print(f'Page name: {source.metadata["url"]}')
    for item in process_metadata(sources):
        print(item)
