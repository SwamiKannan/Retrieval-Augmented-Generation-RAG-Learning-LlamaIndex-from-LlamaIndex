from llama_index.vector_stores import ChromaVectorStore
import chromadb
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.storage import StorageContext
from llama_index import VectorStoreIndex, ServiceContext
from dotenv import load_dotenv
from llama_index.callbacks import LlamaDebugHandler, CallbackManager

import streamlit as st

load_dotenv()

# Parameters
MODEL = 'gpt-3.5-turbo'
EMBEDDINGS = 'text-embedding-3-small'
TEMPERATURE = 0

# Create embeddings object
embed_model = OpenAIEmbedding(model=EMBEDDINGS)
embedding = OpenAIEmbedding()

# load vectorstores
db = chromadb.PersistentClient(path="./llamassist_chroma")
chroma_collection = db.get_collection("llamaindex-assistant")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Setup llm
llm = OpenAI(model=MODEL, temperature=TEMPERATURE)

# Contexts
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# setup index


@st.cache_resource(show_spinner=False)
def get_index(callback=False):
    if callback:
        debug_handler = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager(handlers=[debug_handler])
        service_context = ServiceContext.from_defaults(
            callback_manager=callback_manager)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context, service_context=service_context)

    else:
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    print('Data loaded... !')
    return index


# Testing to see if the data is loaded correctly
if __name__ == "__main__":
    index = get_index()
    query_engine = index.as_query_engine()

    question = "What is a LlamaIndex query engine?"
    response = query_engine.query(question)
    print(response)
    query_engine = index.as_query_engine()
