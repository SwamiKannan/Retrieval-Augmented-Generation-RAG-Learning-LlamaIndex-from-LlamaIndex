from dotenv import load_dotenv
from llama_index import ServiceContext, StorageContext, Document, VectorStoreIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores import ChromaVectorStore
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt

import chromadb
import os
import json

load_dotenv()

# Parameters
MODEL = "E:\\models\\OpenHermes_Mistral_GGUF\\openhermes-2.5-mistral-7b.Q5_K_M.gguf"


EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CACHE_DIR = "E:\\Embeddings\\mistral"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 20

TEMPERATURE = 0

DATA_DIR = 'data_json'

# Initializing the relevant objects / tools

# Text processing
embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL, cache_folder=CACHE_DIR)

splitter = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=50)

# LLM
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url='https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/blob/main/openhermes-2.5-mistral-7b.Q5_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path="E:\\models\\OpenHermes_Mistral_GGUF\\openhermes-2.5-mistral-7b.Q5_K_M.gguf",
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# VectorStores
db = chromadb.PersistentClient(path="./llamassist_chroma")
chroma_collection = db.get_or_create_collection(
    "llamaindex-assistant")
vectorstore = ChromaVectorStore(
    chroma_collection=chroma_collection, override=True)

# Initiating contexts
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model, node_parser=splitter)
storage_context = StorageContext.from_defaults(vector_store=vectorstore)


def process_file(file_name: str):
    with open(os.path.join(DATA_DIR, file_name), 'r', encoding='utf=8') as f:
        file = json.load(f)
        text = file['text']
        header = file['header']
        link = file['url']
        doc = Document(text=text, metadata={'name': header, 'url': link})
        return doc


def ingestion():
    print(f'{len(os.listdir(os.path.join(DATA_DIR)))} files identified')
    documents = [process_file(filename)
                 for filename in os.listdir(os.path.join(DATA_DIR))]
    for doc in documents:
        if "chroma" in doc.text:
            print('Chroma document present:\t', doc.metadata['name'])
    print(f'{len(documents)} documents created')
    print('Initiating vector store update.....')
    index = VectorStoreIndex.from_documents(
        documents=documents, storage_context=storage_context, service_context=service_context, show_progress=True)
    print('Ingestion complete. Model is now ready to answer your queries....')


if __name__ == "__main__":
    ingestion()
