from dotenv import load_dotenv
from llama_index import ServiceContext, StorageContext, Document, VectorStoreIndex
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.vector_stores import ChromaVectorStore
import chromadb

import os
import json

load_dotenv()

# Parameters
MODEL = 'gpt-3.5-turbo'
EMBEDDINGS = 'text-embedding-3-small'

CHUNK_SIZE = 500
CHUNK_OVERLAP = 20

TEMPERATURE = 0
EMBED_BATCH_SIZE = 100

DATA_DIR = '../data_json'

# Initializing the relevant objects / tools

# Text processing
embed_model = OpenAIEmbedding(model=EMBEDDINGS, embed_batch_size=100)

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95, embed_model=embed_model
)
embedding = OpenAIEmbedding()

# LLM
llm = OpenAI(model=MODEL, temperature=TEMPERATURE)

# VectorStores
db = chromadb.PersistentClient(path="./chroma_llamaindex")
chroma_collection = db.get_or_create_collection("llamaindex-assistant")
vectorstore = ChromaVectorStore(
    chroma_collection=chroma_collection, override=True)

# Initiating contexts
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model, node_parser=splitter)
storage_context = StorageContext.from_defaults(vector_store=vectorstore)


def process_file(file_name: str):
    print(file_name)
    with open(os.path.join(DATA_DIR, file_name), 'r', encoding='utf=8') as f:
        file = json.load(f)
        text = file['text']
        header = file['header']
        link = file['url']
        doc = Document(text=text, metadata={'name': header, 'url': link})
        return doc


def ingestion():
    documents = [process_file(filename)
                 for filename in os.listdir(os.path.join(DATA_DIR))]
    print('Documents created')
    docs = splitter.get_nodes_from_documents(documents)
    print(
        f'Documents split...{len(documents)} split into {len(docs)} documents')
    index = VectorStoreIndex.from_documents(
        documents=docs, storage_context=storage_context, service_context=service_context, show_progress=True)
    print('Ingestion complete...')
# document = process_file('TimescaleVectorStore.json')


if __name__ == "__main__":
    ingestion()
