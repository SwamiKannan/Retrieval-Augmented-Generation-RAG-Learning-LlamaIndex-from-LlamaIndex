from dotenv import load_dotenv
from llama_index import ServiceContext, StorageContext, Document, VectorStoreIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceLLM
from llama_index.vector_stores import ChromaVectorStore
import chromadb
import os
import json

# load_dotenv()

# Parameters
MODEL = 'teknium/OpenHermes-2.5-Mistral-7B'
MODEL_LOCAL = "E:\\models\\Mistral-7b"
EMBEDDINGS = 'sentence-transformers/all-MiniLM-L6-v2'
CACHE_DIR = "E:\\Embeddings\\all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 20

TEMPERATURE = 0
EMBED_BATCH_SIZE = 100

DATA_DIR = 'data_json'

# Initializing the relevant objects / tools

# Text processing
embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDINGS, cache_folder=CACHE_DIR)

splitter = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=50)

# LLM
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": TEMPERATURE, "do_sample": False},
    tokenizer_name=EMBEDDINGS,
    model_name=MODEL,
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
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
