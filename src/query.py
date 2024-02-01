from llama_index.vector_stores import ChromaVectorStore
import chromadb
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.storage import StorageContext
from llama_index import VectorStoreIndex
from dotenv import load_dotenv

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


def get_index():
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    print('Index setup')
    return index
