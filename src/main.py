from query import get_index
from ingestion import ingestion


index = get_index()
query_engine = index.as_query_engine()

question = "What is a LlamaIndex query engine?"
response = query_engine.query(question)
print(response)
