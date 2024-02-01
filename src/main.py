from query import get_index
from ingestion import ingestion
import nest_asyncio
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata

nest_asyncio.apply()


def process_metadata(metadata):
    for i, item in enumerate(metadata.values()):
        print(f'Reference {i+1}')
        print(f"Page name: {item['name']}")
        print(f"Link: {item['url']}")
        print('***********************************************************')


index = get_index()
query_engine = index.as_query_engine()


def qa(question):
    response = query_engine.query(question)
    print(response)
    print('\n')
    metadata = response.metadata
    print('References:')
    process_metadata(metadata)
    return response, metadata


if __name__ == "__main__":
    question = "What is a LlamaIndex query engine?"
    response, metadata = qa(question)
