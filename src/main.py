from query import get_index
from ingestion import ingestion
import nest_asyncio
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata

nest_asyncio.apply()

'''
args parse
callbacks
describe_steps: If you want all the steps to be followed
'''
callback = True
steps = False


def process_metadata(metadata):
    for i, item in enumerate(metadata.values()):
        print(f'Reference {i+1}')
        print(f"Page name: {item['name']}")
        print(f"Link: {item['url']}")
        print('***********************************************************')


index = get_index(callback)
query_engine = index.as_query_engine()


def qa(question, steps):
    response = query_engine.query(question)
    print('\n')
    print(response)
    print('\n')
    metadata = response.metadata
    context = '\n'.join(
        [node.text for node in response.source_nodes]) if steps else None
    return response, metadata, context


if __name__ == "__main__":
    question = "What is a LlamaIndex query engine?"
    response, metadata, context = qa(question, steps)
    print('\n'+'Answer:')
    print(response)
    if steps:
        print('\n'+'Context:')
        print(context)
    print('\n'+'References:')
    process_metadata(metadata)
