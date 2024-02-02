from query import get_index
from ingestion import ingestion
import nest_asyncio


nest_asyncio.apply()

'''
args parse
callbacks
show_perf: If you want all the steps to be followed
'''
callback = True
show_perf = False


def process_metadata(metadata):
    for i, item in enumerate(metadata.values()):
        print(f'Reference {i+1}')
        print(f"Page name: {item['name']}")
        print(f"Link: {item['url']}")
        print('***********************************************************')


index = get_index(callback)
query_engine = index.as_query_engine()


def qa(question, steps=False, query_engine=query_engine):
    response = query_engine.query(question)
    metadata = response.metadata
    context = '\n'.join(
        [node.text for node in response.source_nodes]) if steps else None
    return response, metadata, context


if __name__ == "__main__":
    question = "What is a LlamaIndex query engine?"
    response, metadata, context = qa(question, show_perf)
    print('\n'+'Answer:')
    print(response)
    if show_perf:
        print('\n'+'Performance analysis:')
        print(context)
    print('\n'+'References:')
    process_metadata(metadata)
