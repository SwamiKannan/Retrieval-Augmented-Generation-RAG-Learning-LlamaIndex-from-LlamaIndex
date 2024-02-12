from query import get_index
from ingestion import ingestion
import nest_asyncio
from llama_index.memory import ChatMemoryBuffer
import argparse
nest_asyncio.apply()

'''
args parse
callbacks
show_perf: If you want all the steps to be followed
'''
callback = True
show_perf = False

parser =argparse.ArgumentParser()

parser.add_argument('question')
args = parser.parse_args()
question = args['question']


def process_metadata(metadata):
    return [f"Reference {i+1}\nPage name: {item['name']}Link: {item['url']}" for i, item in enumerate(metadata.values())]


index = get_index(callback)
query_engine = index.as_query_engine()


def qa(question, chat_engine, steps=False):
    response_query = query_engine.query(question)
    response = chat_engine.chat(question).response
    metadata = response_query.metadata
    context = '\n'.join(
        [node.text for node in response.source_nodes]) if steps else None
    metadata_sources = process_metadata(metadata) if metadata else None
    return response_query.response, metadata_sources, context


if __name__ == "__main__":
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    chat_engine = index.as_query_engine()
    response = query_engine.query(question)
    print('\n'+'Answer:')

    # if show_perf:
    #     print('\n'+'Performance analysis:')
    #     print(context)
    # print('\n'+'References:')
    # process_metadata(metadata)
