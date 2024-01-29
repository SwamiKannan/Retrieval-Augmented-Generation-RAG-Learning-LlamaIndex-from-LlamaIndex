from dotenv import load_dotenv
from llama_index.readers import SimpleDirectoryReader
from llama_index import Document
import os
import json

load_dotenv()

loader = SimpleDirectoryReader(input_dir="samples")


def process_file(file_name: str):
    with open(os.path.join('samples', file_name), 'r', encoding='utf=8') as f:
        file = json.load(f)
        text = file['text']
        header = file['header']
        link = file['url']
        doc = Document(text=text, metadata={'name': header, 'url': link})
        return doc


document = process_file('TimescaleVectorStore.json')
# print(document,'\n')
print(document.metadata, '\n')
print(document.text, '\n')
# documents = loader.load_data()
# docs = []
# for filename in os.listdir(os.path.join('samples')):
#     with open(os.path.join('data', filename), encoding='utf-8') as f:
#         file = f.read()
#         doc = Document(text=file, metadata={"filename": filename})
#         docs.append(doc)
#         print('Is metadata correct?\t', doc.metadata['filename']==filename)
# print('Simple Directory Reader', type(documents))
# print('Simple Directory Reader entry', type(documents[0]))

# print('Document', type(docs))
# print('Document', type(docs[0]))
