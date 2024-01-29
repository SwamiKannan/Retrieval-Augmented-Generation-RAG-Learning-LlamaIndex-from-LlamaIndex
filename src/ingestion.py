from dotenv import load_dotenv
import os

load_dotenv()

print(os.environ['OPENAI_API_KEY'])
print(os.environ['PINECONE_ENVIRONMENT'])
print(os.environ['PINECONE_API_KEY'])
