# Learn LlamaIndex using a LlamaIndex helper
<p align="center">
  <img src = "https://github.com/SwamiKannan/Learning-LlamaIndex-from-LlamaIndex/blob/main/images/image%203.jpg", width = 60%>
</p>

## Introduction

I was trying to learn LlamaIndex and was getting a little frustrated by the syntax since it was different from LangChain (with storage context and service context). Hence, to start it off, I decided to first build a RAG and a chat assistant that helps me navigate the documentation.

## Requirements
Running this code as-is will require you to have an OpenAI account and an API key.
### Get OpenAI key
After creating an account <a href="https://platform.openai.com/signup/"> here </a>, get your API key as described <a href="https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key"> here </a>
### Store the OpenAI key in the env file
1. Create a text file using Notepad and name it ".env"
2. Open it and enter:
```
OPENAI_API_KEY= <the API key you just received>
```
3. Save it in the src folder
4. The code will automatically pick up the API key when running

#### Note: Please do not share this API key with anyone. They will directly be able to use your account and your credits for their usage. 

## Usage
### 1. Scraping the LlamaIndex website
```
cd src
python scrape_json.py
```

### 2. Ingest the data into a vectordb:
There are three vectordbs that you can use for storing your embeddings:<br>
a. Milvus
```
python ingestion-milvus.py
```
b.Pinecone
```
python ingestion-pinecone.py
```
c. Chromadb (Rest of the process is based on Chromadb"
```
python ingestion.py
```
### 3. Querying the vectordb
There are two ways to query the database]
#### Main.py


#### Using the app
```
streamlit app.py
```
## Demo
<p align = "center">
  
https://github.com/SwamiKannan/Learning-LlamaIndex-from-LlamaIndex/assets/65940566/d6ba5aae-45f6-4847-bf7e-89d0c332de96

</p>

<sub>
<b>Image credit:</b> <br/> <a href="https://www.segmind.com/models/sdxl1.0-txt2img"> Segmind </a><br>
Prompt: cinematic film still, 4k, realistic, ((cinematic photo:1.3)) of a person being overwhelmed walking through an infinite hallway of a digital library and looking at documentation, Fujifilm XT3, long shot, ((low light:1.4)), wide angle lens, landscape perspective, neon, somber, shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy
