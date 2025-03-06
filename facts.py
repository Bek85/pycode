from dotenv import load_dotenv
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings


load_dotenv()

# embeddings = OpenAIEmbeddings()

# emb = embeddings.embed_query("Hello, world!")

# print(emb)

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

loader = TextLoader("facts.txt")
documents = loader.load_and_split(text_splitter)


for doc in documents:
    print(doc.page_content)
    print("\n")
