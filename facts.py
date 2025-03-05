from dotenv import load_dotenv
import os
from langchain.document_loaders import TextLoader


load_dotenv()

loader = TextLoader("facts.txt")
documents = loader.load()

print(documents)
