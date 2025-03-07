from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(persist_directory="emb", embedding_function=embeddings)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)

result = chain.invoke(
    {"query": "What is the interesting fact about the English language?"}
)

print(result["result"])
