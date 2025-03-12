from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# import langchain

# langchain.debug = True


load_dotenv()

chat = ChatOpenAI(model="gpt-4o-mini")

embeddings = OpenAIEmbeddings()

db = Chroma(persist_directory="emb", embedding_function=embeddings)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=retriever, verbose=True
)

result = chain.invoke(
    {"query": "What is the interesting fact about the English language?"}
)

print(result["result"])
