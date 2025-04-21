from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

langchain.debug = True


load_dotenv()

# Initialize the LLM
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Initialize the embeddings
embeddings = OpenAIEmbeddings()

# Initialize the Chroma database
db = Chroma(persist_directory="emb", embedding_function=embeddings)

# Initialize the retriever
retriever = db.as_retriever()

# Initialize the retriever
# retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

# Initialize the chain
chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, verbose=True
)

# Invoke the chain
result = chain.invoke(
    {"query": "What is the interesting fact about the English language?"}
)

print(result["result"])
