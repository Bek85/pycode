from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Handle both direct execution and module import
try:
    from ..config import get_llm
    from ..config.embeddings import get_embeddings
except ImportError:
    # Fallback for direct execution
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_llm
    from config.embeddings import get_embeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

# langchain.debug = True


load_dotenv()

# Initialize the LLM
llm = get_llm("local")

# Initialize the embeddings
embeddings = get_embeddings("openai")

# Initialize the Chroma database
db = Chroma(persist_directory="emb_openai", embedding_function=embeddings)

# Initialize the retriever
# retriever = db.as_retriever()

# Initialize the retriever
retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

# Initialize the chain
# chain_type="stuff" is a type of chain that uses the retriever to retrieve the most relevant documents and then use the LLM to answer the question

chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, verbose=True
)

# Invoke the chain
result = chain.invoke(
    {"query": "What is the interesting fact about the English language?"}
)

print(result["result"])
