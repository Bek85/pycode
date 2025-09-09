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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

# langchain.debug = True


load_dotenv()

# Initialize the LLM
llm = get_llm("remote")

# Initialize the embeddings
embedding_type = "openai"
embeddings = get_embeddings(embedding_type)

# Initialize the Chroma database
db = Chroma(persist_directory=f"emb_{embedding_type}", embedding_function=embeddings)

# Initialize the retriever
# retriever = db.as_retriever()

# Initialize the retriever
retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

# Initialize the chain with modern LangChain LCEL approach
# Create a chat prompt template for better prompt handling
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the document chain and retrieval chain using modern LCEL
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

# Invoke the chain
# Note: Modern retrieval chain uses "input" instead of "query"
result = chain.invoke(
    {"input": "What is the interesting fact about the English language?"}
)

# Modern chain returns "answer" instead of "result"
print(result["answer"])
