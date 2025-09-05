from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

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
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

# Uncomment for debugging
# langchain.debug = True

load_dotenv()

# Initialize the LLM
llm = get_llm("local")

# Initialize the embeddings
embedding_type = "openai"
embeddings = get_embeddings(embedding_type)

# Initialize the Chroma database
db = Chroma(persist_directory=f"emb_{embedding_type}", embedding_function=embeddings)

# Initialize the custom retriever
retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

# Create a prompt template for the QA system
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Keep the answer informative but concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the document chain and retrieval chain using LCEL
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# Function to query the RAG system
def query_rag_system(question: str) -> dict:
    """
    Query the RAG system with a question and return the result.

    Args:
        question (str): The question to ask

    Returns:
        dict: Contains 'answer' and 'context' (source documents)
    """
    try:
        result = rag_chain.invoke({"input": question})
        return result
    except Exception as e:
        print(f"Error querying RAG system: {e}")
        return {
            "answer": "An error occurred while processing your question.",
            "context": [],
        }


# Main execution
if __name__ == "__main__":
    # Query the system
    question = "What is the interesting fact about the English language?"
    result = query_rag_system(question)

    # Print the answer
    print("Question:", question)
    print("\nAnswer:", result.get("answer", "No answer found"))

    # Print source documents if available
    if "context" in result and result["context"]:
        print("\n" + "=" * 50)
        print("SOURCE DOCUMENTS:")
        print("=" * 50)
        for i, doc in enumerate(result["context"], 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            if doc.metadata:
                print(f"Metadata: {doc.metadata}")
    else:
        print("\nNo source documents found.")
