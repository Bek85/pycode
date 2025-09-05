from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.embeddings import get_embeddings


def main(embedding_type="openai"):
    # Load environment variables
    load_dotenv()

    # Verify API key is loaded if using OpenAI embeddings
    if embedding_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize embeddings
    embeddings = get_embeddings(embedding_type)

    # Use RecursiveCharacterTextSplitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Larger chunks often work better
        chunk_overlap=50,  # Some overlap helps maintain context
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    # Load and split documents
    try:
        loader = TextLoader("facts.txt", encoding="utf-8")
        documents = loader.load_and_split(text_splitter)
        print(f"Loaded {len(documents)} document chunks")
    except FileNotFoundError:
        print("Error: facts.txt file not found")
        return

    # Create or load existing vector database
    # Note: Chroma 0.4.x automatically persists documents
    persist_dir = f"emb_{embedding_type}"
    db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_dir)

    # Search for similar documents
    query = "What is an interesting fact about the English language?"
    results = db.similarity_search(query, k=1)  # Get top 1 result

    print(f"\nQuery: {query}")
    print("=" * 50)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("-" * 20)
        print(result.page_content)

        # Show metadata if available
        if result.metadata:
            print(f"Metadata: {result.metadata}")


def search_existing_db(query: str, embedding_type="openai"):
    """Function to search an existing database without recreating it"""
    load_dotenv()

    embeddings = get_embeddings(embedding_type)

    # Load existing database
    persist_dir = f"emb_{embedding_type}"
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    results = db.similarity_search(query, k=3)

    print(f"\nQuery: {query}")
    print("=" * 50)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("-" * 20)
        print(result.page_content)


if __name__ == "__main__":
    # Run the main pipeline with OpenAI embeddings (default)
    main("openai")

    # Example of using local embeddings
    # main("local")

    # Example of searching existing database
    # search_existing_db("Tell me about language facts", "openai")
