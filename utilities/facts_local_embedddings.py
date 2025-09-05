from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.embeddings import get_embeddings


def main(embedding_type="local"):
    # Load environment variables
    load_dotenv()

    # Initialize embeddings
    embeddings = get_embeddings(embedding_type)

    # Use RecursiveCharacterTextSplitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
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

    # Use directory based on embedding type
    persist_dir = f"emb_{embedding_type}"

    # Create vector database with embeddings
    db = Chroma.from_documents(
        documents, embedding=embeddings, persist_directory=persist_dir
    )

    # Search for similar documents
    query = "What is an interesting fact about the English language?"
    results = db.similarity_search(query, k=3)

    print(f"\nQuery: {query}")
    print("=" * 50)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("-" * 20)
        print(result.page_content)

        if result.metadata:
            print(f"Metadata: {result.metadata}")


def search_existing_db(query: str, embedding_type="local"):
    """Function to search an existing database without recreating it"""
    load_dotenv()

    embeddings = get_embeddings(embedding_type)

    # Load existing database
    persist_dir = f"emb_{embedding_type}"
    try:
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        results = db.similarity_search(query, k=3)

        print(f"\nQuery: {query}")
        print("=" * 50)

        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print("-" * 20)
            print(result.page_content)
    except Exception as e:
        print(f"Error loading database: {e}")
        print("Make sure to run main() first to create the database")


if __name__ == "__main__":
    # Run the main pipeline with local embeddings (default)
    main("local")

    # Example of using OpenAI embeddings
    # main("openai")

    # Example of searching existing database
    print("\n" + "=" * 60)
    print("SEARCHING EXISTING DATABASE")
    print("=" * 60)
    search_existing_db("Tell me about strawberry", "local")
