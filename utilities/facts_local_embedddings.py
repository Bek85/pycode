from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated
from langchain_chroma import Chroma  # Updated
import os
import shutil


def main():
    # Load environment variables
    load_dotenv()

    # Initialize local embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

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

    # Use a different directory for local embeddings
    persist_dir = "emb_local"

    # Create vector database with local embeddings
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


def search_existing_db(query: str, persist_dir="emb_local"):
    """Function to search an existing database without recreating it"""
    load_dotenv()

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load existing database
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
    # Run the main pipeline
    main()

    # Example of searching existing database
    print("\n" + "=" * 60)
    print("SEARCHING EXISTING DATABASE")
    print("=" * 60)
    search_existing_db("Tell me about strawberry")
