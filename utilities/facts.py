from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os


def main():
    # Load environment variables
    load_dotenv()

    # Verify API key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

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
    db = Chroma.from_documents(documents, embedding=embeddings, persist_directory="emb")

    # Search for similar documents
    query = "What is an interesting fact about the English language?"
    results = db.similarity_search(query, k=3)  # Get top 3 results

    print(f"\nQuery: {query}")
    print("=" * 50)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("-" * 20)
        print(result.page_content)

        # Show metadata if available
        if result.metadata:
            print(f"Metadata: {result.metadata}")


def search_existing_db(query: str):
    """Function to search an existing database without recreating it"""
    load_dotenv()

    embeddings = OpenAIEmbeddings()

    # Load existing database
    db = Chroma(persist_directory="emb", embedding_function=embeddings)

    results = db.similarity_search(query, k=3)

    print(f"\nQuery: {query}")
    print("=" * 50)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("-" * 20)
        print(result.page_content)


if __name__ == "__main__":
    # Run the main pipeline
    main()

    # Example of searching existing database
    # search_existing_db("Tell me about language facts")
