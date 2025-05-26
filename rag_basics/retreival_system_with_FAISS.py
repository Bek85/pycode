import faiss
from tokenization_embeddings_for_rag import document_embeddings, generate_embeddings
from transformers import AutoTokenizer, AutoModel
from sample_dataset import documents

# Initialize the tokenizer and model for generating embeddings
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
)
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Initialize a FAISS index

index = faiss.IndexFlatL2(document_embeddings.shape[1])

index.add(document_embeddings)

# Retrieval -> create a function to retrieve information


def retrieve(query, tokenizer, model, index, documents, top_k=3):
    query_embedding = generate_embeddings(query, model, tokenizer)

    # Search for the top k most similar documents using the index distances
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the documents at the indices
    retrieved_documents = [documents[i] for i in indices[0]]

    return retrieved_documents, distances


# Test the retrieval function
# query = "What is the capital of Germany?"
# retrieved_documents, distances = retrieve(query, tokenizer, model, index, documents)
# print(retrieved_documents)
# print(distances)
