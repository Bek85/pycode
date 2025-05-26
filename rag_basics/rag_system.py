from retreival_system_with_FAISS import retrieve
from generative_system import generate_text, generative_model, generative_tokenizer
from sample_dataset import documents
import faiss
from tokenization_embeddings_for_rag import document_embeddings
from retreival_system_with_FAISS import (
    tokenizer as retrieval_tokenizer,
    model as retrieval_model,
)

# Initialize the tokenizer and model for generation
gen_tokenizer = generative_tokenizer
gen_model = generative_model

# Initialize the FAISS index
retrieval_index = faiss.IndexFlatL2(document_embeddings.shape[1])

retrieval_index.add(document_embeddings)


# Define RAG function which integrates retrieval and generation


def rag_pipeline(
    query,
    retrieval_tokenizer,
    retrieval_model,
    retrieval_index,
    gen_model,
    gen_tokenizer,
    documents,
    top_k,
):
    retrieved_docs, distances = retrieve(
        query, retrieval_tokenizer, retrieval_model, retrieval_index, documents, top_k
    )

    context = " ".join(retrieved_docs)

    generated_answer = generate_text(
        context, query, gen_model, gen_tokenizer, max_length=100
    )

    return generated_answer


# Test the RAG pipeline
query = "What is the capital of Germany?"

generated_answer = rag_pipeline(
    query,
    retrieval_tokenizer,
    retrieval_model,
    retrieval_index,
    gen_model,
    gen_tokenizer,
    documents,
    top_k=3,
)

print(generated_answer)
