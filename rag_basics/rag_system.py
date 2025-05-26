from retreival_system_with_FAISS import retrieve
from generative_system import generate_text
from sample_dataset import documents
import faiss
from tokenization_embeddings_for_rag import document_embeddings
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from check_relevance import is_relevant

# Initialize the tokenizer and model for generation
gen_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gen_tokenizer.pad_token = gen_tokenizer.eos_token
gen_model = AutoModelForCausalLM.from_pretrained("gpt2")


# Initialize the FAISS index
retrieval_index = faiss.IndexFlatL2(document_embeddings.shape[1])
retrieval_index.add(document_embeddings)

# Initialize the tokenizer and model for retrieval
retrieval_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
)
retrieval_model = AutoModel.from_pretrained(
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
)


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

    # Discard all documents that do not meet the relevance criteria
    relevant_docs = [
        doc
        for doc, distance in zip(retrieved_docs, distances[0])
        if is_relevant(distance, 40)
    ]

    # Add a message if no retrieved_doc is relevant
    if not relevant_docs:
        return "No relevant documents found"

    context = " ".join(relevant_docs)

    generated_answer = generate_text(
        context, query, gen_model, gen_tokenizer, max_length=100
    )

    return generated_answer


# Test the RAG pipeline
# query = "What is the capital of Germany?"

# generated_answer = rag_pipeline(
#     query,
#     retrieval_tokenizer,
#     retrieval_model,
#     retrieval_index,
#     gen_model,
#     gen_tokenizer,
#     documents,
#     top_k=3,
# )

# Test the RAG pipeline with multiple queries

queries = [
    "What is the capital of Germany?",
    "What is Berlin famous for?",
    "Who discovered the Americas?",
    "Who is the most famous person in the world?",
]


for query in queries:
    generated_answer = rag_pipeline(
        query,  # Query to answer
        retrieval_tokenizer,  # Tokenizer for retrieval
        retrieval_model,  # Model for retrieval
        retrieval_index,  # FAISS index
        gen_model,  # Model for generation
        gen_tokenizer,  # Tokenizer for generation
        documents,  # List of documents to retrieve from
        top_k=3,  # Number of documents to retrieve
    )
    print(f"Query: {query}\nAnswer: {generated_answer}\n")


# for query in queries:
#     retrieved_docs, distances = retrieve(
#         query, retrieval_tokenizer, retrieval_model, retrieval_index, documents, top_k=3
#     )
#     print(f"Query: {query}\nRetrieved docs: {retrieved_docs}\nDistances: {distances}\n")
