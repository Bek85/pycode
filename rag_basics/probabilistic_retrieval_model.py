# Requirements
# rank_bm25==0.2.2
# numpy==1.26.4

from rank_bm25 import BM25Okapi
from text_preprocessing import process_text
from sample_documents import documents
import numpy as np

# Tokenize each normalized document
tokenized_docs = [process_text(doc) for doc in documents]

# Initialize the BM25 model
bm25 = BM25Okapi(tokenized_docs)

# Query
query = "Top 10 things to do in Croatia"


# Define probabilistic BM25 search function
def bm25_search(query: str, model: BM25Okapi) -> list[int]:
    query_tokens = process_text(query)
    doc_scores = model.get_scores(query_tokens)
    return doc_scores


bm25_results = bm25_search(query, bm25)

# Sort and display the results
sorted_bm25_results = np.argsort(bm25_results)[::-1]


for doc_idx in sorted_bm25_results[:10]:
    print(f"Score: {bm25_results[doc_idx]:.3f}")
    print(documents[doc_idx])
    print("\n")
