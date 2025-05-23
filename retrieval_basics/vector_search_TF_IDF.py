from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from rag_basics.text_preprocessing import process_text
from sample_documents import documents


# Index the documents with TF-IDF
processed_text = [" ".join(process_text(doc)) for doc in documents]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(processed_text)

# Sample query
query = "Croatia is a beautiful country"

# Try the sample query in our documents
query_vector = vectorizer.transform([query])

# print(query_vector.T.toarray())

results = np.dot(tfidf_matrix, query_vector.T).toarray()

# print(f"The results are: {results}")


# A function to search the documents with TF-IDF
def search(query, vectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])
    results = np.dot(tfidf_matrix, query_vector.T).toarray()
    return results


# Sort and display the function
search_results = search(query, vectorizer, tfidf_matrix)
sorted_results = sorted(enumerate(search_results), key=lambda x: x[1][0], reverse=True)

# Iterate through the results and display the documents
for i, score in enumerate(sorted_results):
    print(f"Document {i+1}: {documents[i]}")
    print(f"The score is: {score[1][0]:.3f}")
