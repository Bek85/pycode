from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from text_preprocessing_steps import processed_text

# Index the documents with TF-IDF

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(processed_text)

# Sample query
query = "Croatia is a beautiful country"

# Try the sample query in our documents
query_vector = vectorizer.transform([query])

# print(query_vector.T.toarray())

results = np.dot(tfidf_matrix, query_vector.T).toarray()

print(f"The results are: {results}")
