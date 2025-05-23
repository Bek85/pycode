# Requirements
# nltk==3.9.1
# numpy==1.26.4
# rank_bm25==0.2.2
# scikit-learn==1.5.2

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sample_documents import documents

nltk.download("punkt")
nltk.download("stopwords")


# Sample documents about sailing in Croatia

# Tokenize words
sample_document = documents[0]
tokenized_words = nltk.word_tokenize(sample_document)

print(tokenized_words)


# Tokenize sentences
sample_document = " ".join(documents[0:2])
tokenized_sentences = nltk.sent_tokenize(sample_document)

print(tokenized_sentences)
