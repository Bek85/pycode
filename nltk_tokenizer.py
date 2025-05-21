import nltk

nltk.download("punkt")  # Download the tokenizer data

# Sample text
text = "I'm a software engineer at Google and I'm working on a new project. I'm also a student at MIT."

# Word Tokenization
word_tokens = nltk.word_tokenize(text)
print("Word Tokens:", word_tokens)

# Sentence Tokenization
sentence_tokens = nltk.sent_tokenize(text)
print("Sentence Tokens:", sentence_tokens)


# Preprocess the documents
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum()]


# Example documents
documents = [
    "I love programming in Python",
    "I also like to code in Java",
    "I'm a software engineer at Google",
]

processed_documents = [" ".join(preprocess_text(doc)) for doc in documents]
print("Processed Documents:", processed_documents)
