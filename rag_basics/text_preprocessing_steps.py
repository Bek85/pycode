import nltk
from sample_documents import documents


def process_text(text):
    processed_text = text.lower()
    processed_text = nltk.word_tokenize(processed_text)
    processed_text = [word for word in processed_text if word.isalnum()]

    stop_words = nltk.corpus.stopwords.words("english")
    processed_text = [word for word in processed_text if word not in stop_words]

    return processed_text


processed_text = [" ".join(process_text(doc)) for doc in documents]

# print(processed_text)
