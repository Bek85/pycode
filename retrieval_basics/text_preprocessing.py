import nltk
from sample_documents import documents


# A function to process the text which lowercases, tokenizes, removes non-alphanumeric characters and stop words
def process_text(text: str) -> list[str]:

    processed_text = text.lower()

    processed_text = nltk.word_tokenize(processed_text)
    processed_text = [word for word in processed_text if word.isalnum()]

    stop_words = nltk.corpus.stopwords.words("english")
    processed_text = [word for word in processed_text if word not in stop_words]

    return processed_text


def process_text_boolean(text: str) -> list[str]:

    processed_text = text.lower()

    processed_text = nltk.word_tokenize(processed_text)
    processed_text = [word for word in processed_text if word.isalnum()]

    stop_words = set(nltk.corpus.stopwords.words("english")) - {"and", "or", "not"}
    processed_text = [word for word in processed_text if word not in stop_words]

    return processed_text


# Process the documents and join them into a single string
processed_text = [" ".join(process_text(doc)) for doc in documents]

# print(processed_text)
