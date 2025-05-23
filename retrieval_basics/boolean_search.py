# Boolean search
from text_preprocessing import process_text_boolean
from sample_documents import documents


def boolean_search(query: str, documents: list[str]):

    query_tokens = process_text_boolean(query)

    results = []

    for doc_id, doc in enumerate(documents):
        doc_tokens = set(process_text_boolean(doc))
        include_doc = False

        for i, token in enumerate(query_tokens):
            # If the token is "and" and there is a next token, include the document if the next token is in the document
            if token == "and" and i + 1 < len(query_tokens):
                include_doc = include_doc and (query_tokens[i + 1] in doc_tokens)
            # If the token is "or" and there is a next token, include the document if the next token is in the document
            elif token == "or" and i + 1 < len(query_tokens):
                include_doc = include_doc or (query_tokens[i + 1] in doc_tokens)
            # If the token is "not" and there is a next token, include the document if the next token is not in the document
            elif token == "not" and i + 1 < len(query_tokens):
                if query_tokens[i + 1] in doc_tokens:
                    include_doc = False
                    break

            else:
                # for normal tokens, check if the document should be included
                include_doc = token in doc_tokens or include_doc

        if include_doc:
            results.append((doc_id, doc))

    return results


# Define search query

query = "Sailing in Split or Hvar."

# Perform boolean search

boolean_results = boolean_search(query, documents)

# Print results
for doc_id, doc in boolean_results:
    print(f"Document {doc_id}:")
    print(doc)
    print("\n")
