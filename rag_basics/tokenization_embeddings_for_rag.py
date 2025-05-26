# Requirements
# faiss-cpu==1.9.0.post1
# numpy==1.26.4
# torch==2.5.1+cu121
# transformers==4.46.3

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sample_dataset import documents

# Initialize the tokenizer and model for generating embeddings
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
)
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")


# Create a function to tokenize input and generate its embeddings


def generate_embeddings(text, model, tokenizer):
    # Tokenize the input text, return tensors in pytorch, apply padding and truncation
    tokens = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Disable gradient calculation
    with torch.no_grad():
        # Pass the tokenized inputs through the model to the last state
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state

        # Extract the embeddings from the model
        embeddings = embeddings.mean(dim=1)

    return embeddings


# Initiliaze a list to store the embeddings

document_embeddings = []

# Loop through the documents and generate embeddings
for doc in documents:
    doc_embedding = generate_embeddings(doc, model, tokenizer)
    document_embeddings.append(doc_embedding)

# Concatenate all embeddings into a pytorch tensor, move it to the CPU, and convert to numpy array
document_embeddings = torch.cat(document_embeddings).cpu().numpy()

# print(document_embeddings)
