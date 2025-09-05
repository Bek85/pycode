"""Embedding configurations for different providers"""

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Embedding configurations
EMBEDDING_CONFIGS = {
    "openai": {
        "class": OpenAIEmbeddings,
        "kwargs": {},
    },
    "local": {
        "class": HuggingFaceEmbeddings,
        "kwargs": {
            "model_name": "all-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True},
        },
    },
    # Easy to add more configurations
    "openai_large": {
        "class": OpenAIEmbeddings,
        "kwargs": {
            "model": "text-embedding-3-large",
        },
    },
    "local_multilingual": {
        "class": HuggingFaceEmbeddings,
        "kwargs": {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True},
        },
    },
}


def get_embeddings(embedding_type: str):
    """Get embeddings instance based on embedding type
    
    Args:
        embedding_type: One of 'openai', 'local', 'openai_large', 'local_multilingual'
        
    Returns:
        Initialized embeddings instance
        
    Raises:
        ValueError: If embedding_type is not recognized
    """
    if embedding_type not in EMBEDDING_CONFIGS:
        available = ", ".join(EMBEDDING_CONFIGS.keys())
        raise ValueError(f"Unknown embedding type: {embedding_type}. Available: {available}")
    
    config = EMBEDDING_CONFIGS[embedding_type]
    return config["class"](**config["kwargs"])


def list_available_embeddings():
    """List all available embedding configurations"""
    return list(EMBEDDING_CONFIGS.keys())