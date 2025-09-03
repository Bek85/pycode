"""Model configurations for LangChain LLMs"""

from langchain.chat_models import init_chat_model

# Model configurations
MODEL_CONFIGS = {
    "local": {
        "model": "ProkuraturaAI",
        "model_provider": "openai",
        "openai_api_base": "http://172.18.35.123:8000/v1",
    },
    "remote": {
        "model": "gpt-4o-mini",
        "model_provider": "openai",
    },
    # Easy to add more configurations
    "claude": {
        "model": "claude-3-sonnet-20240229",
        "model_provider": "anthropic",
    },
}


def get_llm(model_type: str):
    """Get LLM instance based on model type

    Args:
        model_type: One of 'local', 'remote', 'claude'

    Returns:
        Initialized LLM instance

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

    config = MODEL_CONFIGS[model_type]
    return init_chat_model(**config)


def list_available_models():
    """List all available model configurations"""
    return list(MODEL_CONFIGS.keys())
