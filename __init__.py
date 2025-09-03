"""
PyCode - AI and LangChain Learning Project

A collection of Python applications demonstrating various AI/ML concepts,
LangChain integrations, and chat applications.

Modules:
- chat_apps: Chat and conversation applications
- langchain_demos: LangChain learning examples and demonstrations
- api_integrations: Third-party API integrations (Ollama, Whisper)
- utilities: Helper scripts and utilities
- config: Centralized model configuration
"""

__version__ = "1.0.0"
__author__ = "Alex"

# Make key components easily accessible
from .config import get_llm, list_available_models, MODEL_CONFIGS

__all__ = [
    "get_llm",
    "list_available_models", 
    "MODEL_CONFIGS",
]