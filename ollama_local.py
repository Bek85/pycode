from dotenv import load_dotenv
import requests
import json

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

load_dotenv()


# Direct API approach
def generate_with_direct_api(prompt, model_name="qwen3:8b"):
    """Generate text using direct API calls to Ollama"""
    api_url = (
        "http://127.0.0.1:11434/api/generate"  # Using generate endpoint instead of chat
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,  # Ensure we get a complete response
    }

    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            # Print debug info
            print(f"DEBUG - Response keys: {data.keys()}")
            return data.get("response", "No response field found")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error in API call: {str(e)}"


# LangChain approach
def generate_with_langchain(prompt, model_name="qwen3:8b"):
    """Generate text using LangChain's Ollama integration"""
    ollama_llm = OllamaLLM(model=model_name, base_url="http://127.0.0.1:11434")
    result = ollama_llm.invoke(prompt)
    return result


# Chat model approach with LangChain
def chat_with_langchain(prompt, model_name="qwen3:8b"):
    """Chat with Ollama models using LangChain's chat interface"""
    chat_model = ChatOllama(model=model_name, base_url="http://127.0.0.1:11434")
    message = HumanMessage(content=prompt)
    response = chat_model.invoke([message])
    return response.content


if __name__ == "__main__":
    user_input = input("Enter a prompt: ")

    print("\n--- Direct API Response ---")
    direct_result = generate_with_direct_api(user_input)
    print(direct_result)

    print("\n--- LangChain Response ---")
    langchain_result = generate_with_langchain(user_input)
    print(langchain_result)

    print("\n--- LangChain Chat Response ---")
    chat_result = chat_with_langchain(user_input)
    print(chat_result)
