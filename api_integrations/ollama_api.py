from dotenv import load_dotenv
import requests
import json

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

load_dotenv()

base_url = "http://127.0.0.1:11434"
model_name = "qwen3:4b"

# json_prompt must be be passed to OllamaLLM and ChatOllama for structured output
json_prompt = """
Generate a JSON object containing 3 fictional users.
Each user must have a firstName, lastName, birthdate (in YYYY-MM-DD format), and a country.
Return that data in JSON format.
"""

json_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "firstName": {
                "type": "string",
                "description": "The first name of the user",
            },
            "lastName": {"type": "string", "description": "The last name of the user"},
            "birthdate": {
                "type": "string",
                "description": "The birthdate of the user in YYYY-MM-DD format",
            },
            "country": {"type": "string", "description": "The country of the user"},
        },
        "required": ["firstName", "lastName", "birthdate", "country"],
    },
}


# Direct API approach
def generate_with_direct_api(prompt, model_name=model_name):
    """Generate text using direct API calls to Ollama"""
    api_url = f"{base_url}/api/generate"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model_name,
        "prompt": prompt,
        "format": json_schema,
        "stream": False,  # Ensure we get a complete response
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()

        # Print debug info
        print(f"DEBUG - Response keys: {data.keys()}")

        # Extract the relevant part of the response
        if "response" in data:
            return data["response"]
        else:
            return f"No response key found. Full response: {data}"
    except requests.exceptions.RequestException as e:
        print(f"Error contacting Ollama API: {str(e)}")
        return None


# LangChain approach
def generate_with_langchain(prompt, model_name=model_name):
    """Generate text using LangChain's Ollama integration"""
    ollama_llm = OllamaLLM(model=model_name, base_url=base_url)
    result = ollama_llm.invoke(prompt, json_schema)
    return result


# Chat model approach with LangChain
def chat_with_langchain(prompt, model_name=model_name):
    """Chat with Ollama models using LangChain's chat interface"""
    chat_model = ChatOllama(model=model_name, base_url=base_url)
    message = HumanMessage(content=prompt)
    response = chat_model.invoke([message], json_schema)
    return response.content


if __name__ == "__main__":
    user_input = input("Enter a prompt: ")

    print("\n--- Direct API Response ---")
    direct_result = generate_with_direct_api(json_prompt)
    print(direct_result)

    print("\n--- LangChain Response ---")
    langchain_result = generate_with_langchain(json_prompt)
    print(langchain_result)

    print("\n--- LangChain Chat Response ---")
    chat_result = chat_with_langchain(json_prompt)
    print(chat_result)
